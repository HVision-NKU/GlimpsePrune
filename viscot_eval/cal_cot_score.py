import os
import re
import json
import argparse
import warnings
import time
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset
from torch.utils.data import DataLoader
from accelerate.utils import set_seed
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from utils import extract_one_bbox_from_str, cal_paired_ious

SYS_PROMPT = """
You are responsible for proofreading the answers, you need to give a score to the model's answer by referring to the standard answer, based on the given question. The full score is 1 point and the minimum score is 0 points. Please output the score in the form "score: <score>". The evaluation criteria require that the closer the model's answer is to the standard answer, the higher the score.
"""

PROMPT = """
question: {}
standard answer: {}
model's answer: {}
"""


def resume_from_path(file_path) -> int:
    if os.path.exists(file_path):
        if os.path.isfile(file_path):
            with open(file_path, 'r') as fp:
                return len(fp.readlines())
        elif os.path.isdir(file_path):
            return len(os.listdir(file_path))
        else:
            raise ValueError(f"Invalid path: {file_path}")
    else:
        if file_path.endswith('.jsonl') or file_path.endswith('.json'):
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
        else:
            os.makedirs(file_path, exist_ok=True)
        return 0


def check_exists_score_file(tmp_file_path) -> int:
    pattern_dir = os.path.dirname(tmp_file_path)
    pattern_file_name = os.path.basename(tmp_file_path)
    pattern = pattern_file_name.replace('-tmp.jsonl', '-0.')
    for file in os.listdir(pattern_dir):
        if file.startswith(pattern) and file.endswith('.jsonl'):
            print(f"Found existing score file: {file}")
            return True
    return False


def print_n_times(N):
    count = 0
    def inner_function(message):
        nonlocal count
        if count < N:
            count += 1
            print(message)
    return inner_function

debug_print = print_n_times(3)

class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class CausalLMSingleton(metaclass=SingletonMeta):
    def __init__(self, model_name):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto")

    def get_tokenizer(self):
        return self.tokenizer

    def get_model(self):
        return self.model
    
class VLLMSingleton(metaclass=SingletonMeta):
    def __init__(self, model_name, max_model_len, max_num_seqs, tensor_parallel_size, pipeline_parallel_size, gpu_memory_utilization):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.model = LLM(model=model_name,
                         max_model_len=max_model_len,
                         max_num_seqs=max_num_seqs, 
                         tensor_parallel_size=tensor_parallel_size,
                         pipeline_parallel_size=pipeline_parallel_size,
                         gpu_memory_utilization=gpu_memory_utilization)
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_model(self):
        return self.model
    
    
class ClientSingleton(metaclass=SingletonMeta):
    def __init__(self, api_key, base_url):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        
    def get_client(self):
        return self.client




SCORE_MAPPERS = {}

def register_score_mappers():
    def wrapper(func):
        name = func.__name__.replace("_dataset_mapper", "")
        SCORE_MAPPERS[name] = func
        return func
    return wrapper


@register_score_mappers()
def cot_bench_dataset_mapper(one_data, args):
    user_query = one_data["conversations"][0]["value"].replace("Please provide the bounding box coordinate of the region that can help you answer the question better.", "").strip()
    pred_resp = one_data['response']
    gt_resp = one_data['conversations'][-1]['value']
    one_data['user_query'] = user_query
    one_data['gt_resp'] = gt_resp
    one_data['pred_resp'] = pred_resp
    return one_data

@register_score_mappers()
def cot_train_dataset_mapper(one_data, args):
    user_query = one_data['question']
    pred_resp = one_data['response']
    gt_resp = one_data['answer']
    one_data['user_query'] = user_query
    one_data['gt_resp'] = gt_resp
    one_data['pred_resp'] = pred_resp
    return one_data
    
@register_score_mappers()
def vstar_bench_dataset_mapper(one_data, args):
    user_query = one_data['text']
    pred_resp = one_data['response']
    gt_resp = one_data['label']
    one_data['user_query'] = user_query
    one_data['gt_resp'] = gt_resp
    one_data['pred_resp'] = pred_resp
    return one_data
    
@register_score_mappers()
def refcoco_dataset_mapper(one_data, args):
    one_data['user_query'] = one_data['problem']
    one_data['gt_resp'] = one_data['solution']
    one_data['pred_resp'] = one_data['response']
    return one_data


@register_score_mappers()
def scienceqa_img_dataset_mapper(one_data, args):
    one_data['user_query'] = one_data['question']
    one_data['gt_resp'] = chr(65 + one_data['answer'])
    one_data['pred_resp'] = one_data['response']
    return one_data

    
    
SCORE_FUNCS = {}

def register_score_func():
    def wrapper(func):
        name = func.__name__.split('_score')[0]
        SCORE_FUNCS[name] = func
        return func
    return wrapper
    
LOCAL_SCORE_FUNCS = {}

def register_local_score_func():
    def wrapper(func):
        name = func.__name__.split('_score')[0]
        LOCAL_SCORE_FUNCS[name] = func
        return func
    return wrapper
    
def extract_score_from_str(score_str):
    if not isinstance(score_str, str):
        return score_str
    lower_str = score_str.lower()
    if 'score' not in lower_str:
        return 0.0
    res = re.findall(r'score: ([\d\.]+)', lower_str)
    if len(res) != 1:
        return 0.0
    res = float(res[0])
    if res > 1.0:
        res = 1
    if res < 0.0:
        res = 0
    return res

def gpt_score(query, img_path, gt_resp, pred_resp, model_name, client_getter):
    client = client_getter.get_client()
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": PROMPT.format(query, gt_resp, pred_resp)},
        ],
        max_tokens=512,
        temperature=0,
    )
    score_str = completion.choices[0].message.content
    return score_str

@register_score_func()
def gpt_3_5_turbo_score(query, img_path, gt_resp, pred_resp, **kwargs):
    client_getter = ClientSingleton(os.getenv("OPENAI_API_KEY"), "https://api.openai.com/v1")
    return gpt_score(query, img_path, gt_resp, pred_resp, 'gpt-3.5-turbo-1106', client_getter)

def qwen_score(query, img_path, gt_resp, pred_resp, model_name, client_getter):
    client = client_getter.get_client()
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": PROMPT.format(query, gt_resp, pred_resp)},
        ],
        max_tokens=512,
        temperature=0,
    )
    # breakpoint()
    score_str = completion.choices[0].message.content
    return score_str
    
@register_score_func()
def qwen_plus_score(query, img_path, gt_resp, pred_resp, **kwargs):
    client_getter = ClientSingleton(os.getenv("DASHSCOPE_API_KEY"), "https://dashscope.aliyuncs.com/compatible-mode/v1")
    return qwen_score(query, img_path, gt_resp, pred_resp, 'qwen-plus', client_getter)   # qwen-plus-2024-11-25

@register_score_func()
def qwen_max_score(query, img_path, gt_resp, pred_resp, **kwargs):
    client_getter = ClientSingleton(os.getenv("DASHSCOPE_API_KEY"), "https://dashscope.aliyuncs.com/compatible-mode/v1")
    return qwen_score(query, img_path, gt_resp, pred_resp, 'qwen-max', client_getter)

@register_score_func()
def qwen_turbo_score(query, img_path, gt_resp, pred_resp, **kwargs):
    client_getter = ClientSingleton(os.getenv("DASHSCOPE_API_KEY"), "https://dashscope.aliyuncs.com/compatible-mode/v1")
    return qwen_score(query, img_path, gt_resp, pred_resp, 'qwen-turbo', client_getter)

@register_score_func()
def api_qwen_2_5_32b_int8_score(query, img_path, gt_resp, pred_resp, **kwargs):
    client_getter = ClientSingleton(None, "http://localhost:8000/v1")
    return qwen_score(query, img_path, gt_resp, pred_resp, 'Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8', client_getter)


def qwen_local_score(query, img_path, gt_resp, pred_resp, model_getter: CausalLMSingleton):
    model = model_getter.get_model()
    tokenizer = model_getter.get_tokenizer()
    messages = []
    for one_query, one_gt_resp, one_pred_resp in zip(query, gt_resp, pred_resp):
        one_messages = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": PROMPT.format(one_query, one_gt_resp, one_pred_resp)},
        ]
        messages.append(one_messages)
    texts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    generated_ids = model.generate(**model_inputs, max_length=512)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    debug_print(response)
    return response
    

@register_local_score_func()
def qwen_2_5_0_5b_score(query, img_path, gt_resp, pred_resp, **kwargs):  # This is only for debug
    model_getter = CausalLMSingleton('Qwen/Qwen2.5-0.5B-Instruct') 
    return qwen_local_score(query, img_path, gt_resp, pred_resp, model_getter)

@register_local_score_func()
def qwen_2_5_14b_score(query, img_path, gt_resp, pred_resp, **kwargs):
    model_getter = CausalLMSingleton('Qwen/Qwen2.5-14B-Instruct')
    return qwen_local_score(query, img_path, gt_resp, pred_resp, model_getter)
    
    
def qwen_local_vllm_score(query, img_path, gt_resp, pred_resp, model_getter: VLLMSingleton):
    llm = model_getter.get_model()
    tokenizer = model_getter.get_tokenizer()
    sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
    messages = []
    for one_query, one_gt_resp, one_pred_resp in zip(query, gt_resp, pred_resp):
        one_messages = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": PROMPT.format(one_query, one_gt_resp, one_pred_resp)},
        ]
        messages.append(one_messages)
    texts = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = llm.generate(texts, sampling_params)
    responses = []
    for output in outputs:
        responses.append(output.outputs[0].text)
    debug_print(responses)
    return responses
    

@register_local_score_func()
def vllm_qwen_2_5_0_5b_score(query, img_path, gt_resp, pred_resp, max_model_len, max_num_seqs, tensor_parallel_size, pipeline_parallel_size, gpu_memory_utilization,**kwargs):
    model_getter = VLLMSingleton('Qwen/Qwen2.5-0.5B-Instruct', max_model_len, max_num_seqs, tensor_parallel_size, pipeline_parallel_size, gpu_memory_utilization)
    return qwen_local_vllm_score(query, img_path, gt_resp, pred_resp, model_getter)

@register_local_score_func()
def vllm_qwen_2_5_14b_score(query, img_path, gt_resp, pred_resp, max_model_len, max_num_seqs, tensor_parallel_size, pipeline_parallel_size, gpu_memory_utilization,**kwargs):
    model_getter = VLLMSingleton('Qwen/Qwen2.5-14B-Instruct', max_model_len, max_num_seqs, tensor_parallel_size, pipeline_parallel_size, gpu_memory_utilization)
    return qwen_local_vllm_score(query, img_path, gt_resp, pred_resp, model_getter)   

@register_local_score_func()
def vllm_qwen_2_5_32b_score(query, img_path, gt_resp, pred_resp, max_model_len, max_num_seqs, tensor_parallel_size, pipeline_parallel_size, gpu_memory_utilization,**kwargs):
    model_getter = VLLMSingleton('Qwen/Qwen2.5-32B-Instruct', max_model_len, max_num_seqs, tensor_parallel_size, pipeline_parallel_size, gpu_memory_utilization)
    return qwen_local_vllm_score(query, img_path, gt_resp, pred_resp, model_getter)
 
@register_local_score_func()
def vllm_qwen_2_5_32b_int8_score(query, img_path, gt_resp, pred_resp, max_model_len, max_num_seqs, tensor_parallel_size, pipeline_parallel_size, gpu_memory_utilization, **kwargs):
    model_getter = VLLMSingleton('Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8', max_model_len, max_num_seqs, tensor_parallel_size, pipeline_parallel_size, gpu_memory_utilization)
    return qwen_local_vllm_score(query, img_path, gt_resp, pred_resp, model_getter)
    
    
@register_local_score_func()  # for debug
def dummy_score(query, img_path, gt_resp, pred_resp, **kwargs):
    return ['0.0'] * len(query)
    
@register_local_score_func()
def precise_match(query, img_path, gt_resp, pred_resp, **kwargs):
    scores = []
    for one_gt_resp, one_pred_resp in zip(gt_resp, pred_resp):
        if one_gt_resp == one_pred_resp:
            scores.append('score: 1.0')
        else:
            scores.append('score: 0.0')
    return scores


@register_local_score_func()
def single_choice(query, img_path, gt_resp, pred_resp, **kwargs):
    scores = []
    for one_gt_resp, one_pred_resp in zip(gt_resp, pred_resp):
        one_gt_resp = one_gt_resp.strip().upper()

        patterns = [
            # 匹配 "Answer: A", "The answer is A", "The correct option is: A" 等格式
            # re.IGNORECASE 忽略大小写
            r'(?:(?:the|my|the correct)\s+)?(?:answer|choice|option)\s*(?:is)?\s*[:：]?\s*([A-Z])',
            
            # 匹配被括号包围的答案, e.g., "(A)"
            r'\(([A-Z])\)',

            # 匹配以句号或右括号结尾的答案, e.g., "A." or "A)"
            # \b 是单词边界，确保不会匹配到单词中间的字母，例如从 "U.S.A." 中错误提取 "A"
            r'\b([A-Z])[\.\)]',

            # 匹配字符串开头的大写字母, e.g., "A. The cat is on the mat"
            r'^([A-Z])\b',
            
            # 作为最后的手段，匹配文本中第一个独立的单个大写字母
            r'\b([A-Z])\b'
        ]

        extracted_ans = None
        for pattern in patterns:
            match = re.search(pattern, one_pred_resp, re.IGNORECASE)
            if match:
                extracted_ans = match.group(1).upper()
                break
        
        # 比较提取出的答案和标准答案
        if extracted_ans and extracted_ans == one_gt_resp:
            scores.append(1)
        else:
            scores.append(0)

    return scores
    
    
@register_local_score_func()
def one_box_iou05(query, img_path, gt_resp, pred_resp, **kwargs):
    pred_bboxes = np.array([extract_one_bbox_from_str(resp) for resp in pred_resp])
    gt_bboxes = np.array(gt_resp)
    ious = cal_paired_ious(pred_bboxes, gt_bboxes)
    scores = ious > 0.5
    return scores.astype(np.float32).tolist()

    
def cal_score(result_data, cache, score_func):
    gt_resp = result_data['labels']
    pred_resp = result_data['response']
    query = result_data['messages'][0]['content']
    img_path = result_data['images'][0]['path']
    score_str = score_func(query, img_path, gt_resp, pred_resp)
    score = extract_score_from_str(score_str)
    cache.append({'query': query, 'gt': gt_resp, 'pred': pred_resp, 'score': score, 'score_str': score_str,'images': [img_path]})
    return score
    
def load_jsonl(file_path):
    with open(file_path, 'r') as f:
        return [json.loads(line) for line in f]
    
def save_jsonl(data, file_path):
    with open(file_path, 'w') as f:
        for line in data:
            f.write(json.dumps(line) + '\n')

def append_jsonl(one_data, file_path):
    with open(file_path, 'a') as f:
        f.write(json.dumps(one_data) + '\n')
        


def make_batch_inference_input(batch_data, tmp_in_path, model_name):
    tmp_in_lines = []
    # for result in tqdm(results, desc='Making batch inference input', total=len(results)):
    for r_id, result in enumerate(batch_data):
        user_query = result.pop('user_query')
        gt_resp = result.pop('gt_resp')
        pred_resp = result.pop('pred_resp')
        request_line = {
            "custom_id": f"request-{r_id}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model_name.replace('_', '-'),
                "messages": [
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": PROMPT.format(user_query, gt_resp, pred_resp)},
                ]
            }
        }
        tmp_in_lines.append(request_line)
    save_jsonl(tmp_in_lines, tmp_in_path)
    print(f'Batch inference input saved to {tmp_in_path}')


def convert_batch_inference_output_to_score(results, tmp_out_path):
    tmp_out_lines = load_jsonl(tmp_out_path)
    
    if len(results) != len(tmp_out_lines):
        raise ValueError(f"Length of input lines and output lines are not equal: {len(results)} vs {len(tmp_out_lines)}")
    score_dict = {}
    for line in tqdm(tmp_out_lines, desc='Converting batch inference output to score', total=len(tmp_out_lines)):
        rid = line['custom_id']
        result_idx = int(rid.split('-')[-1])
        status = line['response']['status_code']
        if status == 200:
            score_str = line['response']['body']['choices'][0]['message']['content']
            score = extract_score_from_str(score_str)
        else:
            # warning
            warnings.warn(f"Request {rid} failed with status code {status}")
            score_str = ''
            score = 0.0
        score_dict[result_idx] = score
    score_dict = {k: score_dict[k] for k in sorted(score_dict.keys())}
    scores = list(score_dict.values())
    return scores
    

def upload_file(client, file_path):
    print(f"Uploading the JSONL file containing request information...")
    file_object = client.files.create(file=Path(file_path), purpose="batch")
    print(f"File uploaded successfully. Obtained file ID: {file_object.id}\n")
    return file_object.id

def create_batch_job(client, input_file_id):
    print(f"Creating a Batch job based on the file ID...")
    batch = client.batches.create(input_file_id=input_file_id, endpoint="/v1/chat/completions", completion_window="24h")
    print(f"Batch job created successfully. Obtained Batch job ID: {batch.id}\n")
    return batch.id

def check_job_status(client, batch_id):
    batch = client.batches.retrieve(batch_id=batch_id)
    return batch.status

def get_output_id(client, batch_id):
    print(f"Retrieving the output file ID of successfully executed requests in the Batch job...")
    batch = client.batches.retrieve(batch_id=batch_id)
    print(f"Output file ID: {batch.output_file_id}\n")
    return batch.output_file_id

def get_error_id(client, batch_id):
    print(f"Retrieving the output file ID of requests with errors in the Batch job...")
    batch = client.batches.retrieve(batch_id=batch_id)
    print(f"Error file ID: {batch.error_file_id}\n")
    return batch.error_file_id

def download_results(client, output_file_id, output_file_path):
    print(f"Printing and downloading the successful request results of the Batch job...")
    content = client.files.content(output_file_id)
    print(f"Printing the first 1000 characters of the successful request results: {content.text[:1000]}...\n")
    content.write_to_file(output_file_path)
    print(f"The complete output results have been saved to the local output file: {output_file_path}\n")

def download_errors(client, error_file_id, error_file_path):
    print(f"Printing and downloading the failed request information of the Batch job...")
    content = client.files.content(error_file_id)
    print(f"Printing the first 1000 characters of the failed request information: {content.text[:1000]}...\n")
    content.write_to_file(error_file_path)
    print(f"The complete failed request information has been saved to the local error file: {error_file_path}\n")
    

def collect_as_batched_obj(batch_objs):
    return batch_objs
        
        
def calculate_avg_score(batch_data, args):
    scores = []
    for one_data in batch_data:
        scores.append(one_data['score'])
    return sum(scores) / len(scores)

def update_scores(batch_data, scores, args):
    assert len(batch_data) == len(scores)
    for one_data, score in zip(batch_data, scores):
        one_data['score'] = score  
        
        
def process_batch_data_by_local(batch_data, args):
    gt_resp = [one_data.pop('gt_resp') for one_data in batch_data]
    pred_resp = [one_data.pop('pred_resp') for one_data in batch_data]
    user_query = [one_data.pop('user_query') for one_data in batch_data]
    score_strs = LOCAL_SCORE_FUNCS[args.score_func](user_query, None, gt_resp, pred_resp, **vars(args))
    scores = [extract_score_from_str(score_str) for score_str in score_strs]
    return scores
        
def process_one_data_by_api(one_data, args):
    gt_resp = one_data.pop('gt_resp')
    pred_resp = one_data.pop('pred_resp')
    user_query = one_data.pop('user_query')
    score_str = SCORE_FUNCS[args.score_func](user_query, None, gt_resp, pred_resp)
    score = extract_score_from_str(score_str)
    return score
    
def process_batch_data_by_api(batch_data, bid, args):
    score_func_name = args.score_func
    # assert score_func_name in ['qwen_max', 'qwen_plus', 'qwen_turbo'], f"Batch inference is not supported for {score_func_name}"
    tmp_in_path = os.path.join(os.path.dirname(args.tmp_score_path), os.path.basename(args.tmp_score_path).replace('.jsonl', f'_batch-{bid}_in.jsonl'))
    tmp_out_path = os.path.join(os.path.dirname(args.tmp_score_path), os.path.basename(args.tmp_score_path).replace('.jsonl', f'_batch-{bid}_out.jsonl'))
    error_path = os.path.join(os.path.dirname(args.tmp_score_path), os.path.basename(args.tmp_score_path).replace('.jsonl', f'_batch-{bid}_error.jsonl'))
    
    # if os.path.exists(tmp_in_path) and args.batch_id is None:
    #     raise ValueError(f"Batch inference input file {tmp_in_path} already exists")
    if not os.path.exists(tmp_out_path):
        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 百炼服务的base_url
        )
        if args.batch_id is not None:
            batch_id = args.batch_id
        else:
            make_batch_inference_input(batch_data, tmp_in_path, score_func_name)
            input_file_id = upload_file(client, tmp_in_path)
            batch_id = create_batch_job(client, input_file_id)
        status = ""
        check_times = 0
        with tqdm(
            desc=f"[0s] 等待任务完成...",
            bar_format="{desc}",
        ) as pbar:
            while True:
                status = check_job_status(client, batch_id)
                elapsed_time = check_times * args.interval
                pbar.set_description(f"[{elapsed_time}s] 等待任务完成... 当前状态: {status}")
                if status in ["completed", "failed", "expired", "cancelled"]:
                    break
                check_times += 1
                time.sleep(args.interval)
        
        if status == "failed":
            batch = client.batches.retrieve(batch_id)
            print(f"Batch failed: {batch.errors}\n")
            print(f"Refer to: https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
        output_file_id = get_output_id(client, batch_id)
        if output_file_id:
            download_results(client, output_file_id, tmp_out_path)
        error_file_id = get_error_id(client, batch_id)
        if error_file_id:
            download_errors(client, error_file_id, error_path)
            print(f"Refer to: https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
            raise ValueError(f"Batch job failed, error file saved to {error_path}")

    print(f"Batch inference finished.")
    scores = convert_batch_inference_output_to_score(batch_data, tmp_out_path)
    if os.path.exists(tmp_in_path):
        os.remove(tmp_in_path)
    return scores
    
    
def process_batch_data(batch_data, bid, args):
    if args.score_func in SCORE_FUNCS:
        if len(batch_data) == 1:
            scores = [process_one_data_by_api(batch_data[0], args)]
        else:
            scores = process_batch_data_by_api(batch_data, bid, args)
    else:
        scores = process_batch_data_by_local(batch_data, args)
    update_scores(batch_data, scores, args)
    return batch_data

def post_process_batch_data(batch_data, args):
    # remove intermediate data
    keys = ['user_query', 'gt_resp', 'pred_resp']
    for one_data in batch_data:
        for key in keys:
            if key in one_data:
                one_data.pop(key)




def gather_batched_obj(batch_src_data, mask):
    return [batch_src_data[i] for i in range(len(batch_src_data)) if mask[i]]
    
    
def scatter_batched_obj(batch_dst_data, batch_src_data, mask):
    j = 0
    for i in range(len(batch_dst_data)):
        if mask[i]:
            batch_dst_data[i] = batch_src_data[j]
            j += 1
    assert j == len(batch_src_data)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-jsonl', type=str, required=True, nargs='+', help='Result JSONL file paths')
    parser.add_argument('--score-func', type=str, default='vllm_qwen_2_5_32b_int8', choices=list(SCORE_FUNCS.keys()) + list(LOCAL_SCORE_FUNCS.keys()))
    parser.add_argument('--interval', type=int, default=10, help='Interval for checking batch job status, in seconds')
    parser.add_argument('--batch-id', type=str, default=None, help='Batch ID for checking batch job status')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--mapper', type=str, default='cot_bench', choices=SCORE_MAPPERS.keys())
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val', 'test'])
    parser.add_argument('--max-num-seqs', type=int, default=256, help='Max number of sequences for VLLM')
    parser.add_argument('--max-model-len', type=int, default=4096, help='Max model length for VLLM')
    parser.add_argument('--tensor-parallel-size', type=int, default=1, help='Tensor parallel size for VLLM')
    parser.add_argument('--pipeline-parallel-size', type=int, default=1, help='Pipeline parallel size for VLLM')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9, help='GPU memory utilization for VLLM')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    if args.tensor_parallel_size > 1 or args.pipeline_parallel_size > 1:
        os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
    
    for result_jsonl in args.result_jsonl:
        try:
            print(f"\nProcessing file: {result_jsonl}")
            
            len_dataset = len(load_dataset("json", data_files=result_jsonl, split=args.split))
            dataset = load_dataset("json", data_files=result_jsonl, split=args.split, streaming=True)
            
            if "vstar" in result_jsonl:
                mapper_name = 'vstar_bench'
            else:
                mapper_name = args.mapper
            
            dataset = dataset.map(lambda x: SCORE_MAPPERS[mapper_name](x, args))
            
            tmp_score_path = os.path.join(os.path.dirname(result_jsonl), os.path.basename(result_jsonl).replace('.jsonl', f'_{args.score_func}-tmp.jsonl'))
            if check_exists_score_file(tmp_score_path):
                continue
            beg_line = resume_from_path(tmp_score_path)
            beg_batch = beg_line // args.batch_size
            dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collect_as_batched_obj)
            args.tmp_score_path = tmp_score_path
            print(f"Output to {tmp_score_path}, start from line {beg_line} in batch {beg_batch}")
            
            if beg_line == 0:
                avg_score = 0.0
                total_num = 0
            else:
                old_datas = load_jsonl(tmp_score_path)
                avg_score = calculate_avg_score(old_datas, args)
                total_num = len(old_datas)
                del old_datas
                
            with open(tmp_score_path, 'a') as out_fp:
                for bid, batch_data in tqdm(enumerate(dataloader), total=len_dataset//args.batch_size):
                    if bid < beg_batch:
                        continue
                    if bid == beg_batch:
                        skip = beg_line % args.batch_size
                        if skip > 0:
                            print(f"Skip {skip} lines in batch {bid}")
                            batch_data = batch_data[skip:]
                    batch_data = process_batch_data(batch_data, bid, args)
                    post_process_batch_data(batch_data, args)            
                    total_num += len(batch_data)
                    avg_score += (calculate_avg_score(batch_data, args) - avg_score) * len(batch_data) / total_num
                    for one_output in batch_data:
                        out_fp.write(json.dumps(one_output) + '\n')
                    out_fp.flush()
                    
            score_path = tmp_score_path.replace('-tmp', f'-{avg_score:.4f}')
            os.rename(tmp_score_path, score_path)
            print(f"Output to {score_path} done!, final avg score: {avg_score}, total num: {total_num}")
            
        except Exception as e:
            print(f"Error processing file {result_jsonl}: {str(e)}")
            print("Skipping to next file...")
            continue
        

if __name__ == "__main__":
    main()