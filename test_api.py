import os
import time
import threading
import queue
from utils import LLMClient

# ===============================================================
# 2. 性能测试逻辑
# ===============================================================

def generate_dummy_data(num_items):
    """生成用于测试的假数据。"""
    queries = [f"What is the capital of France? {i}" for i in range(num_items)]
    answers = [f"The capital of France is Paris. {i}" for i in range(num_items)]
    completions = [f"Paris is the capital of France. {i}" for i in range(num_items)]
    return queries, completions, answers

def worker(task_queue, client_params, results_queue):
    """线程工作函数：从队列中获取任务并处理。"""
    # 每个线程都获取自己的客户端实例（但由于是单例，它们会共享同一个）
    client = LLMClient(**client_params)
    
    while True:
        try:
            # 从队列获取一个任务，设置超时以避免永久阻塞
            task_data = task_queue.get(timeout=1.0)
        except queue.Empty:
            # 队列已空，工作线程可以退出了
            break
        
        query_texts, completion_texts, answer_texts = task_data
        
        # 记录单个请求的开始时间
        req_start_time = time.time()
        
        # 使用 LLMClient 的 score 方法处理任务
        scores = client.score(query_texts, completion_texts, answer_texts)
        
        req_end_time = time.time()
        
        # 将处理结果（延迟时间，处理的项目数）放入结果队列
        latency = req_end_time - req_start_time
        num_items = len(scores)
        results_queue.put((latency, num_items))
        
        # 标记任务完成
        task_queue.task_done()


def run_performance_test(base_url, model_name, num_requests, num_workers, batch_size, timeout):
    """
    运行性能测试的主函数。
    
    :param base_url: LLM API 的基础URL。
    :param model_name: 要测试的模型名称。
    :param num_requests: 总共要处理的请求（任务）数量。
    :param num_workers: 并发工作线程的数量（模拟并发用户）。
    :param batch_size: 每个请求中包含的数据项数量。
    :param timeout: API请求的超时时间。
    """
    print("="*50)
    print(f"Starting Performance Test")
    print(f"  Endpoint: {base_url}")
    print(f"  Model: {model_name}")
    print(f"  Concurrency: {num_workers} workers")
    print(f"  Total Requests: {num_requests}")
    print(f"  Items per Request: {batch_size}")
    print("="*50)

    # 1. 初始化 Client (仅用于预检)
    #    工作线程会自己创建/获取实例
    try:
        client_params = {"base_url": base_url, "model_name": model_name, "timeout": timeout, "api_key": "dummy"}
        _ = LLMClient(**client_params)
    except Exception as e:
        # 打印我们捕获到的具体异常信息
        print("\n" + "="*20 + " INITIALIZATION FAILED " + "="*20)
        print(f"An error occurred during the initial client setup: {e}")
        print("The test cannot continue. Please review the error message and check your setup.")
        print("="*61 + "\n")
        # 明确地退出
        return

    # 2. 准备任务队列和结果队列
    task_queue = queue.Queue()
    results_queue = queue.Queue()
    total_items_to_process = num_requests * batch_size
    
    print(f"Generating {total_items_to_process} dummy data items and queuing {num_requests} tasks...")
    for _ in range(num_requests):
        q, c, a = generate_dummy_data(batch_size)
        task_queue.put((q, c, a))
    print("Tasks are ready in the queue.")

    # 3. 创建并启动工作线程
    threads = []
    
    print(f"\nStarting {num_workers} worker threads...")
    for _ in range(num_workers):
        thread = threading.Thread(target=worker, args=(task_queue, client_params, results_queue))
        thread.start()
        threads.append(thread)

    # 4. 记录总体开始时间并等待所有任务完成
    total_start_time = time.time()
    task_queue.join() # 阻塞直到队列中的所有任务都被 task_done()
    total_end_time = time.time()

    # 5. 等待所有线程真正终止
    for thread in threads:
        thread.join()
        
    # 6. 收集并计算结果
    total_time = total_end_time - total_start_time
    
    latencies = []
    total_items_processed = 0
    while not results_queue.empty():
        latency, num_items = results_queue.get()
        latencies.append(latency)
        total_items_processed += num_items

    # 7. 计算并报告指标
    # 避免除以零错误
    if not latencies:
        print("No requests were processed successfully.")
        return

    avg_latency_per_request = sum(latencies) / len(latencies)
    throughput_rps = num_requests / total_time  # Requests per second (RPS)
    
    # 你的 score 方法是串行的，所以每个 item 都会产生一个 API 调用
    # 因此 "Items per second" 实际上等于 "API calls per second"
    throughput_api_calls_ps = total_items_processed / total_time 
    avg_latency_per_api_call = total_time / total_items_processed if total_items_processed > 0 else 0

    print("\n" + "="*50)
    print("Test Finished!")
    print("="*50)
    print(f"Total time taken: {total_time:.4f} seconds")
    print(f"Total requests processed: {len(latencies)} / {num_requests}")
    print(f"Total API calls made (items scored): {total_items_processed}")
    print("-" * 50)
    print(f"Throughput (Requests/sec): {throughput_rps:.4f} RPS")
    print(f"Throughput (API calls/sec): {throughput_api_calls_ps:.4f} calls/sec")
    print(f"Avg. Latency per Request (batch): {avg_latency_per_request:.4f} seconds")
    print(f"Avg. Latency per API call (item): {avg_latency_per_api_call:.4f} seconds")
    print("="*50)


# ===============================================================
# 3. 运行测试
# ===============================================================
if __name__ == "__main__":
    # --- 可配置的测试参数 ---
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = "xxx"    
    
    # 你的LLM API服务地址
    API_BASE_URL = "http://localhost:8000/v1"
    
    # 你在服务中加载的模型名称
    MODEL_NAME = "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8" # 请替换为你的模型名, 如 "meta-llama/Llama-2-7b-chat-hf"
    # MODEL_NAME = None
    
    # --- 负载配置 ---
    
    # 总共发起多少个 score() 调用
    TOTAL_REQUESTS = 50
    
    # 并发数：模拟多少个用户同时请求
    CONCURRENT_WORKERS = 10
    
    # 批处理大小：每次 score() 调用处理多少条数据
    # 注意：因为你的 score() 内部是循环，这会产生 `ITEMS_PER_REQUEST` 个连续的API请求
    ITEMS_PER_REQUEST = 4
    
    # 单个API请求的超时时间（秒）
    REQUEST_TIMEOUT = 30.0
    
    # 运行测试
    run_performance_test(
        base_url=API_BASE_URL,
        model_name=MODEL_NAME,
        num_requests=TOTAL_REQUESTS,
        num_workers=CONCURRENT_WORKERS,
        batch_size=ITEMS_PER_REQUEST,
        timeout=REQUEST_TIMEOUT
    )