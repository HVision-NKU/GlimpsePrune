import re
from openai import OpenAI


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class LLMClient(metaclass=SingletonMeta):
    SYS_PROMPT = """
You are responsible for proofreading the answers, you need to give a score to the model's answer by referring to the standard answer, based on the given question. The full score is 1 point and the minimum score is 0 points. Please output the score in the form "score: <score>". The evaluation criteria require that the closer the model's answer is to the standard answer, the higher the score.
"""

    PROMPT = """
question: {}
standard answer: {}
model's answer: {}
"""
    def __init__(self, base_url, api_key, model_name, timeout=20.0):
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model_name = model_name
        self._check_init()
        
    
    def _check_init(self):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": "Hello!"}
            ]
            )
        print(completion.choices[0].message)
        
    def _extract_score_from_str(self, score_str: str) -> float:
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
    
    def score(self, query_texts, completion_texts, answer_texts):
        """
        Scores the completions based on the query and answer texts.
        """
        scores = []
        for query, answer, completion in zip(query_texts, answer_texts, completion_texts):
            messages = [
                {"role": "system", "content": self.SYS_PROMPT},
                {"role": "user", "content": self.PROMPT.format(query, answer, completion)},
            ]
            try:
                completion = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.0,
                )
                score_str = completion.choices[0].message.content
                score = self._extract_score_from_str(score_str)
            except Exception as e:
                print(f"Error during scoring: {e}")
                score = 0.0
            scores.append(score)
        return scores