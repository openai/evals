import os
import requests
from tqdm import tqdm
import numpy as np
from gensim.models import KeyedVectors

import evals
from evals.api import CompletionFn
from evals.elsuite import utils
from evals.record import RecorderBase
from evals import metrics


def extract_entities(input_str):
    """
    Extracts entities from the input string that are formatted as (...), (...), (...).
    
    Args:
    - input_str (str): The input string containing entities.
    
    Returns:
    - list of entities: A list containing all extracted entities.
    """
    pattern = r'\(\s*([^,]+)\s*\)'
    try:
        matches = re.findall(pattern, input_str)
        return matches
    except:
        print(f"cannot extract entities from {input_str}")
        return []

def extract_tuple(input_str):
    """
    Extracts tuples from the input string that are formatted as (element1, element2),
    allowing for variable spacing between elements.
    
    Args:
    - input_str (str): The input string containing pairs.
    
    Returns:
    - list of tuples: A list containing all extracted pairs.
    """
    pattern = r'\(\s*([^,]+)\s*,\s*([^,]+)\s*\)'
    try:
        matches = re.findall(pattern, input_str)
        return matches
    except:
        print(f"cannot extract tuples from {input_str}")
        return []


def extract_triplets(input_str):
    """
    Extracts tuples from the input string that are formatted as (..., ..., ...).
    
    Args:
    - input_str (str): The input string containing tuples.
    
    Returns:
    - list of tuples: A list containing all extracted tuples.
    """
    # 正则表达式用于匹配格式为 (..., ..., ...) 的三元组
    pattern = r'\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*\)'
    try:
        matches = re.findall(pattern, input_str)
        return matches
    except:
        print(f"cannot extract triplets from {input_str}")
        return []


# 定义下载用的进度条以及下载行为
def download_file_with_progress(url, path):
    # 尝试获取已下载的文件大小，如果文件不存在，则大小为0
    if os.path.exists(path):
        first_byte = os.path.getsize(path)
    else:
        first_byte = 0
    
    # 获取文件总大小
    headers = {"Range": f"bytes={first_byte}-"}
    response = requests.get(url, headers=headers, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0)) + first_byte
    
    progress_bar = tqdm(total=total_size_in_bytes, initial=first_byte, unit='iB', unit_scale=True)
    
    # 以追加模式打开文件
    with open(path, 'ab') as file:
        for data in response.iter_content(chunk_size=5120):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")

def get_word2vec_model_path():
    W2V_MODEL_PATH = os.getenv('W2V_MODEL_PATH')
    if not W2V_MODEL_PATH:
        default_path = '/root/model/GoogleNews-vectors-negative300.bin'
        if os.path.exists(default_path):
            print("Warning: W2V_MODEL_PATH environment variable is not set. Using default path:", default_path)
            W2V_MODEL_PATH = default_path
        else:
            print("Default path is not accessible. Attempting to download the model.")
            # Placeholder for the model download URL
            download_url = 'https://dp-filetrans.oss-accelerate.aliyuncs.com/linmujie/word2vec_model/GoogleNews_vectors_negative300.bin?OSSAccessKeyId=LTAI5tRnCpedMnKSH3APDceY&Expires=1708944899&Signature=CMrBE6ugLE8bWzOufAtlbi1Phng%3D'
            #定义好下载的路径
            W2V_MODEL_PATH = '/root/model/GoogleNews-vectors-negative300.bin'
            try:
                download_file_with_progress(download_url, W2V_MODEL_PATH)
                print(f"Model downloaded and saved to {W2V_MODEL_PATH}")
            except Exception as e:
                raise Exception(f"Failed to download the model. Error: {e}")

    return W2V_MODEL_PATH

class Word2VecModel:
    _instance = None

    @staticmethod
    def get_instance(model_path=None):
        if Word2VecModel._instance is None:
            if model_path is None:
                raise ValueError("A model_path must be provided for the first initialization!")
            Word2VecModel._instance = Word2VecModel(model_path)
        # elif model_path is not None:
        #     # 如果实例已存在且提供了新的model_path，重新加载模型
        #     Word2VecModel._instance.load_model(model_path)
        return Word2VecModel._instance

    def __init__(self, model_path):
        self.load_model(model_path)

    def load_model(self, model_path):
        print(f"Loading Pretrained Word2Vec Model:{model_path}")
        self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        print(f"Pretrained Word2Vec Model loaded/reloaded from {model_path}")

class CosineMatchEntity(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        samples_jsonl: str,
        *args,
        max_tokens: int = 100,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "CosineMatch only supports one completion fn"
        self.max_tokens = max_tokens
        self.samples_jsonl = samples_jsonl

        # 加载并缓存word2vec模型
        W2V_MODEL_PATH = get_word2vec_model_path()
        self.word2vec_model = Word2VecModel.get_instance(W2V_MODEL_PATH).model


    def eval_sample(self, test_sample, rng):
        del rng

        assert isinstance(test_sample, dict), "sample must be a dict"
        assert "input" in test_sample, "sample must have an 'input' key"
        assert "ideal" in test_sample, "sample must have an 'ideal' key"

        prompt, correct_answers = test_sample["input"], test_sample["ideal"]
        correct_answers = parse_triplets(correct_answers)
        # if not isinstance(correct_answers, list):
        #     correct_answers = [correct_answers]

        result = self.completion_fn(
            prompt=prompt,
            temperature=0.0,  # Q: why are these hardcoded?
            max_tokens=self.max_tokens,
        )
        sampled = result.get_completions()[0]
        sample_list = extract_triplets(sampled)
        # 以下matches = [True, True, False, ....]
        if sample_list:
            matches = [utils.same_triplets(self.word2vec_model, sample_list, correct_answer) for correct_answer in correct_answers]
        else:
            matches = [False for correct_answer in correct_answers]

        evals.record.record_match(
            True in matches,
            expected=correct_answers, #这一条文献的标准答案
            picked=[utils.pick_most_similar_entity_in_pred(self.word2vec_model, sample_list, correct_answers[i]) for i in range(len(correct_answers)) if matches[i]], #输出的list里边能在答案找到的三元组
        )
        evals.record.record_metrics(
            accuracy = sum(matches)/len(matches) if matches else 0.0, 
            f1_score = utils.macro_f1_score_3(self.word2vec_model, sample_list, correct_answers),
        )

    def run(self, recorder: RecorderBase):
        samples = self.get_samples()
        self.eval_all_samples(recorder, samples)

        return {
            "accuracy": np.mean(recorder.get_scores("accuracy")),
            "f1_score": np.mean(recorder.get_scores("f1_score")),
        }

class CosineMatchTuple(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        samples_jsonl: str,
        *args,
        max_tokens: int = 100,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "CosineMatchTuple only supports one completion fn"
        self.max_tokens = max_tokens
        self.samples_jsonl = samples_jsonl

        # 加载并缓存word2vec模型
        W2V_MODEL_PATH = get_word2vec_model_path()
        self.word2vec_model = Word2VecModel.get_instance(W2V_MODEL_PATH).model


    def eval_sample(self, test_sample, rng):
        del rng

        assert isinstance(test_sample, dict), "sample must be a dict"
        assert "input" in test_sample, "sample must have an 'input' key"
        assert "ideal" in test_sample, "sample must have an 'ideal' key"

        prompt, correct_answers = test_sample["input"], test_sample["ideal"]
        correct_answers = extract_tuple(correct_answers)
        # if not isinstance(correct_answers, list):
        #     correct_answers = [correct_answers]

        result = self.completion_fn(
            prompt=prompt,
            temperature=0.0,  # Q: why are these hardcoded?
            max_tokens=self.max_tokens,
        )
        sampled = result.get_completions()[0]
        sample_list = extract_tuple(sampled)
        if sample_list:
            matches = [utils.same_turples(self.word2vec_model, sample_list, correct_answer) for correct_answer in correct_answers]
        else:
            matches = [False for correct_answer in correct_answers]
        evals.record.record_match(
            True in matches, 
            expected=correct_answers,
            picked=[utils.pick_same_turples_in_pred(self.word2vec_model, sample_list, correct_answers[i]) for i in range(len(correct_answers)) if matches[i]],
        )
        evals.record.record_metrics(
            accuracy=sum(matches)/len(matches) if matches else 0.0, 
            f1_score=utils.macro_f1_score_3(self.word2vec_model, sample_list, correct_answers),
        )
        

    def run(self, recorder: RecorderBase):
        samples = self.get_samples()
        self.eval_all_samples(recorder, samples)

        return {
            "accuracy": np.mean(recorder.get_scores("accuracy")),
            "f1_score": np.mean(recorder.get_scores("f1_score")),
        }

class CosineMatchTriplet(evals.Eval):
    def __init__(
        self,
        completion_fns: list[CompletionFn],
        samples_jsonl: str,
        *args,
        max_tokens: int = 100,
        **kwargs,
    ):
        super().__init__(completion_fns, *args, **kwargs)
        assert len(completion_fns) == 1, "CosineMatch only supports one completion fn"
        self.max_tokens = max_tokens
        self.samples_jsonl = samples_jsonl

        # 加载并缓存word2vec模型
        W2V_MODEL_PATH = get_word2vec_model_path()
        self.word2vec_model = Word2VecModel.get_instance(W2V_MODEL_PATH).model


    def eval_sample(self, test_sample, rng):
        del rng

        assert isinstance(test_sample, dict), "sample must be a dict"
        assert "input" in test_sample, "sample must have an 'input' key"
        assert "ideal" in test_sample, "sample must have an 'ideal' key"

        prompt, correct_answers = test_sample["input"], test_sample["ideal"]
        correct_answers = parse_triplets(correct_answers)
        # if not isinstance(correct_answers, list):
        #     correct_answers = [correct_answers]

        result = self.completion_fn(
            prompt=prompt,
            temperature=0.0,  # Q: why are these hardcoded?
            max_tokens=self.max_tokens,
        )
        sampled = result.get_completions()[0]
        sample_list = extract_triplets(sampled)
        # 以下matches = [True, True, False, ....]
        if sample_list:
            matches = [utils.same_triplets(self.word2vec_model, sample_list, correct_answer) for correct_answer in correct_answers]
        else:
            matches = [False for correct_answer in correct_answers]

        evals.record.record_match(
            True in matches,
            expected=correct_answers, #这一条文献的标准答案
            picked=[utils.pick_same_triplets_in_pred(self.word2vec_model, sample_list, correct_answers[i]) for i in range(len(correct_answers)) if matches[i]], #输出的list里边能在答案找到的三元组
        )
        evals.record.record_metrics(
            accuracy = sum(matches)/len(matches) if matches else 0.0, 
            f1_score = utils.macro_f1_score_3(self.word2vec_model, sample_list, correct_answers),
        )

    def run(self, recorder: RecorderBase):
        samples = self.get_samples()
        self.eval_all_samples(recorder, samples)

        return {
            "accuracy": np.mean(recorder.get_scores("accuracy")),
            "f1_score": np.mean(recorder.get_scores("f1_score")),
        }
