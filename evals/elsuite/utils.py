import copy
import re
import string
from collections import Counter, defaultdict
from typing import Optional, Union, List, Any
from scipy.spatial.distance import cosine
from gensim.models import KeyedVectors
from evals import metrics

from evals import CompletionFn
from evals.prompt.base import (
    OpenAICreateChatPrompt,
    OpenAICreatePrompt,
    Prompt,
    chat_prompt_to_text_prompt,
    is_chat_prompt,
)


def get_answer(text, answer_prompt, ignore_case=False):
    if ignore_case:
        idx = text.lower().rfind(answer_prompt.lower())
    else:
        idx = text.rfind(answer_prompt)

    if idx == -1:
        return None
    return text[idx : idx + len(answer_prompt)]


def get_consensus(answers):
    counts = defaultdict(int)
    for answer in answers:
        counts[answer] += 1
    counts[None] = 0
    return max(counts, key=counts.get)


def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def fuzzy_match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)

    if s1 == "" or s2 == "":
        return s1 == s2

    return s1 in s2 or s2 in s1

def get_scores_from_text(text: str) -> dict:
    pattern = r"## (.+?)\n.+?(\d)/5"
    matches = re.findall(pattern, text, re.DOTALL)
    return {k: int(v) for k, v in dict(matches).items()}


def get_yesno_from_text(text: str) -> dict:
    pattern = r"## (.+?)\n.+?([yn])"
    matches = re.findall(pattern, text, re.DOTALL)
    return {k: v for k, v in dict(matches).items()}


def get_letter_from_data(data: str) -> str:
    last_y = (data.rfind("y"), "y")
    last_n = (data.rfind("n"), "n")
    char = max(last_y, last_n)[1]
    return char

#定义使用Word2Vec计算词间相似度的metric
#sentence -> vector
def sentence_to_vec(sentence:str, model):
    words = sentence.split()
    word_vectors = []
    for word in words:
        if word in model:
            word_vectors.append(model[word])
    
    if not word_vectors:
        return np.zeros(model.vector_size)
    
    word_vectors = np.array(word_vectors)
    sentence_vector = word_vectors.mean(axis=0)
    return sentence_vector


def cosine_similarity(model, sentence1:str,sentence2:str):
    try:
        vec1 = sentence_to_vec(sentence1, model)
        vec2 = sentence_to_vec(sentence2, model)
        return 1 - cosine(vec1, vec2)
    except KeyError as e:
        return 0  # Return 0 similarity if word not found

def same_entities(model, word1: str, word2: str, threshold: float) -> bool:
    if (cosine_similarity(model, word1, word2) - threshold) > 1e-8:
        return True
    else:
        return False

def cosine_match(s1: str, s2: str) -> bool:
    s1 = normalize(s1)
    s2 = normalize(s2)

    if s1 == "" or s2 == "":
        return s1 == s2
    # if same_entities(s1,s2) == True:
    #     return s1
    # else:
    #     return 
    return s1 in s2 or s2 in s1

#线性查找算法（暴力穷举）：匹配两个list
def LinearSearch(item: List[Any], lst: List[List[Any]]) -> bool:
    
    # 检查lst中的每个元素的类型，如果是字符串，则尝试将其转换为列表
    lst = [eval(element) if isinstance(element, str) else element for element in lst]
    
    index = 0
    found = False     
    while index < len(lst) and not found:
        # 只有当[compound, disease]两个都匹配上才算是同一条关系
        if same_entities(lst[index][0], item[0], 0.7) and same_entities(lst[index][1], item[1], 0.7):
            found = True
        else:
            index += 1
    return found

def list_match_ratio(s1: List[List[Any]], s2: List[List[Any]]) -> float:
    # s1 = normalize(s1)
    # s2 = normalize(s2)
    count = 0
    #在正确答案（ideal）中遍历，以检查
    for item in s2:
        if LinearSearch(item, s1):
            count += 1 
    return count/len(s2) if s2 else 0

def f1_score(prediction: str, answers: list[str]) -> float:
    def _f1_score(prediction: str, ground_truth: str):
        # 检查 ground_truth 是否是列表
        if not isinstance(ground_truth, list):
            ground_truth_tokens = normalize(ground_truth).split()
        else:
            ground_truth_tokens = ground_truth
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    return max([_f1_score(prediction, answer) for answer in answers])


#定义用于实体识别的f1-score
def cos_f1_score(model, prediction: str, answers: list[str]) -> float:
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    matched_ground_truth_tokens = set()
    similarity_threshold = 0.7 #这里暂时把相同词义的判定阈值写死
    prediction_tokens = normalize(prediction).split()
    for pred_entity in prediction_tokens:
        found_match = False
        for idx, true_entity in enumerate(answers):
            similarity = metrics.cosine_similarity(model, pred_entity, true_entity)
            if similarity > similarity_threshold:
                found_match = True
                if idx not in matched_ground_truth_tokens:
                    true_positives += 1
                    matched_ground_truth_tokens.add(idx)
                break
        if not found_match:
            false_positives += 1
    false_negatives = len(answers) - len(matched_ground_truth_tokens)
    # Calculate precision, recall, and F1 score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1

#定义用于关系抽取的f1-score
def macro_f1_score(prediction: List[List[Any]], answers: List[List[Any]]) -> float:
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    matched_ground_truth_tokens = set()
    # similarity_threshold = 0.7 
    for pred_entity in prediction:
        found_match = False
        for idx, true_entity in enumerate(answers):
            if same_entities(pred_entity[0],true_entity[0], 0.7) and same_entities(pred_entity[1],true_entity[1],0.7):
                found_match = True
                if idx not in matched_ground_truth_tokens:
                    true_positives += 1
                    matched_ground_truth_tokens.add(idx)
                break
        if not found_match:
            false_positives += 1
    false_negatives = len(answers) - len(matched_ground_truth_tokens)
    # Calculate precision, recall, and F1 score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1


def scrub_formatting_from_prompt(prompt):
    scrubbed_prompt = copy.copy(prompt)

    if is_chat_prompt(prompt):
        for i, msg in enumerate(scrubbed_prompt):
            if "content" in msg:
                scrubbed_prompt[i]["content"] = msg["content"].replace("{", "{{").replace("}", "}}")
    else:
        scrubbed_prompt = scrubbed_prompt.replace("{", "{{").replace("}", "}}")
    return scrubbed_prompt


def format_necessary(template: str, allow_missing: bool = False, **kwargs: dict[str, str]) -> str:
    """Format a template string with only necessary kwargs."""
    keys = [k[1] for k in string.Formatter().parse(template) if k[1]]
    if allow_missing:
        assert (
            len([k for k in keys if k in kwargs]) > 0
        ), f"Required: {keys}, got: {sorted(kwargs)}, no inputs are used.\nTemplate:\n{template}"
        cur_keys = {k: kwargs.get(k, "{" + k + "}") for k in keys}
    else:
        assert all(
            k in kwargs for k in keys
        ), f"Required: {keys}, got: {sorted(kwargs)}.\nTemplate:\n{template}"
        cur_keys = {k: kwargs[k] for k in keys}
    return template.format(**cur_keys)


def format_prompt(
    prompt: OpenAICreatePrompt, allow_missing: bool = False, **kwargs: dict[str, str]
) -> OpenAICreatePrompt:
    """Format a prompt with only necessary kwargs."""
    # if any input kwargs is chat prompt, convert to text prompt
    kwargs = {
        k: chat_prompt_to_text_prompt(v, for_completion=False) if is_chat_prompt(v) else v
        for k, v in kwargs.items()
    }
    if is_chat_prompt(prompt):
        new_prompt = []
        for msg in prompt:
            formatted_msg = copy.copy(msg)
            if "content" in formatted_msg:
                formatted_msg["content"] = format_necessary(
                    formatted_msg["content"], allow_missing=allow_missing, **kwargs
                )
            new_prompt.append(formatted_msg)
        prompt = new_prompt
    else:
        # Prompt is a string
        prompt = format_necessary(prompt, allow_missing=allow_missing, **kwargs)
    return prompt


class PromptFn:
    """
    Wrap calls to a completion_fn with a prompt template with applicable keyword args.
    This will pass many args relevant to OpenAI Completion API, may be ignored by other completion_fn.
    """

    def __init__(
        self,
        prompt: Union[OpenAICreatePrompt, OpenAICreateChatPrompt, Prompt],
        completion_fn: CompletionFn,
        max_tokens: int,
        temperature: int = 0,
        n_samples: Optional[int] = None,
        completion_kwargs: Optional[dict] = {},
    ):
        self.prompt = prompt
        self.max_tokens = max_tokens
        self.completion_fn = completion_fn
        self.temperature = temperature
        self.completion_kwargs = completion_kwargs
        self.n_samples = n_samples

    def __call__(self, **kwargs):
        # if any input kwargs is chat prompt, convert to text prompt
        kwargs = {
            k: chat_prompt_to_text_prompt(v, for_completion=False) if is_chat_prompt(v) else v
            for k, v in kwargs.items()
        }
        if is_chat_prompt(self.prompt):
            prompt = []
            for msg in self.prompt:
                formatted_msg = copy.copy(msg)
                if "content" in formatted_msg:
                    formatted_msg["content"] = format_necessary(formatted_msg["content"], **kwargs)
                prompt.append(formatted_msg)
        else:
            # Prompt is a string
            prompt = format_necessary(self.prompt, **kwargs)

        result = self.completion_fn(
            prompt=prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            n=(1 if self.n_samples is None else self.n_samples),
            **self.completion_kwargs,
        )
        sampled = result.get_completions()[0]
        return sampled, prompt


def markdown_format_prompt(prompt):
    if type(prompt) == list:
        return "\n\n".join([f"**{message['role']}**: {message['content']}" for message in prompt])
    else:
        return prompt
