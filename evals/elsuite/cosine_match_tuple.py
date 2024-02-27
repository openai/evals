import numpy as np

import evals
from evals.api import CompletionFn
from evals.elsuite import utils
from evals.record import RecorderBase
from evals import metrics

#将形如“1. [Doxorubicin, acute cardiomyopathy]\n2. [Doxorubicin, chronic cardiomyopathy]\n3. [Doxorubicin, cardiac damage]”的字符串，
# 转化成列表List[List[Any]]类型，结构如下：[[Doxorubicin, acute cardiomyopathy],[Doxorubicin, chronic cardiomyopathy],[Doxorubicin, cardiac damage]]

def str2list(input_str: str):
    # input_str = "1. [Doxorubicin, acute cardiomyopathy]\n2. [Doxorubicin, chronic cardiomyopathy]\n3. [Doxorubicin, cardiac damage]"
    # 使用分割和解析的方法将输入字符串转换成所需的列表结构
    # 首先，按换行符分割字符串，得到每一项
    items = input_str.split('\n')
    result = []
    for item in items:
       # 移除序号和点号，然后找到方括号内的内容
        item_clean = item[item.index('[') + 1:item.rindex(']')]
        # 将方括号内的内容按逗号分割，并去除两边的空白，然后转换为列表
        item_list = [x.strip() for x in item_clean.split(',')]
        # 处理每个元素，确保去除了引号
        item_list = [x.strip('\'"') for x in item_list]
        # 将处理后的列表添加到结果中
        result.append(item_list)

    return result


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

    def eval_sample(self, test_sample, rng):
        del rng

        assert isinstance(test_sample, dict), "sample must be a dict"
        assert "input" in test_sample, "sample must have an 'input' key"
        assert "ideal" in test_sample, "sample must have an 'ideal' key"

        prompt, correct_answers = test_sample["input"], test_sample["ideal"]
        if not isinstance(correct_answers, list):
            correct_answers = [correct_answers]

        result = self.completion_fn(
            prompt=prompt,
            temperature=0.0,  # Q: why are these hardcoded?
            max_tokens=self.max_tokens,
        )
        sampled = result.get_completions()[0]
        matches = [utils.list_match_ratio(str2list(sampled), correct_answer) for correct_answer in correct_answers]

        evals.record.record_match(
            any(match > 0.5 for match in matches), ##这里修改了
            expected=correct_answers,
            picked=[sampled for i in range(len(correct_answers)) if matches[i]>0.5],#整个relations大于0.5就认为是匹配的
        )
        evals.record.record_metrics(
            accuracy=float(np.argmax(matches)), 
            f1_score=utils.macro_f1_score(str2list(sampled), correct_answers),#改成新写的函数cos_f1_score
        )

    def run(self, recorder: RecorderBase):
        samples = self.get_samples()
        self.eval_all_samples(recorder, samples)

        return {
            "accuracy": np.mean(recorder.get_scores("accuracy")),
            "f1_score": np.mean(recorder.get_scores("f1_score")),
        }
