# 如何添加一个定制化的eval
本教程将向您展示编写和添加自定义eval的简单示例。示例eval将测试模型进行基本算术运算的能力。我们假定您已按照[README](../../README.md)中的设置说明进行设置，并查看了其他关于如何运行和构建eval的文档。

在编写自己的eval时，感兴趣的主要文件包括：

- `evals/api.py`，提供eval创建者用于从模型中采样和处理结果的常用接口和工具，
- `evals/record.py`，定义记录器类，以不同的方式记录eval结果，例如记录到本地JSON文件或到远程Snowflake数据库，以及
- `evals/metrics.py`，定义各种常见的有用指标。

这些文件提供了一套工具，用于编写新的eval。完成本教程后，您可以在机器翻译的评估示例中看到这些工具的更实际的示例，该示例还实现了自定义eval逻辑，而不是使用现有的模板。

## 创建数据集
首先，需要创建用于 eval 的数据集。在这里，我们将创建两个起到示例作用的训练集和测试集。测试示例是我们用来评估模型的，我们将在提示中将训练示例作为 few-shot 示例包含进去。

我们将使用[这里](https://platform.openai.com/docs/guides/chat/introduction)描述的新聊天格式。默认情况下，如果您想评估我们的新模型，我们鼓励所有 eval 使用聊天格式编写。在底层，我们将聊天格式的数据[转换](../../evals/prompt/base.py)为旧版非聊天模型的原始字符串。

要创建示例数据集，请在终端中输入：
```bash
echo -e '[{"role": "system", "content": "2+2=", "name": "example_user"}, {"role": "system", "content": "4", "name": "example_assistant"}]\n[{"role": "system", "content": "4*4=", "name": "example_user"}, {"role": "system", "content": "16", "name": "example_assistant"}]' > /tmp/train.jsonl
echo -e '[{"role": "system", "content": "48+2=", "name": "example_user"}, {"role": "system", "content": "50", "name": "example_assistant"}]\n[{"role": "system", "content": "5*20=", "name": "example_user"}, {"role": "system", "content": "100", "name": "example_assistant"}]' > /tmp/test.jsonl
```

## 创建一个eval
接下来的步骤是编写一个 Python 类来表示实际的评估。该类使用您的数据集创建提示，并将其传递给模型生成完成结果。评估类通常将继承自 `evals.Eval` 基类（定义在 `evals/eval.py` 中），并将覆盖两个方法：`eval_sample` 和 `run`。

我们将在 `evals/elsuite` 文件夹下创建一个名为 `arithmetic.py` 的文件。我们将从定义评估类开始。它的 `__init__` 方法将接受我们需要的参数（指向训练集和测试集的引用），以及其他由基类处理的 `kwargs`。我们还将定义 `run` 方法，该方法接受一个 `recorder` 并返回最终的有用指标。

```python
import random
import textwrap

import evals
import evals.metrics

class Arithmetic(evals.Eval):
    def __init__(self, train_jsonl, test_jsonl, train_samples_per_prompt=2, **kwargs):
        super().__init__(**kwargs)
        self.train_jsonl = train_jsonl
        self.test_jsonl = test_jsonl
        self.train_samples_per_prompt = train_samples_per_prompt

    def run(self, recorder):
        """
        Called by the `oaieval` CLI to run the eval. The `eval_all_samples` method calls `eval_sample`.
        """
        self.train_samples = evals.get_jsonl(self.train_jsonl)
        test_samples = evals.get_jsonl(self.test_jsonl)
        self.eval_all_samples(recorder, test_samples)

        # Record overall metrics
        return {
            "accuracy": evals.metrics.get_accuracy(recorder.get_events("match")),
        }
```

通常情况下，大多数 `run` 方法将按照此处所示的相同模式进行：加载数据、调用 `eval_all_samples` 并汇总结果（在此情况下，使用 `evals/metrics.py` 中的 `get_accuracy` 函数）。`eval_all_samples` 接收 `recorder` 和 `test_samples`，在底层将对 `test_samples` 中的每个样本调用 `eval_sample` 方法。因此，现在让我们编写该 `eval_sample` 方法：

```python
    def eval_sample(self, test_sample, rng: random.Random):
        """
        Called by the `eval_all_samples` method to evaluate a single sample.

        ARGS
        ====
        `test_sample`: a line from the JSONL test file
        `rng`: should be used for any randomness that is needed during evaluation

        This method does the following:
        1. Generate a prompt that contains the task statement, a few examples, and the test question.
        2. Check if the model generates the correct answer.
        """
        stuffing = rng.sample(self.train_samples, self.train_samples_per_prompt)

        prompt = [
            {"role": "system", "content": "Solve the following math problems"},
        ]

        for i, sample in enumerate(stuffing + [test_sample]):
            if i < len(stuffing):
                prompt += [
                    {"role": "system", "content": sample["problem"], "name": "example_user"},
                    {"role": "system", "content": sample["answer"], "name": "example_assistant"},
                ]
            else:
                prompt += [{"role": "user", "content": sample["problem"]}]

        evals.check_sampled_text(self.model_spec, prompt, expected=sample["answer"])
```
你会注意到 `eval_sample` 方法没有将 `recorder` 作为参数传递进去。这是因为 `eval_all_samples` 在调用 `eval_sample` 之前将其设置为默认 `recorder` ，而 `evals/record.py` 中定义的记录实用程序使用默认 `recorder`。在这个例子中，`eval_sample` 方法将许多繁重的工作传递给了 `evals/api.py` 中定义的 `evals.check_sampled_text` 实用函数。该实用函数使用给定的 `prompt` 查询由 `self.model_spec` 定义的模型，并检查结果是否与 `expected` 答案匹配（如果给定列表，则匹配其中之一）。然后，它使用默认 `recorder`记录这些匹配（或非匹配）。

`eval_sample` 方法可能根据您的用例有很大不同。如果您正在构建自定义evals，则最好熟悉 `evals/record.py`、`evals/metrics.py` 以及特别是 `evals/api.py` 中可用的函数。

## 注册eval
下一步是将eval注册到注册表中，以便可以使用`oaieval` CLI运行它。

让我们在 `evals/registry/evals` 文件夹下创建一个名为 `arithmetic.yaml` 的文件，并添加以下条目：
```yaml
# Define a base eval
arithmetic:
  # id specifies the eval that this eval is an alias for
  # in this case, arithmetic is an alias for arithmetic.dev.match-v1
  # When you run `oaieval davinci arithmetic`, you are actually running `oaieval davinci arithmetic.dev.match-v1`
  id: arithmetic.dev.match-v1
  # The metrics that this eval records
  # The first metric will be considered to be the primary metric
  metrics: [accuracy]
  description: Evaluate arithmetic ability
# Define the eval
arithmetic.dev.match-v1:
  # Specify the class name as a dotted path to the module and class
  class: evals.elsuite.arithmetic:Arithmetic
  # Specify the arguments as a dictionary of JSONL URIs
  # These arguments can be anything that you want to pass to the class constructor
  args:
    train_jsonl: /tmp/train.jsonl
    test_jsonl: /tmp/test.jsonl
```
`args` 字段应该与你的评估类的 `__init__` 方法所需的参数匹配。

## 运行eval
最后一步是运行你的eval，并查看结果。

```sh
pip install .  # you can omit this if you used `pip install -e .` to install
oaieval gpt-3.5-turbo arithmetic
```
如果你使用 `gpt-3.5-turbo` 模型运行，应该会看到类似于以下输出的结果（我们在此处稍微清理了输出以便更易读）：
```
% oaieval gpt-3.5-turbo arithmetic
... [registry.py:147] Loading registry from .../evals/registry/evals
... [registry.py:147] Loading registry from .../.evals/evals
... [oaieval.py:139] Run started: <run_id>
... [eval.py:32] Evaluating 2 samples
... [eval.py:138] Running in threaded mode with 1 threads!
100%|██████████████████████████████████████████| 2/2 [00:00<00:00,  3.35it/s]
... [record.py:320] Final report: {'accuracy': 1.0}. Logged to /tmp/evallogs/<run_id>_gpt-3.5-turbo_arithmetic.jsonl
... [oaieval.py:170] Final report:
... [oaieval.py:172] accuracy: 1.0
... [record.py:309] Logged 6 rows of events to /tmp/evallogs/<run_id>_gpt-3.5-turbo_arithmetic.jsonl: insert_time=2.038ms
```
如果注意到evals已缓存您的数据并且需要清除该缓存，可以使用 `rm -rf /tmp/filecache` 执行此操作。