# 创建一个新的eval

本文将详细介绍构建一个eval（数据集和eval类的选择）的端到端过程。示例文件夹`examples`包含了跟随以下步骤构建几个学术eval的Jupyter笔记本，从而帮助说明整个过程。

该过程的步骤包括构建数据集、使用数据集注册新的eval以及运行eval。关键是，我们假设您正在直接使用[现有的eval模板](eval-templates.md)（如果不是这种情况，请参阅[构建自定义eval的示例](custome-eval.md)）。如果您有兴趣公开贡献您的eval，我们还在底部包含了一些标准，以说明我们认为哪些eval具有趣味性。

## 格式化您的数据
一旦您想要实现一个eval，您需要将样本转换为正确的JSON行（JSONL）格式。JSONL文件只是一个JSON文件，每行有一个唯一的JSON对象。

我们在 [registry/data/README.md](../../evals/registry/data/README.md) 中提供了一些JSONL eval文件的示例。

每个JSON对象将代表您eval中的一个数据点。您在JSON对象中需要的键取决于eval模板。所有模板都需要一个名为`“input”`的键，这是提示语，最好采用聊天格式（尽管也支持字符串）。即使您正在评估非聊天模型，我们也建议使用[chat format](https://platform.openai.com/docs/guides/chat/introduction)。如果您要同时评估聊天和非聊天模型，我们会处理聊天格式提示和原始字符串提示之间的转换逻辑（请参阅此处的[转换逻辑](../../evals/prompt/base.py)）。

对于基本evals `Match`，`Includes`和`FuzzyMatch`，另一个必需的键是`"ideal"`，它是一个字符串（或字符串列表），指定正确的参考答案。对于模型评分的evals，所需的键因eval而异，但是由评估`prompt`中未被`args`（可选的）覆盖的`{key}`s来确定。

我们已经为各种eval模板实现了[CoQA](https://stanfordnlp.github.io/coqa/)数据集的小子集，以说明数据应该如何格式化。请参见[`coqa/match.jsonl`](../../evals/registry/data/coqa/match.jsonl)以获取适用于`Match`基本eval模板的数据示例，以及[`coqa/samples.jsonl`](../../evals/registry/data/coqa/samples.jsonl)以获取适用于`fact`和`closedqa`模型评分eval的数据。请注意，即使这两个模型评分eval期望不同的键，我们也可以在我们的数据中包括键的超集，以支持这两个eval。

如果数据集文件在您的本地计算机上，请将YAML文件放在`evals/registry/evals/data/<eval_name>/samples.jsonl`中。如果在Cloud Object Storage中，则我们支持主要云服务的路径样式URL（仅供您个人使用，我们不会接受具有云URL的PR）。




## 注册eval

注册eval需要按照elsuite注册格式添加文件到`evals/registry/evals/<eval_name>.yaml`. 例如，对于Match评估，yaml文件应该形如：
```
<eval_name>:
  id: <eval_name>.dev.v0
  metrics: [accuracy]

<eval_name>.dev.v0:
  class: evals.elsuite.basic.match:Match
  args:
    samples_jsonl: <eval_name>/samples.jsonl
```
在运行eval时，数据将在`evals/registry/data`中搜索，如果`test_match/samples.jsonl`是提供的文件路径，则期望数据在`evals/registry/data/test_match/samples.jsonl`中。

eval的命名规则的格式为`<eval_name>.<split>.<version>`。

`<eval_name>`是eval名称，用于分组分数可比较的evals。
`<split>`是数据集分割，用于进一步分组处于同一`<base_eval>`下的eval。例如，用于测试的“val”，“test”或“dev”。
`<version>`是eval的版本，可以使用任何描述性文本（尽管最好不要包含“.”）。
通常，使用相同的eval名称对同一模型进行评估应该始终给出类似的结果，以便其他人可以重现它。因此，当您更改eval时，应该增加版本号。

## 运行eval
现在，您可以使用所选模型从 CLI 上对数据运行您的eval：
```
oaieval gpt-3.5-turbo <eval_name>
```
恭喜，您已经构建了您的eval！继续对其进行迭代，直到您对结果感到满意。请记住，如果您更改了数据文件，请删除 `/tmp/filecache`，以便使用您更新的数据运行eval。

## 针对模型评分的eval：一个逐步的工作流
我们预计现有的模型评分evals，如`fact`、`closedqa`和`battle`将适合许多用例。然而，其他用例可能会从更多定制中受益，例如不同的评估提示。对于这些情况，需要更多的工作，但通常不需要编码！

1. 如果您无法使用现有的模型评分eval，请在`evals/registry/modelgraded`中创建一个新的YAML，以指定您的eval[参数](eval-templates.md#parameters-for-model-graded-evals)。参见[`humor.yaml`](../../evals/registry/modelgraded/humor.yaml)的示例。
    - 请注意，即使您正在创建新的YAML，您可能会发现将现有的YAML复制为起点最容易。例如，将检查模型是否完成与评分表匹配的模型评分eval可以直接复制`closedqa.yaml`，然后只需编辑args即可。
2. 接下来，您将创建数据集并注册eval，如上所述。参见[`joke_fruits_labeled.jsonl`](../../evals/registry/data/test_metaeval/joke_fruits_labeled.jsonl)和[`joke-fruits`](../../evals/registry/evals/test-modelgraded.yaml)的示例。
    - 请注意，在此步骤中建议指定`eval_type`，而不是在第1步中指定。
3. 运行您的eval，例如，`oaleval gpt-3.5-turbo joke-fruits`。
4. （推荐）为针对模型评分的eval添加meta-eval. 每个针对模型评分的eval都带有一些细节要调整，主要是`prompt`，但也包括`eval_type`。为了确保评估的质量，我们建议每个针对模型评分的Eval贡献都附带“选择标签”，这基本上是人工提供的标签，用于指示模型应该做出哪个评估选择。例如（假装这些笑话真的很有趣），请参见[`joke_fruits_labeled.jsonl`](../../evals/registry/data/test_metaeval/joke_fruits_labeled.jsonl)中的`"choice"`键，该键不被`joke-fruits` eval 使用，但被下面的`joke-fruits-meta` meta-eval使用。运行meta-eval，例如`oaieval gpt-3.5-turbo joke-fruits-meta`，报告将输出`metascore/ accuracies`，对于良好的模型评分eval，这些值应该接近“1.0”。

## 提交eval的标准
重要提示：如果您正在贡献代码，请在提交和推送之前运行`pip install pre-commit; pre-commit install`，以确保运行了 `black`、`isort` 和 `autoflake`。

我们有兴趣策划一组多样化和有趣的eval，以改进我们未来的模型。以下是我们认为好的eval的一些标准：

 - [ ] eval应该在主题上保持一致。我们希望看到一些提示围绕同一个用例、主题领域、故障模式等。
 - [ ] eval应该具有挑战性。如果 GPT-4 或 GPT-3.5-Turbo 在所有提示上表现良好，那么这就不是那么有趣了。当然，eval也应该考虑模型的限制和约束，一个好的经验法则是，一个人（可能是一个学科专家）能否在提示上表现良好。
 - [ ] eval应该具有明确的方向性。数据应该包含正确行为的良好信号。这意味着，例如高质量的参考答案或评估答案的详尽评分表。
 - [ ] eval应该经过精心设计。在提交之前，您应该考虑是否为良好的性能设计了提示，是否使用了最佳的评估模板，是否进行了点检以确保准确性等。

一旦您准备好公开提交您的eval，请提交一个PR，OpenAI团队将很乐意审查它。请确保填写PR消息中预填充的模板的所有部分。请注意，提交PR并不保证OpenAI最终会合并它。我们将运行自己的检查并使用最佳判断去决定哪些evals需要跟进。