# Evals

Evals 是一个用于评估OpenAI模型的开源框架和基准测试库。

您可以使用 Evals 创建和运行以下评估：
- 使用数据集生成提示，
- 衡量 OpenAI 模型提供的完成质量，以及
- 比较不同数据集和模型的性能。

通过 Evals，我们的目标是尽可能简化构建评估的过程，同时尽可能少地编写代码。为了开始使用 Evals，我们建议您按照以下步骤**顺序进行**操作：
1. 阅读本文档并按照 [下面的设置说明](README.zh.md#设置说明)。
2. 了解如何运行现有评估: [run-evals.md](docs/run-evals.md)。
3. 熟悉现有的评估模板：[eval-templates.md](docs/eval-templates.md)。
4. 演练构建评估的过程：[build-eval.md](docs/build-eval.md)。
5. 查看实现自定义评估逻辑的示例： [custom-eval.md](docs/custom-eval.md)。

如果你认为你有一个有趣的评估，请贡献你的PR。在考虑改进即将推出的模型时，OpenAI 工作人员会积极审查这些评估。

____________________
🚨 有限的时间内，我们将授予那些提供高质量评估的人GPT-4的访问权限。请按照上述提到的说明进行操作，并注意垃圾邮件或低质量提交将被忽略。❗️

将授权访问与接受评估相关联的电子邮件地址。由于数量庞大，我们无法授权访问除用于拉取请求的电子邮件地址以外的任何电子邮件。
____________________

## 设置说明

要运行评估，您需要设置并指定您的 OpenAI API 密钥。 您可以在 <https://platform.openai.com/account/api-keys> 生成你的密钥。 获取API密钥后, 使用`OPENAI_API_KEY` 环境变量指定它。 **请注意在运行评估时使用 API 的相关[费用](https://openai.com/pricing)。**

### 下载 evals

我们的 Evals 注册表使用[Git-LFS](https://git-lfs.com/)进行存储。一旦您下载并安装了LFS，您可以使用以下命令获取 evals ：
```sh
git lfs fetch --all
git lfs pull
```

您可能只想为选择的评估获取数据。您可以通过以下方式实现：
```sh
git lfs fetch --include=evals/registry/data/${your eval}
git lfs pull
```

### 创建 evals

如果您要创建 evals，我们建议直接从 GitHub 克隆此存储库并使用以下命令安装依赖：

```sh
pip install -e .
```

使用 `-e`, 您对 evals 所做的更改将立即反映出来，而无需重新安装。

### 运行 evals

如果你不想贡献新的 evals，而只是想在本地运行它们，你可以通过 pip 安装 evals 包：

```sh
pip install evals
```

如果你有Snowflake数据库或希望建立一个，我们为你提供了将你的评估结果记录到Snowflake数据库的选项。对于这个选项，你需要指定 `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_DATABASE`, `SNOWFLAKE_USERNAME`,  `SNOWFLAKE_PASSWORD` 环境变量。

## FAQ

您有任何关于如何从头到尾构建评估的示例吗？

- 是的！这些在`examples` 文件夹中。 我们建议您通读 [build-eval.md](docs/build-eval.md) 以便更深入地了解这些示例中发生的事情。

您有任何以多种不同方式实施评估的示例吗？

- 是的！ 特别是，请参阅 `evals/registry/evals/coqa.yaml`。 我们已经为各种评估模板实现了 [CoQA](https://stanfordnlp.github.io/coqa/) 数据集的小子集，以帮助说明差异。

我更改了我的数据，但这在运行我的评估时没有反映出来，这是怎么回事？

- 您的数据可能已缓存到 `/tmp/filecache`。尝试删除此缓存并重新运行您的评估。

有很多代码，我只想启动一个快速评估。 帮助？ 或者，

我是世界级的提示工程师。 我选择不编码。 我如何贡献我的智慧？

- 如果您按照现有的 [评估模板](docs/eval-templates.md) 构建一个基础评估或模型分级评估，您根本不需要编写任何评估代码！只需以 JSON 格式提供您的数据并在 YAML 中指定您的评估参数。 [build-eval.md](docs/build-eval.md) 会指导你完成这些步骤， 您可以使用 `examples` 文件夹中的 Jupyter 笔记本补充这些说明以帮助您快速入门。 但请记住，一个好的评估不可避免地需要仔细思考和严格的实验！

## 免责声明

通过对 Evals 的贡献，你同意将你的评估逻辑和数据置于与本资源库相同的MIT许可之下。您必须有足够的权利来上传评估中使用的任何数据。OpenAI保留在未来对我们产品的服务改进中使用这些数据的权利。对OpenAI Evals的贡献将受到我们通常的使用政策的约束：https://platform.openai.com/docs/usage-policies 。
