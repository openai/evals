# Evals

EvalはOpenAIモデルを評価するためのフレームワークであり、ベンチマークのオープンソースのレジストリでもある。

Evalを使用すると、以下のような評価を作成し、実行することができます。
- データセットを使ってプロンプトを生成する。
- OpenAIモデルが提供するCompletionの品質を測定し、
- 異なるデータセットやモデルで性能を比較することができます。

Evalでは、できるだけ少ないコードでevalをビルドすることができるようにすることを目的としています。開始するには、以下のステップを **順番に踏むことをお勧めします:
1. このドキュメントを読み、以下の[セットアップ手順](README.md#Setup)に従ってください。
2. 既存のevalを実行する方法について学びます。[run-evals.md](docs/run-evals.md).
3. 既存のevalテンプレートに慣れましょう。[eval-templates.md](docs/eval-templates.md).
4. evalをビルドする手順を確認する: [build-eval.md](docs/build-eval.md)
5. カスタムevalロジックの実装例はこちらをご覧下さい。[custom-eval.md](docs/custom-eval.md).

もし、あなたが興味深いevalをお持ちでしたら、ぜひPRを開いて投稿してください。OpenAIのスタッフは、次期モデルの改良を検討する際に、これらのevalを積極的に検討します。

____________________
🚨 期間限定で、質の高いevalを投稿していただいた方にGPT-4へのアクセス権を付与します。上記の指示に従ってください。スパムや低品質の投稿は無視されますのでご注意ください❗️

アクセスは、受け付けたEvalに紐づくEメールアドレスに付与されます。大量にあるため、プルリクエストに使用されたEメール以外へのアクセスは許可されません。
____________________

## セットアップ

Evalを実行するには、OpenAI APIキーの設定と指定が必要です。<https://platform.openai.com/account/api-keys>で生成することができます。APIキーを取得したら、環境変数 `OPENAI_API_KEY` を使って指定します。**Evalを実行する際にAPIを使用する際に発生する[コスト](https://openai.com/pricing)にご注意ください。**

### evalのダウンロード

当社のEvalレジストリは、[Git-LFS](https://git-lfs.com/)を使用して保存されています。LFSをダウンロードし、インストールすると、evalを取得することができます。
```sh
git lfs fetch --all
git lfs pull
```

You may just want to fetch data for a select eval. You can achieve this via:
```sh
git lfs fetch --include=evals/registry/data/${your eval}
git lfs pull
```

### Making evals

If you are going to be creating evals, we suggest cloning this repo directly from GitHub and installing the requirements using the following command:

```sh
pip install -e .
```

Using `-e`, changes you make to your eval will be reflected immediately without having to reinstall.

### Running evals

If you don't want to contribute new evals, but simply want to run them locally, you can install the evals package via pip:

```sh
pip install evals
```

We provide the option for you to log your eval results to a Snowflake database, if you have one or wish to set one up. For this option, you will further have to specify the `SNOWFLAKE_ACCOUNT`, `SNOWFLAKE_DATABASE`, `SNOWFLAKE_USERNAME`, and `SNOWFLAKE_PASSWORD` environment variables.

## FAQ

Do you have any examples of how to build an eval from start to finish?

- Yes! These are in the `examples` folder. We recommend that you also read through [build-eval.md](docs/build-eval.md) in order to gain a deeper understanding of what is happening in these examples.

Do you have any examples of evals implemented in multiple different ways?

- Yes! In particular, see `evals/registry/evals/coqa.yaml`. We have implemented small subsets of the [CoQA](https://stanfordnlp.github.io/coqa/) dataset for various eval templates to help illustrate the differences.

I changed my data but this isn't reflected when running my eval, what's going on?

- Your data may have been cached to `/tmp/filecache`. Try removing this cache and rerunning your eval.

There's a lot of code, and I just want to spin up a quick eval. Help? OR,

I am a world-class prompt engineer. I choose not to code. How can I contribute my wisdom?

- If you follow an existing [eval template](docs/eval-templates.md) to build a basic or model-graded eval, you don't need to write any evaluation code at all! Just provide your data in JSON format and specify your eval parameters in YAML. [build-eval.md](docs/build-eval.md) walks you through these steps, and you can supplement these instructions with the Jupyter notebooks in the `examples` folder to help you get started quickly. Keep in mind, though, that a good eval will inevitably require careful thought and rigorous experimentation!

## Disclaimer

By contributing to Evals, you are agreeing to make your evaluation logic and data under the same MIT license as this repository. You must have adequate rights to upload any data used in an Eval. OpenAI reserves the right to use this data in future service improvements to our product. Contributions to OpenAI Evals will be subject to our usual Usage Policies: https://platform.openai.com/docs/usage-policies.
