# カスタム評価の追加方法について

このチュートリアルでは、カスタム評価を作成し追加する簡単な例を説明します。この例の評価では、モデルが基本的な算数を行う能力をテストします。 ここでは、あなたが[README](../README.md)にあるセットアップ手順に従い、評価の実行と構築方法に関する他のドキュメントを読み進めたことを前提に説明します。

独自の評価を作成する場合、主に以下のファイルが注目されます:
- `evals/api.py`は評価の作成者がモデルからサンプリングして結果を処理するために使用する共通のインターフェースとユーティリティを提供します。
- `evals/record.py` は評価の結果をローカルのJSONファイルやリモートのSnowflakeデータベースなど、さまざまな方法でログに記録するレコーダークラスを定義したものです。
- `evals/metrics.py` は様々な興味深い一般的な指標を定義しています。

これらのファイルは、新しい評価を作成するためのツールセットを提供します。このチュートリアルを終えた後、[機械翻訳](../evals/elsuite/translate.py) [評価例](../examples/lafand-mt.ipynb)で、これらのツールが実際にどのように動作するか、より現実的な例を見ることができます。これは、既存のテンプレートを使用する代わりに、カスタム評価ロジックを実装しています。

## データセットの作成

まず最初に、評価用のデータセットを作成します。ここでは、2つずつの例からなるtoy trainとtestのセットを作成します。 testの例はモデルを評価するもので、trainの例はモデルへのプロンプトにFew-shotの例を含めることにします。

私たちは、[こちら](https://platform.openai.com/docs/guides/chat/introduction)で説明されている新しいチャットのフォーマットを使用する予定です。デフォルトで、私たちの新しいモデルを評価したいのであれば、すべての評価はチャットのフォーマットで書かれることを推奨します。 内部ではチャットフォーマットのデータをチャット以外の古いモデルのための生の文字列に[変換](../evals/prompt/base.py)しています。

トイデータセットを作成するためにターミナルで次のように入力します:
```bash
echo -e '[{"role": "system", "content": "2+2=", "name": "example_user"}, {"role": "system", "content": "4", "name": "example_assistant"}]\n[{"role": "system", "content": "4*4=", "name": "example_user"}, {"role": "system", "content": "16", "name": "example_assistant"}]' > /tmp/train.jsonl
echo -e '[{"role": "system", "content": "48+2=", "name": "example_user"}, {"role": "system", "content": "50", "name": "example_assistant"}]\n[{"role": "system", "content": "5*20=", "name": "example_user"}, {"role": "system", "content": "100", "name": "example_assistant"}]' > /tmp/test.jsonl
```

## 評価の作成

次のステップは、実際の評価を表す Python クラスを作成することです。このクラスはデータセット内にプロンプトを作成し、それをモデルに渡すことでcompletionを生成します。評価クラスは一般的に `evals.Eval` ベースクラス (`evals/eval.py` で定義) を継承し、 `eval_sample` と `run` という2つのメソッドをオーバーライドします。

`eval/elsuite`フォルダの下に `arithmetic.py` というファイルを作成しましょう。まず、評価 クラスを定義します。その `__init__` メソッドは、必要な引数（トレーニングセットとテストセットへの参照）と、ベースクラスによって処理されるその他の `kwargs` を受け取ります。また、`recorder` を受け取って対象の最終的なメトリクスを返す `run` メソッドを定義します。

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

一般的にほとんどの`run`メソッドはここで示されているパターンに従います：データをロードし`eval_all_samples`を呼び出し、結果を集約します（この場合は、`evals/metrics.py`の`get_accuracy`関数を使用）。`eval_all_samples`は、`recorder`と`test_samples`の両方を入力として受け取り内部的には`test_samples`の各サンプルに対して`eval_sample`メソッドを呼び出します。それでは、`eval_sample`メソッドを書いてみましょう：

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
`eval_sample`が`recorder`を引数として取らないことに気付くでしょう。これは、`eval_all_samples`が`eval_sample`を呼び出す前にデフォルトのレコーダーに設定し、`evals/record.py`で定義されている記録ユーティリティがデフォルトのレコーダーを使用するためです。この例では、`eval_sample`メソッドは、`evals/api.py`で定義されている`evals.check_sampled_text`ユーティリティ関数に多くの重要な処理を委ねています。このユーティリティ関数は、`self.model_spec`で定義されたモデルに対して与えられた`prompt`を問い合わせ、結果が`expected`の答え（または、リストが与えられた場合はそのうちの一つ）と一致するかどうかを確認します。その後、デフォルトのレコーダーを使用してこれらの一致（または不一致）を記録します。

また`eval_sample`メソッドはユースケースによって大きく異なる可能性があります。カスタム評価を構築する場合、`eval/record.py`、`evals/metrics.py`、特に`evals/api.py`で利用できる関数に精通しておくと良いでしょう。

## 評価を登録する

次のステップは、`oaieval` CLIを使って実行できるように、評価をレジストリに登録することです。

ここでは`eval/registry/eval`フォルダの下に`arithmetic.yaml`というファイルを作成し、以下のようにevalのエントリーを追加してみましょう。

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

また、`args`フィールドは、評価クラスの `__init__` メソッドが受け取るべき引数と一致させる必要があります。

## 評価の実行

最後に評価を実行し、結果を表示します。

```sh
pip install .  # you can omit this if you used `pip install -e .` to install
oaieval gpt-3.5-turbo arithmetic
```

`gpt-3.5-turbo`モデルで実行すると、次のような出力が表示されるはずです（ここでは読みやすくするために出力を少し整理しています）。

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
