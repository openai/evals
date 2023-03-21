# Evalをビルドする

このドキュメントでは、Evalをビルドするときの端から端までのプロセス（データセットとEvalクラスの選択）を解説しています。`examples`フォルダには、以下のステップに従っていくつかの学術的なEvalをビルドするJupyterノートブックが含まれており、全体のプロセスを説明するのに役立ちます。

このプロセスのステップは、 データセット内にevalをビルドする、 そのデータを使い新しいEvalを登録する、そしてそのEvalを実行するというものです。重要なのは、あなたが[既存のevalテンプレート](eval-templates.md)をそのまま使っていると仮定していることです（そうでない場合は、[カスタムevalをビルドする例](custom-eval.md)をご参照ください）。もし、あなたが自分のevalを公開することに興味があるのであれば、私たちが考える興味深いEvalの基準も下部に記載しています。

以下のカテゴリーでEvalを募集しています。

- 過剰な拒否反応
- 安全性
- システムメッセージの操作可能性
- 既に広まっているhallucination。
- 数学／論理／物理的な推論
- 現実のユースケース（この機能が商品でどのように使用されるかをPRで記述してください。）
- その他の基礎的能力

このカテゴリーから外れていても、多様な事例となるEvalがあれば、ぜひ投稿してください。

## データをフォーマットする

実装したいEvalが決まったら、サンプルを正しいJSON行（JSONL）フォーマットに変換する必要があります。JSONLファイルは、1行に1つのユニークなJSONオブジェクトを持つ、ただのJSONファイルです。

JSONLのEvalファイルの例を[registry/data/README.md](../eval/registry/data/README.md)で紹介しています。

各JSONオブジェクトは、Evalの1つのデータポイントを表します。JSONオブジェクトに必要なキーは、Evalのテンプレートに依存します。すべてのテンプレートで `"input"` キーが必要とされ、プロンプトは理想的には [chat format](https://platform.openai.com/docs/guides/chat/introduction) で指定されているものとします (文字列もサポートされています)。チャット以外のモデルを評価する場合でも、チャットフォーマットを推奨します。チャットモデルと非チャットモデルの両方を評価する場合、チャットフォーマットのプロンプトと生の文字列プロンプトの間での変換を行います（変換ロジックは [こちら](../eval/prompt/base.py) を参照してください）。

基本的なEvalである `Match`、`Includes`、`FuzzyMatch` において、その他のキーとして必要なものは `"ideal"` で、これは正しい基準解答を指定する文字列 (または文字列のリスト) です。モデルグレードによる評価では、必要なキーは Eval によって異なりますが、（オプションの） `args` でカバーできない `{key}` を `prompt` で指定することで決定されます。

[CoQA](https://stanfordnlp.github.io/coqa/)データセットの小さなサブセットを様々なEvalテンプレートに実装し、データがどのようにフォーマットされる必要があるかを説明しました。基本的な評価テンプレートである `Match` に適したデータの例として [`coqa/match.jsonl`](../evals/registry/data/coqa/match.jsonl) を、`fact` と `closedqa` モデルグレードによる評価に適したデータとして [`coqa/samples.jsonl`](../evals/registry/data/coqa/samples.jsonl) を参考にしています。
この2つのモデルグレードによる評価では、異なるキーを想定していますが、両方のEvalをサポートするために、データにキーのスーパーセットを含めることができることに注意してください。

データセットファイルがローカルマシン上にある場合は、`jsonl`ファイルを `evals/registry/evals/data/<eval_name>/samples.jsonl` に置いてください。Cloud Object Storageにある場合は、主要なクラウドの パス・スタイルのURLをサポートします（個人的な利用に限定し、クラウドURLでのPRは受け付けません）。

## Evalの登録について

elsuiteのレジストリフォーマットを用いて、`eval/registry/eval/<eval_name>.yaml`にファイルを追加して、Evalを登録します。例えば、`Match`のEvalの場合、次のようになります:
```
<eval_name>:
  id: <eval_name>.dev.v0
  metrics: [accuracy]

<eval_name>.dev.v0:
  class: evals.elsuite.basic.match:Match
  args:
    samples_jsonl: <eval_name>/samples.jsonl
```

例えば、`test_match/samples.jsonl`を提供した場合、データは `evals/registry/data/test_match/samples.jsonl` にあることが期待されます。

Evalの命名規則は、`<eval_name>.<split>.<version>`という形式になっています。
- `<eval_name>`はevalの名前で、スコアが同程度のEvalをグループ化するために使用します。
- 同じ`<base_eval>`の下にあるevalをさらにグループ化するために使用される`<split>`はデータスプリットです。例えば、"val"、"test"、"dev" などをテスト用に使用します。
- `<version>` はEvalのバージョンで、任意の説明テキストを使用することができます（ただし、". "を含まないことが望ましいです）。

一般的に、同じモデルに対して同じEvalを実行すると、常に同じような結果が得られ、他の人がそれを再現できるようになります。従って、Evalを変更した場合は、バージョンを上げる必要があります。

## Evalの実行

これで、CLIからデータを使って、好きなモデルでEvalを実行することができるようになりました。
```
oaieval gpt-3.5-turbo <eval_name>
```
おめでとうございます！Evalをビルドすることができました。結果に自信が持てるようになるまで、繰り返し練習してください。データファイルを変更した場合は、`/tmp/filecache`を削除して、更新されたデータでEvalを実行することを忘れないでください。

## モデルグレードによる評価のために：ステップバイステップのワークフロー

我々は、`fact`、`closedqa`、`battle`といった既存のモデルグレードによる評価が、多くのユースケースに適合することを期待しています。しかし、他のユースケースでは、例えば異なる評価プロンプトのような、より多くのカスタマイズのメリットを享受できるかもしれません。このような場合は、もう少し作業が必要になりますが、通常はコーディングの必要はありません。

1. 既存のモデルグレードによる評価が使えない場合は、`evals/registry/modelgraded`に新しいYAMLを作成して、評価の[パラメータ](eval-templates.md#parameters for-model-graded-eval) を指定してください。例として [`humor.yaml`](../evals/registry/modelgraded/humor.yaml) を見てください。
    - 新しい YAML を作成する場合でも、出発点として既存の YAML をコピーするのが一番簡単だと気づくかもしれないことに注目してください。例えば、モデルの完成度をルールに基づいてチェックするモデルグレードのEvalでは `closedqa.yaml` をコピーして `args` を編集するだけでよいでしょう。
2. 次に、上記のようにデータセットを作成し、Evalを登録します。例えば、[`joke_fruits_labeled.jsonl`](../eval/registry/data/test_metaeval/joke_fruits_labeled.jsonl)や[`joke-fruits`](../eval/registry/eval/test-modelgraded.yaml)などを見てください。
    - なお、Evalを登録する際には、ステップ1ではなく、このステップの時点で`eval_type`を指定することが推奨されます。
3. 例えば`oaleval gpt-3.5-turbo joke-fruits`のように、Evalを実行します。
4. (推奨) モデルグレードの評価用のメタ評価を追加する! モデルグレードの評価には、主に `prompt` や `eval_type` など、調整するためのノブがいくつかあります。モデルグレードの評価を高品質にするために、モデルグレードの評価には "choice labels "をつけることを推奨します。choice labelsとは、モデルがどの評価を選択すべきかを示す、人間が提供するラベルのことです。例として、(これらのジョークが実際に面白いということにして) [`joke_fruits_labeled.jsonl`](../evals/registry/data/test_metaeval/joke_fruits_labeled.) の `"choice"` キーを見てください。 jsonl)、これは `joke-fruits` Eval では使用されませんが、そのすぐ下の [`joke-fruits-meta`](../evals/registry/evals/test-modelgraded.yaml) メタEval では使用されます . メタ評価の実行後、例えば `oaieval gpt-3.5-turbo joke-fruits-meta` とすると、レポートには `metascore/` という精度が出力されますが、これは良いモデルグレードの評価では "1.0" に近い精度になるはずです。

## Evalを投稿する基準

重要: コードを提供する場合、`black`, `isort`, `autoflake` が実行されていることを確認するために、コミットおよびプッシュする前に `pip install pre-commit; pre-commit install` を実行していることを確認してください。

今後、モデルを改善するために、多様で興味深いEvalを収集することに関心があります。ここでは、私たちが考えるよいEvalの基準をいくつか紹介します。
- [ ] Evalはテーマごとに一貫性があることが望ましい。同じユースケース、テーマ分野、失敗例などを中心に、多くのプロンプトが展開されることを望みます。
- [ ] Evalはチャレンジングであるべきだ。GPT-4やGPT-3.5-Turboがすべてのプロンプトでうまくいくと、これはあまり面白くない。もちろん、モデルの限界や制約を考慮した上で、Evalは実現可能であるべきです。多くの場合、人間（専門家の可能性もある）がそのプロンプトをうまく処理できるかどうかが、良い経験則となります。
- [ ] Evalの方向性は明確であるべきです。データには、何が正しい行動なのかを示す適切なシグナルが含まれていなければなりません。これは、例えば、高品質の模範解答や、解答を評価するための包括的な基準を意味します。
- [ ] Evalは慎重に作成する必要があります。提出する前に、よいパフォーマンスが得られるようにプロンプトを工夫したか、最適なEvalテンプレートを使用しているか、結果の正確性を確認するために抜き打ちチェックを行ったか、などをよく考えてみてください。

Evalを公開する準備ができたら、PRを提出してください。OpenAIチームは、喜んでそのEvalに目を通します。 PRメッセージに事前に入力されているテンプレートの全ての項目が記入されていることを確認してください。PRを提出しても、OpenAIが最終的にマージすることを保証するものではないことにご注意ください。どのEvalをフォローアップするか検討する際に、私たちは独自のチェックを実行し、最善の判断を下します。
