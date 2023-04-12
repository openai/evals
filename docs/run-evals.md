# Evalの実行方法について

単一のEvalを実行するための`oaieval`と、一連のEvalを実行するための`oaievalset`という2つのコマンドラインインターフェース(CLI)を提供しています。

## Evalの実行

`oaieval`コマンドを使用する場合、評価したいモデルと実行するevalの両方を提供する必要があります。例えば
```sh
oaieval gpt-3.5-turbo test-match
```

この例では、`gpt-3.5-turbo`が評価するモデルで、`test-match`が実行するEvalです。有効なモデル名は、API経由でアクセスできるものです。有効なEval名は `evals/registry/evals` にあるYAMLファイルで指定され、対応する実装は `evals/elsuite` で確認することが出来ます。

これらのCLIはデフォルトの動作を変更するために、さまざまなフラグを受け付けます。例として以下のようなものがあります:
- Snowflakeデータベース（[README](../README.md)に記載されている通りに設定済み）にログを記録したい場合 は `--no-local-run` を追加します。
- デフォルトでは、 ローカルまたはSnowflakeへのログは `tmp/evallogs` に書き込まれますが、異なる `--record_path` を設定することで、これを変更することができます。

`oaieval --help`を実行すると、CLIオプションのすべてのリストを見ることができます。

## Evalセットの実行

```sh
oaievalset gpt-3.5-turbo test
```

同様に、`oaievalset`もモデル名とEvalセット名を必要とし、有効なオプションは `evals/registry/eval_sets` 以下のYAMLファイルに記述されています。

デフォルトでは10スレッドで実行されます。例えば以下のように設定することができます。

```sh
EVALS_THREADS=42 oaievalset gpt-3.5-turbo test
```

もし実行を中止したり、クラッシュしたりすることがあっても、大丈夫です!  `oaievalset`は終了したEvalを `/tmp/oaievalset/{model}.{eval_set}.progress.txt` に記録しています。コマンドを再実行するだけで、前回の続きから再開することができます。 もし、Evalセットを最初から実行したい場合は、このプログレスファイルを削除してください。

残念ながら、一つのEvalを途中から再開することはできません。 最初からやり直すことになるので、個々のEvalを素早く実行するように心がけましょう。
