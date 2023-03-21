# Evals

EvalはOpenAIモデルを評価するためのフレームワークであり、ベンチマークのオープンソースのレジストリでもある。

Evalを使用すると、以下のような評価を作成し、実行することができます。
- データセットを使ってプロンプトを生成する。
- OpenAIモデルが提供するCompletionの品質を測定し、
- 異なるデータセットやモデルで性能を比較することができます。

Evalでは、できるだけ少ないコードでevalをビルドすることができることを目指しています。開始するには、以下のステップを **順番** に踏むことをお勧めします:
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

たとえばevalで選択したデータを取得したいだけの場合もあります。 その場合次を経由して実現する事が出来ます:
```sh
git lfs fetch --include=evals/registry/data/${your eval}
git lfs pull
```

### Evalの作成

Evalを開発するのであれば、このレポをGitHubから直接クローンし、以下のコマンドで要件をインストールすることをお勧めします。

```sh
pip install -e .
```

また`-e`を使用することで、Evalに加えた変更がすぐに反映され、再インストールの必要がありません。

### Evalの実行

新しいEvalを投稿するのではなく、単にローカル環境で動作させたい場合は、pipでevalsパッケージをインストールできます。

```sh
pip install evals
```

Snowflakeデータベースをお持ちの方、またはセットアップしたい方には、Evalの結果をSnowflakeデータベースにログするオプションを用意しています。このオプションでは、さらに `SNOWFLAKE_ACCOUNT`、`SNOWFLAKE_DATABASE`、`SNOWFLAKE_USERNAME`、`SNOWFLAKE_PASSWORD` 環境変数を指定する必要があります。

## FAQ

Evalを最初から最後までビルドする例などありますか？

- はい ！それらは `examples` フォルダにあります。 
これらのサンプルで起こっていることをより深く理解するために、[build-eval.md] (docs/build-eval.md) にも目を通すことをお勧めします。

Evalを複数の異なる方法で実装している例はありますか？

- はい ！特に、`eval/registry/eval/coqa.yaml`を参照してください。違いをわかりやすくするために、[CoQA](https://stanfordnlp.github.io/coqa/)データセットの小さなサブセットを様々なEvalテンプレートに実装しています。

データを変更したのですが、Evalを実行しても反映さ れません、どうなっているのでしょうか？

- データが `/tmp/filecache` にキャッシュされている可能性があります。このキャッシュを削除してEvalを再実行してみてください。

たくさんのコードがありますが、簡単にEvalを回したいんです。お手伝いしますか？ それとも
私は世界的なプロンプトエンジニアでコードを書かないことにしています。どうすれば私の知見を提供できますか？

- 既存の[evalテンプレート](docs/eval-templates.md)に従って、ベーシックまたはモデルグレードによるEvalをビルドする場合、評価コードを書く必要は全くありません! データをJSONフォーマットで提供し、EvalのパラメータをYAMLで指定するだけです。[build-eval.md](docs/build-eval.md) はこれらのステップを解説していて、 `examples` フォルダにある Jupyter ノートブックでこの手順を補足することができるので、すぐに始めることができます。しかしよいEvalを作るには必然的に、慎重な考察と厳密な実験が必要であることを心に留めておいてください!

## 免責事項

Evalに貢献することで、あなたの評価ロジックとデータをこのリポジトリと同じMITライセンスで公開することに同意したものとみなされます。Evalで使用されるデータをアップロードするための適切な権利を有している必要があります。OpenAIは、このデータを将来的に商品におけるサービス改善に利用する権利を有します。OpenAI Evalへの投稿は、通常の利用規定に従うものとします。https://platform.openai.com/docs/usage-policies.
