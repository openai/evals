# Evals 

Evalsは、OpenAIモデルを評価するためのフレームワークであり、ベンチマークのオープンソースレジストリでもあります。

Evalsを使用して、以下のような評価を作成し、実行することができます。
- プロンプトを生成するためにデータセットを使用する。
- OpenAIモデルが提供する補完の品質を測定し
- 異なるデータセットやモデル間で性能を比較することができます。

Evalsでは、できるだけシンプルに、できるだけ少ないコードでevalを構築できるようにすることを目指しています。まずは、以下のステップを**順番に踏むことをお勧めします。
1. このドキュメントを読み、[以下の設定方法](README.md#Setup)に従います。
2. 既存のevalを実行する方法を学びます。[run-evals.md](docs/run-evals.md).
3. 既存のevalのテンプレートに慣れる。[eval-templates.md](docs/eval-templates.md).
4. evalを作成するプロセスを手順に沿って行う。eval-templates.md (docs/eval-templates.md)
5. カスタムevalロジック実装する例をご覧ください。[custom-eval.md](docs/custom-eval.md) を参照してください。

もし面白いevalがあると思ったら、ぜひPullRequestを開いて投稿してください。OpenAIのスタッフは、次期モデルの改良を検討する際に、これらの評価結果を積極的に検討します。

____________________
🚨 期間限定で、高品質なevalを提供した方にGPT-4へのアクセスを付与します。上記の指示に従ってください。スパムや低品質の提出物は無視されます❗️

アクセスは、受理されたEvalに関連する電子メールアドレスに付与されます。アクセスが集中するため、プルリクエストに使用された電子メール以外の電子メールへのアクセスは許可できません。
____________________

## セットアップ

evalsを実行するには、OpenAI APIキーの設定と指定が必要です。<https://platform.openai.com/account/api-keys>で生成することができます。APIキーを取得したら、環境変数 `OPENAI_API_KEY` を使って指定します。**evalsの実行時にAPIを使用する際の[コスト](https://openai.com/pricing)にご注意ください。**

### evalsのダウンロード

私たちのEvalsレジストリは、[Git-LFS](https://git-lfs.com/)を使用して保存されています。LFSをダウンロードした後インストールしたら、次のようにしてEvalsを取得することができます。
SH
git lfs fetch --all
git lfs pull
```

evalの選択のためにデータをフェッチしたいだけかもしれません。このような場合、以下の方法で実現できます。
以下の方法で実現できます：``sh
git lfs fetch --include=evals/registry/data/${your eval}。
git lfs pull
```

### eval(評価)の作成
eval(評価)を作成する予定の場合、GitHubから直接このリポジトリをクローンし、次のコマンドを使用してevalをインストールすることをお勧めします

``sh
pip install -e .
```
-e オプションを使用すると、eval(評価)に加えた変更がすぐに反映されるため、再インストールする必要がありません。

### エバルの実行

新しいevalsを投稿するのではなく、単にローカルで実行したい場合は、pip経由でevalsパッケージをインストールすることができます。

```sh
pip install evals
```

Snowflakeデータベースをお持ちの場合、または設定したい場合は、評価結果をSnowflakeデータベースに記録するオプションを提供します。このオプションでは、さらに `SNOWFLAKE_ACCOUNT`、`SNOWFLAKE_DATABASE`、`SNOWFLAKE_USERNAME`、`SNOWFLAKE_PASSWORD`環境変数を指定する必要があります。

## FAQ

evalを最初から最後まで構築する例はありますか？

- はい！あります。examples`フォルダの中にあります。これらの例で起こっていることをより深く理解するために、 [build-eval.md] (docs/build-eval.md) も読んでおくことをお勧めします。

複数の異なる方法で実装されたevalの例はありますか？

- はい！あります。特に、`evals/registry/evals/coqa.yaml`を参照してください。違いを説明するのに役立つように、様々なevalテンプレートに対して[CoQA](https://stanfordnlp.github.io/coqa/)データセットの小さなサブセットを実装しています。

データを変更したのですが、evalを実行しても反映されないのですが、どうなっているのでしょうか？

- データが`/tmp/filecache`にキャッシュされた可能性があります。このキャッシュを削除して、evalを再実行してみてください。

コードがたくさんあるので、素早く評価を実行したいのですが。助けてください。だとか

私は世界的なプロンプトエンジニアです。私はコードを書かないことにしています。どうすれば私の知恵を提供できますか？
などなど。。

- 既存の [eval テンプレート](docs/eval-templates.md) に従って基本またはモデルグレードのevalを構築する場合、評価コードを書く必要は全くありません! JSON形式でデータを提供し、YAMLで評価パラメータを指定するだけです。[build-eval.md](docs/build-eval.md)はこれらの手順を説明しています。また、これらの説明を `examples` フォルダ内の Jupyter notebooks で補えば、すぐに始めることができます。しかし、良い評価を行うには、必然的に注意深い考察と厳密な実験が必要であることを心に留めておいてください。

## 免責事項

Evalに貢献することで、あなたの評価ロジックとデータをこのリポジトリと同じMITライセンスで作成することに同意したことになります。Evalで使用されるデータをアップロードするための適切な権利を有している必要があります。OpenAIは、このデータを将来的に製品のサービス向上に使用する権利を有します。OpenAI Evalsへの貢献は、通常の使用ポリシーに従います: https://platform.openai.com/docs/usage-policies.

##　注意
www.DeepL.com/Translator（無料版）および、GPT−４の翻訳を組み合わせて翻訳しました。