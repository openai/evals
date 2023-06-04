# generate a jsonl where each line is a sample

import json
import os
import string

REGISTRY_PATH = os.path.join(os.path.dirname(__file__), "../evals/registry")


def format(template: str, **kwargs: str) -> str:
    """Format a template string with kwargs."""
    keys = [k[1] for k in string.Formatter().parse(template) if k[1]]
    assert all(k in kwargs for k in keys), f"Required: {keys}, got: {sorted(kwargs)}"
    cur_keys = {k: kwargs[k] for k in keys}
    return template.format(**cur_keys)


YAML = """
mg-{prompt_name}-{subject}:
  id: mg-{prompt_name}-{subject}.dev.v0
  metrics: [accuracy]
mg-{prompt_name}-{subject}.dev.v0:
  class: evals.elsuite.modelgraded.classify:ModelBasedClassify
  args:
    samples_jsonl: test_modelgraded/{prompt_name}_{subject}.jsonl
    eval_type: {eval_type}
    modelgraded_spec: {modelgraded_spec}
""".strip()


unlabeled_prompts = dict(
    humor=dict(
        modelgraded_spec="humor_jp",
        prompt="{subject}で数行の大喜利を書いて。敬語じゃなく関西弁で吉本のM1チャンピョンみたいに書いて。",
        eval_type="cot_classify_jp",
    ),
)


subjects = dict(
    animals=["cat", "bird", "hamster"],
    fruits=["apple", "banana", "orange"],
    people=["Elon Musk", "Bill Gates", "Jeff Bezos"],
    people_jp=[
        "イーロン・マスク",
        "ビル・ゲイツ",
        "ジェフ・ベゾス",
        "アルベルト・アインシュタイン",
        "ニコラ・テスラ",
        "レオナルド・ダ・ヴィンチ",
        "マハトマ・ガンジー",
        "ウィリアム・シェイクスピア",
        "スティーブ・ジョブズ",
        "ビル・ゲイツ",
        "マイケル・ジャクソン",
        "マダム・キュリー",
        "ジョン・F・ケネディ",
        "マーティン・ルーサー・キング・ジュニア",
        "ネルソン・マンデラ",
        "プラトン",
        "アリストテレス",
        "フィリップ・コッタウェイ",
        "ニール・アームストロング",
        "レオ・トルストイ",
        "マルコ・ポーロ",
        "ウィリアム・テル",
        "モーツァルト",
        "ベートーベン",
        "ショパン",
        "ダンテ・アリギエーリ",
        "フランツ・カフカ",
        "ガリレオ・ガリレイ",
        "アイザック・ニュートン",
        "チャールズ・ダーウィン",
        "フリードリヒ・ニーチェ",
        "シェイクスピア",
        "オスカー・ワイルド",
        "アーサー・コナン・ドイル",
        "アガサ・クリスティ",
        "J・K・ローリング",
        "トルーマン・カポーティ",
        "アルフレッド・ヒッチコック",
        "ウォルト・ディズニー",
        "アンディ・ウォーホル",
        "ピカソ",
        "ミケランジェロ",
        "レオナルド・フィボナッチ",
        "アルキメデス",
        "マルコム・X",
        "ジョージ・ワシントン",
        "エイブラハム・リンカーン",
        "フランクリン・D・ルーズベルト",
        "ワシントン・アーヴィング",
        "マーク・トウェイン",
        "フィリップ・K・ディック",
        "ジョージ・オーウェル",
        "トーマス・モア",
        "ハンス・クリスチャン・アンデルセン",
        "グリム兄弟",
        "アレクサンドル・デュマ",
        "ビクトル・ユーゴー",
        "エミール・ゾラ",
        "フランツ・シューベルト",
        "ゲオルク・フリードリヒ・ヘンデル",
        "ヨハン・セバスチャン・バッハ",
        "ルートヴィヒ・ヴァン・ベートーヴェン",
        "ヨハネス・ブラームス",
        "ロベルト・シューマン",
        "ヨハン・シュトラウス2世",
        "イーロン・マスク",
        "スティーブン・ホーキング",
        "リチャード・ファインマン",
        "アラン・チューリング",
        "ニール・デグラス・タイソン",
        "マイケル・ファラデー",
        "スティーブン・スピルバーグ",
        "クリストファー・ノーラン",
        "スタン・リー",
        "ジョージ・ルーカス",
        "ウィリアム・ゴールディング",
        "ジョージ・オーウェル",
        "エルンスト・ヘッケル",
        "ルイ・パスツール",
        "カール・セーガン",
        "アンリ・ベルクソン",
        "ミハイル・バクーニン",
        "ハンス・モルゲンソー",
        "アンドレ・マルロー",
        "シモーヌ・ド・ボーヴォワール",
        "ベルトルト・ブレヒト",
        "ジャン＝ポール・サルトル",
        "フリードリヒ・ヘーゲル",
        "マックス・ウェーバー",
        "マルクス・アウレリウス",
        "レフ・トルストイ",
        "アントン・チェーホフ",
        "フョードル・ドストエフスキー",
        "トルストイ",
        "ウィリアム・フォークナー",
        "エルネスト・ヘミングウェイ",
        "アーサー・ミラー",
        "テネシー・ウィリアムズ",
        "サミュエル・ベケット",
        "ハロルド・ピンター",
        "フランツ・カフカ",
        "ジョージ・バーナード・ショー",
        "ウィリアム・ゴールディング",
        "ジャック・ケルアック",
        "エドガー・アラン・ポー",
        "ハーマン・メルヴィル",
        "ジョセフ・コンラッド",
        "アーサー・コナン・ドイル",
        "ジョン・スタインベック",
        "ジェームズ・ジョイス",
        "バージニア・ウルフ",
        "トマス・マン",
        "フランツ・カフカ",
        "ヘルマン・ヘッセ",
        "ゲオルク・ヴィルヘルム・フリードリヒ・ヘーゲル",
        "エマニュエル・カント",
        "ジャン＝ジャック・ルソー",
        "ジョン・ロック",
        "トマス・ホッブズ",
        "ジョン・デューイ",
        "ジョン・スチュアート・ミル",
        "ニコロ・マキャヴェッリ",
        "モンテスキュー",
        "ルソー",
        "プラトン",
        "アリストテレス",
        "サー・アイザック・ニュートン",
    ],
)
# remove duplicates
subjects = {k: list(set(v)) for k, v in subjects.items()}

unlabeled_target_sets = [
    ("humor", "people_jp"),
]

data_dir = f"{REGISTRY_PATH}/data/test_modelgraded"
yaml_str = f"# This file is generated by {os.path.basename(__file__)}\n\n"
evals = []
for prompt_name, subject in unlabeled_target_sets:
    prompt = unlabeled_prompts[prompt_name]["prompt"]
    samples = [{"input": format(prompt, subject=s)} for s in subjects[subject]]
    file_name = f"{data_dir}/{prompt_name}_{subject}.jsonl"
    # save samples jsonl
    with open(file_name, "wb") as f:
        for sample in samples:
            # f.write(json.dumps(sample) + "\n")
            json_data = json.dumps(sample, ensure_ascii=False)
            f.write(json_data.encode("utf-8"))
            f.write(b"\n")
    print(f"wrote {len(samples)} samples to {file_name}")
    yaml_str += (
        YAML.format(
            prompt_name=prompt_name,
            subject=subject,
            modelgraded_spec=unlabeled_prompts[prompt_name]["modelgraded_spec"],
            eval_type=unlabeled_prompts[prompt_name]["eval_type"],
        )
        + "\n\n"
    )
    evals += [f"mg-{prompt_name}-{subject}: {file_name}"]


yaml_file = f"{REGISTRY_PATH}/evals/test-modelgraded-generated.yaml"
with open(yaml_file, "w") as yf:
    yf.write(yaml_str)
print(f"wrote {yaml_file}")
for e in evals:
    print(e)
