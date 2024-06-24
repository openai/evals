import json
import os
import random
import re

import pandas as pd

SEED = 42


def load_quran_data(file_path):
    return pd.read_csv(file_path, header=None, names=["chapter", "verse", "text"], sep="|")


def load_chapter_names(file_path):
    return pd.read_json(file_path)


def extract_random_ayas(df, number_of_ayas):
    random.seed(SEED)
    return df.sample(n=number_of_ayas, random_state=SEED)


def load_distractors(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        distractors = json.load(file)
    return distractors


def generate_mcq_questions(aya, distractors_list, n=3):
    random.seed(SEED)
    correct_answer = aya["text"]
    valid_distractors = [d for d in distractors_list if d != correct_answer]
    selected_distractors = random.sample(valid_distractors, n)

    options = selected_distractors + [correct_answer]
    random.shuffle(options)

    option_labels = ["A", "B", "C", "D"][: len(options)]
    labeled_options = {label: option for label, option in zip(option_labels, options)}

    options_text = "; ".join([f"{label}. {option}" for label, option in labeled_options.items()])
    question_content_en = f"Which of the following is a text from the Quran? {options_text}, please answer with the letter of the correct option (A, B, C, D) only"
    question_content_ar = f"أي من التالي هو نص من القرآن؟ {options_text}, يرجى الإجابة بحرف الخيار الصحيح (A, B, C, D) فقط"

    correct_label = [
        label for label, option in labeled_options.items() if option == correct_answer
    ][0]

    return question_content_en, question_content_ar, correct_label


def redact_aya(text, all_texts):
    random.seed(SEED)
    words = text.split()
    if len(words) <= 1:
        return None

    for _ in range(100):
        start = random.randint(0, len(words) - 1)
        end = random.randint(start + 1, len(words))
        first_section = " ".join(words[:start])
        missing_section = " ".join(words[start:end])
        third_section = " ".join(words[end:])
        redacted_aya = f"{first_section} ________ {third_section}".strip()

        pattern = re.escape(first_section) + ".*" + re.escape(third_section)
        if len([t for t in all_texts if re.match(pattern, t)]) == 1:
            return redacted_aya, first_section, missing_section, third_section

    return None


def generate_bilingual_questions(ayas_df, question_type):
    random.seed(SEED)
    bilingual_questions = []
    half_length = len(ayas_df) // 2
    include_extra_info = True

    for index, row in ayas_df.iterrows():
        extra_info_en = (
            f" This text is from Surah {row['name']} (Chapter {row['chapter']}, Verse {row['verse']})."
            if include_extra_info
            else ""
        )
        extra_info_ar = (
            f" هذا النص القرآني من سورة {row['name']} (السورة {row['chapter']}، الآية {row['verse']})."
            if include_extra_info
            else ""
        )

        if question_type == "missing_text":
            question_content_en = f"Fill in the blank of the following Quranic text: (({row['redacted']})) to complete the full verse.{extra_info_en}.  The answer may be one or more words."
            question_content_ar = f"املأ الفراغ في النص القرآني التالي: (({row['redacted']})) لإتمام الآية كاملة.{extra_info_ar}. قد تكون الإجابة عبارة عن كلمة واحدة أو أكثر."
            ideal_answer = [row["missing_section"]]
            ideal_answer_ar = [row["missing_section"]]

        elif question_type == "surah_name":
            question_content_en = f"Identify the Surah (in Arabic) of the following Quranic text: {row['text']} (Please provide the answer without diacritics but keep hamza and madda)."
            question_content_ar = f"حدد اسم السورة للنص القرآني التالي: {row['text']} (يرجى تقديم الإجابة بدون تشكيل ولكن احتفظ بالهمزة والمد)."
            ideal_answer = [row["name"], row["transliteration"], row["translation"]]
            ideal_answer_ar = [row["name"], row["transliteration"], row["translation"]]

        elif question_type == "surah_type":
            question_content_en = f"Determine if the Surah of the following Quranic aya text is meccan or medinan: {row['text']} answer only with either 'meccan' or 'medinan' (exactly in small case)."
            question_content_ar = f"حدد إذا كانت السورة للنص القرآني التالي مكية أو مدنية: {row['text']} أجب فقط بـ 'مكية' أو 'مدنية' (بدون تشكيل)."
            answer_arabic_translations = (
                ["مكية", "مكي", "مكة"] if row["type"] == "meccan" else ["مدنية", "مدني", "المدينة"]
            )
            (
                ["meccan", "meccan", "mecca", "maccan"]
                if row["type"] == "meccan"
                else ["madinan", "medinan", "madina"]
            )
            all_answers = [row["type"]] + answer_arabic_translations
            ideal_answer = all_answers
            ideal_answer_ar = all_answers

        elif question_type == "mcq":
            question_content_en, question_content_ar, correct_label = generate_mcq_questions(
                row, distractors_list
            )
            ideal_answer = [correct_label]
            ideal_answer_ar = [correct_label]

        # Creating questions in both English and Arabic
        if index < half_length:  # English questions
            bilingual_questions.append(
                {
                    "input": [
                        {"role": "system", "content": question_content_en},
                        {
                            "role": "user",
                            "content": "Please provide the answer, and ONLY the answer without any extra commentary"
                            if question_type != "mcq"
                            else "Please provide the answer by selecting the correct letter (A, B, C, or D) without any extra commentary",
                        },
                    ],
                    "ideal": ideal_answer,
                }
            )
        else:  # Arabic questions
            bilingual_questions.append(
                {
                    "input": [
                        {"role": "system", "content": question_content_ar},
                        {
                            "role": "user",
                            "content": "يرجى تقديم الإجابة. وفقط الإجابة دون أي تعليق إضافي"
                            if question_type != "mcq"
                            else "يرجى تقديم الإجابة عن طريق تحديد الحرف الصحيح (A, B, C, أو D) دون أي تعليق إضافي",
                        },
                    ],
                    "ideal": ideal_answer_ar,
                }
            )

        # Toggle extra info for next question
        include_extra_info = not include_extra_info

    return bilingual_questions


if __name__ == "__main__":
    # Main process
    quran_file_path = "evals/registry/data/quran_eval/gen_script/resources/Arabic-Original.csv"
    chapters_file_path = "evals/registry/data/quran_eval/gen_script/resources/chapters-en.json"
    distractors_file_path = (
        "evals/registry/data/quran_eval/gen_script/resources/distractors_not_quranic.json"
    )

    random.seed(SEED)

    # Load and prepare data
    quran_df = load_quran_data(quran_file_path)
    chapters_df = load_chapter_names(chapters_file_path)
    random_ayas_df = extract_random_ayas(quran_df, 350)
    distractors_list = load_distractors(distractors_file_path)

    random_ayas_df = random_ayas_df.merge(chapters_df, left_on="chapter", right_on="id")
    random_ayas_df.drop(columns=["id", "total_verses"], inplace=True)

    # Apply the redaction process and validation
    all_texts = quran_df["text"].tolist()
    validated_ayas = []

    for index, row in random_ayas_df.iterrows():
        result = redact_aya(row["text"], all_texts)
        if result:
            (
                row["redacted"],
                row["first_section"],
                row["missing_section"],
                row["third_section"],
            ) = result
            pattern = row["text"]
            if len([t for t in all_texts if re.match(pattern, t)]) == 1:
                validated_ayas.append(row)

    validated_ayas_df = pd.DataFrame(validated_ayas)

    # Generate bilingual questions
    bilingual_missing_text_questions = generate_bilingual_questions(
        validated_ayas_df, "missing_text"
    )
    bilingual_surah_name_questions = generate_bilingual_questions(validated_ayas_df, "surah_name")
    bilingual_surah_type_questions = generate_bilingual_questions(validated_ayas_df, "surah_type")
    # Generate MCQ questions
    question_type = "mcq"
    mcq_questions = generate_bilingual_questions(random_ayas_df, question_type)

    # Save the questions to separate JSON files
    readable_bilingual_missing_text_file_path = (
        "evals/registry/data/quran_eval/gen_script/generated/masked_quranic_text.json"
    )
    readable_bilingual_surah_name_file_path = (
        "evals/registry/data/quran_eval/gen_script/generated/guess_quran_surah_name.json"
    )
    readable_bilingual_surah_type_file_path = (
        "evals/registry/data/quran_eval/gen_script/generated/guess_quran_surah_type.json"
    )
    readable_biligual_questions_mcq_file_path = (
        "evals/registry/data/quran_eval/gen_script/generated/guess_which_text_is_from_quran.json"
    )

    output_folder = "evals/registry/data/quran_eval/gen_script/generated"

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    with open(readable_bilingual_missing_text_file_path, "w", encoding="utf-8") as file:
        json.dump(bilingual_missing_text_questions, file, ensure_ascii=False, indent=4)

    with open(readable_bilingual_surah_name_file_path, "w", encoding="utf-8") as file:
        json.dump(bilingual_surah_name_questions, file, ensure_ascii=False, indent=4)

    with open(readable_bilingual_surah_type_file_path, "w", encoding="utf-8") as file:
        json.dump(bilingual_surah_type_questions, file, ensure_ascii=False, indent=4)

    with open(readable_biligual_questions_mcq_file_path, "w", encoding="utf-8") as file:
        json.dump(mcq_questions, file, ensure_ascii=False, indent=4)

    # Final output paths for each question type
    missing_text_output_jsonl = "evals/registry/data/quran_eval/masked_quranic_text.jsonl"
    surah_name_output_jsonl = "evals/registry/data/quran_eval/guess_quran_surah_name.jsonl"
    surah_type_output_jsonl = "evals/registry/data/quran_eval/guess_quran_surah_type.jsonl"
    mcq_output_jsonl = "evals/registry/data/quran_eval/guess_which_text_is_from_quran.jsonl"

    output_folder = "evals/registry/data/quran_eval"

    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Save the questions in JSON Lines format for each question type
    with open(missing_text_output_jsonl, "w", encoding="utf-8") as file:
        for question in bilingual_missing_text_questions:
            # Write each question as a separate line in the file
            json_line = json.dumps(question, ensure_ascii=False)
            file.write(json_line + "\n")

    with open(surah_name_output_jsonl, "w", encoding="utf-8") as file:
        for question in bilingual_surah_name_questions:
            # Write each question as a separate line in the file
            json_line = json.dumps(question, ensure_ascii=False)
            file.write(json_line + "\n")

    with open(surah_type_output_jsonl, "w", encoding="utf-8") as file:
        for question in bilingual_surah_type_questions:
            # Write each question as a separate line in the file
            json_line = json.dumps(question, ensure_ascii=False)
            file.write(json_line + "\n")

    with open(mcq_output_jsonl, "w", encoding="utf-8") as file:
        for question in mcq_questions:
            # Write each question as a separate line in the file
            json_line = json.dumps(question, ensure_ascii=False)
            file.write(json_line + "\n")
