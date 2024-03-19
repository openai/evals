# %%
import json

import bs4
import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# TODO: make sure italicised text is crawled properly and that hints are excluded from answers. 
# TODO: Split any multi-part questions into individual questions.

miskito_base_url = "https://en.wikibooks.org/wiki/Miskito/Lesson_{idx}"


def process_practice_section_div(practice_div: bs4.element.Tag):
    tds = practice_div.find_all("td")
    instructions = (
        md(str(tds[1]))
        .replace("*", "")
        .replace("|", "")
        .strip()
        .replace("What do these mean?", "Translate to English:")
        .replace("What do these sentences mean?", "Translate to English:")
    )
    question_text = tds[2]
    questions = question_text.find_all("li")
    questions = [str(q.contents[0]) for q in questions]
    answer_text = tds[3]
    answers = answer_text.find_all("li")
    answers = [str(a.contents[0]) for a in answers]
    return instructions, questions, answers


def extract_toc_sections(content: bs4.element.Tag):
    toc = content.find_all("div", class_="toc")[0]
    lis = toc.find_all("li", class_="toclevel-1")
    lis = [li.find_all("span", class_="toctext")[0].contents[0] for li in lis]

    lis = [md(str(li)).strip().replace("*", "") for li in lis]
    return lis


def process_miskito_page():
    qa_pairs_by_lesson = {}
    articles_without_qa_pairs = []
    for idx in range(1, 11):
        response = requests.get(miskito_base_url.format(idx=idx))
        soup = BeautifulSoup(response.text, "html.parser")
        content = soup.find("div", class_="mw-content-ltr mw-parser-output")

        # Extract the question-answer pairs.
        divs_with_specific_style = content.find_all(
            "div", style=lambda value: value and "width:300px; float:right;" in value
        )
        lesson_qa_pairs = []
        for i, div in enumerate(divs_with_specific_style):
            if i == 0 and idx == 1:  # First section of first lesson is not in the same format.
                instructions = "Translate to English:"
                questions = div.find_all("ul")[0].find_all("li")
                questions = [str(q.contents[0]) for q in questions]
                answers = div.find_all("ul")[1].find_all("li")
                answers = [str(a.contents[0]) for a in answers]
                lesson_qa_pairs += [
                    {"question": q, "answer": a, "instructions": instructions}
                    for q, a in zip(questions, answers)
                ]
                continue
            instructions, questions, answers = process_practice_section_div(div)
            for q, a in zip(questions, answers):
                lesson_qa_pairs += [{"question": q, "answer": a, "instructions": instructions}]
        qa_pairs_by_lesson[f"lesson_{idx}"] = lesson_qa_pairs

        # Remove them from the page and store the page contents.
        for div in divs_with_specific_style:
            div.decompose()

        articles_without_qa_pairs += [content]

    return qa_pairs_by_lesson, articles_without_qa_pairs


# %%
# Write to file: all questions by lesson, and all questions in evallib format.
qa_pairs_by_lesson, clean_articles = process_miskito_page()
qa_by_lesson_file = "miskito_qa_pairs_by_lesson.jsonl"

with open(qa_by_lesson_file, "w") as f:
    for lesson, qa_pairs in qa_pairs_by_lesson.items():
        f.write(json.dumps({"lesson": lesson, "qa_pairs": qa_pairs}) + "\n")

miskito_qa = "miskito_qa.jsonl"
with open(miskito_qa, "w") as f:
    for lesson, qa_list in qa_pairs_by_lesson.items():
        for qa_dict in qa_list:
            instructions = qa_dict["instructions"][:-1] + ": "
            f.write(
                json.dumps(
                    {
                        "input": [{"role": "user", "content": instructions + qa_dict["question"]}],
                        "ideal": qa_dict["answer"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
# %%
as_text = [str(a).split("<h2>")[1:] for a in clean_articles]
sections_by_heading = {}
for article in as_text:
    for heading in article:
        hsoup = BeautifulSoup(heading, "html.parser")
        heading_name = (
            md(str(hsoup.find("span", class_="mw-headline").contents[0])).replace("*", "").strip()
        )
        hsoup.find("span", class_="mw-editsection").decompose()
        content = (
            md(str(hsoup))
            .strip()
            .replace("*", "")
            .replace("|", "")
            .replace("What do they mean?", "")
            .replace(" --- ", "")
            .replace("\u2003", " ")
            .replace("     ", " ")
        )
        content = content.split(" Study ")[1] if "Study " in content else content
        sections_by_heading[heading_name] = content.strip()

sections_by_heading
# %%
file = "lessons_no_exercises.jsonl"
with open(file, "w") as f:
    for heading, content in sections_by_heading.items():
        f.write(json.dumps({"title": heading, "content": content}, ensure_ascii=False) + "\n")
# %%
