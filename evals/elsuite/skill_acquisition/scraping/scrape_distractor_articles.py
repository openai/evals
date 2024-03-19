# %%
import json
import re

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

articles_to_scrape = [
    "https://en.wikipedia.org/wiki/Mosquito",
    "https://en.wikipedia.org/wiki/Mosquito_Coast",
    "https://en.wikipedia.org/wiki/Nicaragua",
    "https://en.wikipedia.org/wiki/Honduras",
    "https://en.wikipedia.org/wiki/Miskito_language",
    "https://en.wikipedia.org/wiki/Miskito_people",
]
dirpath = "evals/registry/data/skill_acquisition/distractor_articles/"


def clean_soup(content):
    for infobox_tag in content.find_all("table", class_="infobox"):
        infobox_tag.decompose()
    for figure_tag in content.find_all("figure"):
        figure_tag.decompose()
    for style_tags in content.find_all("style"):
        style_tags.decompose()
    reflist_div = '<div class="reflist"'
    if reflist_div in str(content):
        reflist_index = str(content).index(reflist_div)
        content = BeautifulSoup(str(content)[:reflist_index], "html.parser")

    return content


def clean_heading_text(
    heading_text,
    strip_list=["a"],
    css_selector_pattern=r"\.\w+[^{}]*\{[^}]*\}",
    whitespace_pattern=r"\s\s+",
):
    text = md(str(heading_text), strip=strip_list)
    text = (
        # text.replace("\n", "")
        text.replace("\u3000", " ")
        .replace("\xa0", " ")
        .replace("| --- |", "")
        .replace("|", "")
    )
    text = re.sub(whitespace_pattern, "", re.sub(css_selector_pattern, "", text))
    return text


for article in articles_to_scrape:
    response = requests.get(article)
    soup = BeautifulSoup(response.text, "html.parser")

    content = soup.find("div", class_="mw-content-ltr mw-parser-output")
    content = clean_soup(content)
    headings = str(content).split("<h2>")

    sections = {}
    for heading_text in headings:
        if "</h2>" not in heading_text:
            sections["Introduction"] = clean_heading_text(heading_text)
            continue
        span = heading_text[: heading_text.index("</h2>")]
        heading_title = BeautifulSoup(span, "html.parser").contents[0].contents[0]
        text = heading_text[heading_text.index("</h2>") + 5 :]
        if heading_title not in ["References", "See also", "External links", "Footnotes"]:
            sections[heading_title] = clean_heading_text(text)

    article_title = article.split("/")[-1]

    print(f"Scraped {article_title} successfully. Headings: {sections.keys()}\n")
    filename = f"{article_title.lower()}.jsonl"

    with open(dirpath + filename, "w") as f:
        for k, v in sections.items():
            f.write(json.dumps({"title": k, "content": v}, ensure_ascii=False) + "\n")

# Separate code to scrape human rights article, as it's in a different format.
with open("human_rights.html", "r") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")
content = soup.find("div", class_="migrated-content")
md_content = md(str(content)).replace("\xa0", " ").replace("\u3000", " ")

with open(dirpath + "human_rights_miskito.jsonl", "w") as f:
    f.write(
        json.dumps(
            {"title": "Declaration of Human Rights in Miskito", "content": md_content},
            ensure_ascii=False,
        )
        + "\n"
    )
