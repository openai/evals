from tqdm import tqdm
import json
import re
import mwxml
import mwparserfromhell

dump_path = "dewiktionary-20230520-pages-articles-multistream.xml"
total_pages = 1215724  # correct value
# total_pages = 40000  # for testing

# mapping part of speech labels
pos_mapping = {
    'adjektiv': 'adjective',
    'adverb': 'adverb',
    'antwortpartikel': 'particle',
    'artikel': 'article',
    'dekliniertes gerundivum': 'adjective',
    'demonstrativpronomen': 'pronoun',
    'erweiterter infinitiv': 'verb',
    'fokuspartikel': 'particle',
    'gradpartikel': 'particle',
    'hilfsverb': 'verb',
    'indefinitpronomen': 'pronoun',
    'interjektion': 'interjection',
    'interrogativadverb': 'adverb',
    'interrogativpronomen': 'pronoun',
    'komparativ': 'adjective',
    'konjugierte form': 'verb',
    'konjunktion': 'conjunction',
    'konjunktionaladverb': 'adverb',
    'lokaladverb': 'adverb',
    'modaladverb': 'adverb',
    'modalpartikel': 'particle',
    'negationspartikel': 'particle',
    'partikel': 'particle',
    'partizip i': 'adjective',
    'partizip ii': 'adjective',
    'personalpronomen': 'pronoun',
    'possessivpronomen': 'pronoun',
    'postposition': 'preposition',
    'präposition': 'preposition',
    'pronomen': 'pronoun',
    'pronominaladverb': 'adverb',
    'reflexivpronomen': 'pronoun',
    'relativpronomen': 'pronoun',
    'reziprokpronomen': 'pronoun',
    'subjunktion': 'conjunction',
    'substantiv': 'noun',
    'superlativ': 'adjective',
    'temporaladverb': 'adverb',
    'verb': 'verb',
    'vergleichspartikel': 'particle',
}
possible_pos = sorted(list(set(pos_mapping.values())))
print(possible_pos)
inflection_mapping = {
    'dekliniertes gerundivum': True,
    'erweiterter infinitiv': True,
    'konjugierte form': True,
    'partizip i': True,
    'partizip ii': True
}

# 'deklinierte form',
# komparativ, superlativ

# iterate over all pages in the dump and print the title
all_words = {}
pages = mwxml.Dump.from_file(open(dump_path, 'rb')).pages
count = 0
for page in tqdm(pages, total=total_pages):
    count += 1
    # Skip titles with one of these characters: " ", "’", "'", "-", "."
    if " " in page.title or "’" in page.title or "'" in page.title or "-" in page.title or "." in page.title:
        continue
    if not any(char.isalpha() for char in page.title):
        continue

    # Parse the page wikicode content
    last_revision = next(page)

    wikicode = mwparserfromhell.parse(last_revision.text)

    # Find the language section 'fr'
    # Regex for {{langue|fr}}
    de_sections = wikicode.get_sections(matches=r"{{Sprache\|Deutsch}}")

    parts_of_speech = {}
    please_skip = False
    for de_section in de_sections:
        # Find the headlines with 'S' templates in 'fr_section' only
        all_headlines = de_section.filter_headings()
        for headline in all_headlines:
            if not str(headline).startswith("=== "):
                continue
            parsed_headline = mwparserfromhell.parse(str(headline))
            templates = parsed_headline.filter_templates(matches="Wortart")

            for template in templates:
                # 2nd argument has to be Deutsch
                if len(template.params) < 2:
                    continue
                if template.params[1].value.strip() != "Deutsch":
                    continue

                part_of_speech = template.params[0].value.strip().lower()
                is_inflection = False
                allowed = None
                if part_of_speech in pos_mapping:
                    part_of_speech = [pos_mapping[part_of_speech]]
                elif part_of_speech == "komparativ" or part_of_speech == "superlativ":
                    allowed = ["adjective", "adverb"]
                elif part_of_speech == "deklinierte form":
                    allowed = ["noun", "adjective", "article", "pronoun"]
                else:
                    please_skip = True

                if allowed is not None:
                    is_inflection = True
                    reg = r"\{\{Wortart\|%s\|Deutsch\}\}" % part_of_speech
                    part_of_speech = []
                    section = de_section.get_sections(
                        matches=lambda x: re.search(reg, str(x), re.IGNORECASE))
                    if len(section) == 0:
                        is_inflection = False
                        please_skip = True
                    else:
                        s = str(section[0])
                        s = re.sub(r"\{\{.*?\}\}", "", s, flags=re.DOTALL)
                        for a in allowed:
                            if ((a == "adjective" and re.search(r"adjektiv", s, re.IGNORECASE)) or
                                (a == "adverb" and re.search(r"adverb", s, re.IGNORECASE)) or
                                (a == "noun" and re.search(r"substantiv|nomen", s, re.IGNORECASE)) or
                                (a == "article" and re.search(r"artikel", s, re.IGNORECASE)) or
                                    (a == "pronoun" and re.search(r"pronom", s, re.IGNORECASE))):
                                part_of_speech.append(a)
                            if ((a == "adjective" and re.search(r"adjektiv", page.title, re.IGNORECASE)) or
                                (a == "adverb" and re.search(r"adverb", page.title, re.IGNORECASE)) or
                                (a == "noun" and re.search(r"substantiv|nomen", page.title, re.IGNORECASE)) or
                                (a == "article" and re.search(r"artikel", page.title, re.IGNORECASE)) or
                                    (a == "pronoun" and re.search(r"pronom", page.title, re.IGNORECASE))):
                                please_skip = True

                # Part of speech needs to be an array
                if type(part_of_speech) is list:
                    for pos in part_of_speech:
                        if pos not in parts_of_speech:
                            parts_of_speech[pos] = is_inflection
                        else:
                            parts_of_speech[pos] = parts_of_speech[pos] or is_inflection

    if please_skip:
        continue

    # Create one string of the sorted parts of speech, using _i if it is a inflection
    # Example: noun_verb_f
    pos_string = "_".join([pos + ("_i" if parts_of_speech[pos] else "")
                          for pos in sorted(parts_of_speech.keys())])
    if pos_string == "":
        continue
    if pos_string not in all_words:
        all_words[pos_string] = {
            "pos": parts_of_speech,
            "words": []
        }
    all_words[pos_string]["words"].append(page.title)

    if count > total_pages:
        break

    if count % 2500 == 0:
        with open("out/de/pos.json", "w") as f:
            json.dump(all_words, f, ensure_ascii=False)

with open("out/de/pos.json", "w") as f:
    json.dump(all_words, f, ensure_ascii=False)
print(count)
