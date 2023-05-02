import json
import os
import random
import re
import time

from bs4 import BeautifulSoup
from hangul_utils import join_jamos
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm

empty_pairs_file = 'empty_pairs.txt'

if os.path.isfile(empty_pairs_file):
    with open(empty_pairs_file, 'r', encoding='utf-8') as f:
        empty_pairs = [tuple(line.strip().split(', ')) for line in f.readlines()]
else:
    empty_pairs = []

driver_path = 'C:/webdrivers/chromedriver.exe'
options = Options()
options.add_argument('--headless')
options.add_argument('--log-level=3')

driver = webdriver.Chrome(executable_path=driver_path, options=options)


# Function to update the JSON file
def update_json_file(file_path, expression, same_words):
    data = []
    if os.path.isfile(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
        except json.JSONDecodeError:
            print(f"Error decoding JSON in file {file_path}. Creating a new file.")

    if not isinstance(data, list):
        print(f"Invalid data format in {file_path}. Creating a new list.")
        data = []

    updated = False
    for expression_dict in data:
        if expression_dict['expression'] == expression:
            if 'same_words' not in expression_dict:
                expression_dict['same_words'] = []
            # Check if 'similar_words' key exists in expression_dict
            # if 'similar_words' not in expression_dict:
            #     expression_dict['similar_words'] = []
            expression_dict['same_words'].extend(same_words)
            # expression_dict['similar_words'].extend(similar_words)
            updated = True
            break

    if not updated:
        new_expression_data = {
            'expression': expression,
            'same_words': same_words,
            # 'similar_words': similar_words
        }
        data.append(new_expression_data)

    with open(file_path, 'w', encoding='utf-8') as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=4)


while True:
    category = input("Enter the category you want to scrape (proverb, phrase, idiom): ").strip().lower()
    if category in ['proverb', 'phrase', 'idiom']:
        break
    else:
        print("Invalid category. Please enter 'proverb', 'phrase', or 'idiom'.")

while True:
    output_dir = input("Enter the directory where you want to save the scraped data: ").strip()
    try:
        os.makedirs(output_dir, exist_ok=True)
        break
    except Exception as e:
        print(f"Error creating directory: {e}")
        print("Please try again.")

while True:
    test_mode = input("Do you want to enable test mode? (yes, no): ").strip().lower()
    sample_size = 1
    if test_mode == 'yes':
        while True:
            try:
                sample_size = int(input("Enter the sample size: "))
                if sample_size > 0:
                    break
            except ValueError:
                print("Invalid input. Please enter a positive integer.")

    if test_mode in ['yes', 'no']:
        break
    else:
        print("Invalid input. Please enter 'yes' or 'no'.")

if test_mode == 'yes':
    while True:
        test_sample_type = input("Select test sample type (random, specific): ").strip().lower()
        if test_sample_type in ['random', 'specific']:
            break
        else:
            print("Invalid input. Please enter 'random' or 'specific'.")

consonants = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
vowels = ['ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']

while True:
    order = input("Enter the order you want to scrape (alphabetical, random): ").strip().lower()
    if order in ['alphabetical', 'random']:
        break
    else:
        print("Invalid order. Please enter 'alphabetical' or 'random'.")

if test_mode == 'yes':
    if test_sample_type == 'random':
        char_combinations = [(consonant, vowel) for consonant in consonants for vowel in vowels if
                             (consonant, vowel) not in empty_pairs]
        random.shuffle(char_combinations)
        char_combinations = char_combinations[:sample_size]
    elif test_sample_type == 'specific':
        specific_consonant = input("Enter specific consonant (e.g., ㄱ, ㄴ, ㄷ): ").strip()
        specific_vowel = input("Enter specific vowel (e.g., ㅏ, ㅑ, ㅓ): ").strip()
        char_combinations = [(specific_consonant, specific_vowel)]
else:
    starting_consonant = input("Enter the starting consonant (e.g., ㄱ, ㄴ, ㄷ): ").strip()
    starting_vowel = input("Enter the starting vowel (e.g., ㅏ, ㅑ, ㅓ): ").strip()

    char_combinations = []
    found_starting_pair = False

    for consonant in consonants:
        for vowel in vowels:
            if not found_starting_pair:
                if consonant == starting_consonant and vowel == starting_vowel:
                    found_starting_pair = True
            if found_starting_pair and (consonant, vowel) not in empty_pairs:
                char_combinations.append((consonant, vowel))

if order == 'random':
    random.shuffle(char_combinations)

# Find the correct starting index
starting_index = -1
for idx, (consonant, vowel) in enumerate(char_combinations):
    if consonant == starting_consonant and vowel == starting_vowel:
        starting_index = idx
        break

if starting_index >= 0:
    char_combinations = char_combinations[starting_index:]
else:
    tqdm.write(f"Starting consonant-vowel pair ({starting_consonant}, {starting_vowel}) not found in the list.")
    tqdm.write("Using the first consonant-vowel pair as the starting point.")
    starting_index = 0

starting_page = None
while starting_page is None or starting_page <= 0:
    try:
        starting_page = int(input("Enter the starting page number: "))
    except ValueError:
        print("Invalid input. Please enter a positive integer.")

while True:
    try:
        retry_attempts = int(input("Enter the number of retry attempts: "))
        if retry_attempts > 0:
            break
    except ValueError:
        print("Invalid input. Please enter a positive integer.")

for consonant, vowel in tqdm(char_combinations, ncols=80, dynamic_ncols=True):
    page_num = starting_page

    combined_char = join_jamos(consonant + vowel)

    url = f"https://ko.dict.naver.com/#/topic/search?category1={category}"
    driver.get(url)
    time.sleep(1)

    wait = WebDriverWait(driver, 1)

    for _ in range(retry_attempts):
        try:
            consonant_radio = wait.until(EC.presence_of_element_located(
                (By.XPATH, f"//label[contains(text(), '{consonant}')]//input[@type='radio']")))
            driver.execute_script("arguments[0].click();", consonant_radio)

            vowel_radio = wait.until(EC.presence_of_element_located(
                (By.XPATH, f"//label[contains(text(), '{vowel}')]//input[@type='radio']")))
            driver.execute_script("arguments[0].click();", vowel_radio)
            break
        except Exception as e:
            print(f"Error while clicking radio buttons (attempt {_ + 1}): {type(e).__name__} - {str(e)}")
            # Inside the 'for consonant, vowel in tqdm(char_combinations, ncols=80, dynamic_ncols=True):' loop
            # After the 'for _ in range(retry_attempts):' loop

    else:
        tqdm.write(
            f"Failed to click radio buttons for consonant-vowel pair ({consonant}, {vowel}) after {retry_attempts} attempts. Skipping this combination.")
        continue

    time.sleep(0.5)

    seen_expressions = set()

    while True:
        url = f"https://ko.dict.naver.com/#/topic/search?category1={category}&consonant={consonant}&vowel={vowel}&page={page_num}"
        driver.get(url)
        time.sleep(0.1)
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        expression_elements = soup.select('.origin a.link')

        if not expression_elements:
            break

        scraped_data = []

        # Define the regex variables here, before the loop
        chinese_char_re = re.compile(r'[\u4e00-\u9fff]+')
        pronunciation_re = re.compile(r'\[ .+ \]')

        tqdm.write(f"{combined_char} {page_num}")

        for element in expression_elements:
            expression = element.text.strip()
            if expression in seen_expressions:
                continue
            seen_expressions.add(expression)
            expression_url = 'https://ko.dict.naver.com' + element['href']

            driver.get(expression_url)
            time.sleep(1)  # necessary

            expression_soup = BeautifulSoup(driver.page_source, 'html.parser')

            same_words_elements = expression_soup.select(
                'em.tit.relateType_same.relatedType_same ~ div.cont a.item')

            same_words = []

            for elem in same_words_elements:
                # for sup in elem.find_all('sup'):
                #     sup.extract()
                # tqdm.write(elem.text.strip())
                same_words.append(elem.text.strip())

            # similar_words_elements = expression_soup.select(
            #     'em.tit.relateType_similar.relatedType_similar ~ div.cont a.item')
            #
            # similar_words = []
            #
            # for elem in similar_words_elements:
            #     for sup in elem.find_all('sup'):
            #         sup.extract()
            #     # tqdm.write(elem.text.strip())
            #     similar_words.append(elem.text.strip())

            # Update the JSON file with the new part (similar words)
            json_file_path = os.path.join(output_dir, f"{combined_char}-{page_num}.json")
            update_json_file(json_file_path, expression, same_words)

            # synonym_elements = expression_soup.select('div.synonym strong.blind ~ em a.word')
            # synonyms = [elem.text.strip() for elem in synonym_elements]
            # similar_words.extend(synonyms)

            # Remove unnecessary elements
            for workbook_btn in expression_soup.select("button#btnAddWordBook"):
                workbook_btn.decompose()
            for special_domain in expression_soup.select('span[class^="mean_addition specialDomain_"]'):
                special_domain.decompose()
            for related_words in expression_soup.select("dt.tit:-soup-contains('Related words')"):
                related_words.decompose()

            for old_pron_area in expression_soup.select('dl[class*="entry_pronounce my_old_pron_area"]'):
                old_pron_area.decompose()

            for play_area in expression_soup.select("div.play_area"):
                play_area.decompose()

            for example_translate in expression_soup.select('div[class^="example _word_mean_ex example_translate"]'):
                example_translate.decompose()

            mean_section = expression_soup.select_one(
                'div.section.section_mean.is-source._section_mean._data_index_1')

            if mean_section is None:
                print(f"\nMeaning section not found for expression: {expression}")
                continue

            component_entries_data = []
            component_entries = expression_soup.select('div.component_entry')
            if component_entries:
                for component_entry in component_entries:
                    component_entry_text = component_entry.text.strip()
                    component_entry_text = re.sub('\n{2,}', '\n', component_entry_text)
                    component_entry_lines = component_entry_text.split('\n')[1:]  # Skip the first line (overlap)

                    chinese_chars = chinese_char_re.search(component_entry_lines[0])
                    pronunciation = None
                    for line in component_entry_lines[1:]:
                        pronunciation_match = pronunciation_re.search(line)
                        if pronunciation_match:
                            pronunciation = pronunciation_match
                            break

                    # tqdm.write(f"component_entry_lines: {component_entry_lines}")  # Replace print() with tqdm.write()
                    component_entries_data.append('\n'.join(component_entry_lines))

            meanings_data = []
            meanings = mean_section.select('.mean_list .mean_item:not([class^="mean_addition specialDomain_"])')
            if not meanings:
                print(f"No meanings found for expression: {expression}")
                continue

            for meaning in meanings:
                meaning_text = meaning.text.strip()
                meaning_text = re.sub('\n{2,}', '\n', meaning_text)
                meaning_text = re.sub('\n.+example _word_mean_ex example_translate.+', '', meaning_text)
                meaning_text = re.sub('(\d\.)\n', '\\1',
                                      meaning_text)  # Remove line breaks after definition numbers
                meanings_data.append(meaning_text)

            driver.back()
            # time.sleep(1)

            inflected_form_re = re.compile(r'Inflected form\n(.+)')
            derivative_re = re.compile(r'Derivative\n(.+)')

            inflected_forms = inflected_form_re.findall('\n'.join(component_entries_data))
            derivatives = derivative_re.findall('\n'.join(component_entries_data))

            all_pronunciations = pronunciation_re.findall('\n'.join(component_entries_data))

            expression_data = {
                'expression': expression,
                'chinese_chars': chinese_chars.group(0) if chinese_chars else None,
                'pronunciations': [pronunciation.group(0).strip('[] ') if pronunciation else None],
                'inflected_forms': inflected_forms if inflected_forms else [],
                'derivatives': derivatives if derivatives else [],
                'meanings': meanings_data,
                # 'similar_words': similar_words,
            }

            scraped_data.append(expression_data)

        # if scraped_data:
        #     output_file = os.path.join(output_dir, f"{combined_char}-{page_num}.json")
        #     with open(output_file, 'w', encoding='utf-8') as f:
        #         json.dump(scraped_data, f, ensure_ascii=False, indent=4)

        page_num += 1
    starting_page = 1

driver.quit()
