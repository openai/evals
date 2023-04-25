import os
import random
import time

from bs4 import BeautifulSoup
from hangul_utils import join_jamos
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm

driver_path = 'C:/webdrivers/chromedriver.exe'
options = Options()
options.add_argument('--headless')
options.add_argument('--log-level=3')

driver = webdriver.Chrome(executable_path=driver_path, options=options)

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

consonants = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
vowels = ['ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']

while True:
    order = input("Enter the order you want to scrape (alphabetical, random): ").strip().lower()
    if order in ['alphabetical', 'random']:
        break
    else:
        print("Invalid order. Please enter 'alphabetical' or 'random'.")

char_combinations = [(c, v) for c in consonants for v in vowels]
if order == 'random':
    random.shuffle(char_combinations)

# Wrap the main loop with tqdm to show the progress bar
for consonant, vowel in tqdm(char_combinations, ncols=80, dynamic_ncols=True):
    combined_char = join_jamos(consonant + vowel)

    # Load the main search page
    url = f"https://ko.dict.naver.com/#/topic/search?category1={category}"
    driver.get(url)

    wait = WebDriverWait(driver, 3)
    retry_attempts = 3

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
    else:
        print(f"Failed to click radio buttons after {retry_attempts} attempts. Skipping this combination.")
        continue

    # Give the page time to update the search results
    time.sleep(3)

    # Continue with the rest of your code
    page_num = 1
    while True:
        # tqdm.write(f"Combined character: {combined_char} - Scraping Page {page_num}")
        url = f"https://ko.dict.naver.com/#/topic/search?category1={category}&consonant={consonant}&vowel={vowel}&page={page_num}"
        driver.get(url)
        time.sleep(3)
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        expression_elements = soup.select('.origin a.link')

        if not expression_elements:
            break

        expressions_found = False
        output_data = ""

        for element in expression_elements:
            expression = element.text.strip()

            expression_url = 'https://ko.dict.naver.com' + element['href']

            driver.get(expression_url)
            time.sleep(3)

            expression_soup = BeautifulSoup(driver.page_source, 'html.parser')

            meanings = expression_soup.select('.mean_list .mean_item')

            output_data += expression + '\n'
            for meaning in meanings:
                output_data += '\t' + meaning.text.strip() + '\n'

            driver.back()
            time.sleep(3)

            expressions_found = True

        if expressions_found:
            output_file = os.path.join(output_dir, f"{combined_char}_page_{page_num}.txt")

            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(output_data)
        else:
            break

        page_num += 1

driver.quit()
