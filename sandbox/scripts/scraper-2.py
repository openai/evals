import os
import random
import time

from bs4 import BeautifulSoup
from selenium import webdriver
from tqdm import tqdm

# Ask user for scraping preferences
category = input("Enter the category you want to scrape (proverb, phrase, idiom): ")
output_dir = input("Enter the output directory: ")

order_input = input("Enter the order for scraping (alphabetical or random): ")
if order_input.lower() == "random":
    random_order = True
else:
    random_order = False

# Replace the path with the location where you've saved the ChromeDriver
driver_path = 'C:/webdrivers/chromedriver.exe'
options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('--log-level=3')  # Suppress console messages

driver = webdriver.Chrome(executable_path=driver_path, options=options)

# Create a directory to save the output files
os.makedirs(output_dir, exist_ok=True)

consonants = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
vowels = ['ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']

char_combinations = [(c, v) for c in consonants for v in vowels]
if random_order:
    random.shuffle(char_combinations)

for consonant, vowel in tqdm(char_combinations, desc="Scraping"):
    page_num = 1
    while True:
        url = f"https://ko.dict.naver.com/#/search?range=topic&query={consonant}{vowel}&category1={category}&page={page_num}"
        driver.get(url)
        time.sleep(5)  # Wait for 5 seconds before parsing the page
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Find the HTML elements containing the idiomatic expressions
        expression_elements = soup.select('.origin a.link')

        if not expression_elements:
            print(f"No expressions found for {consonant}{vowel} - Page {page_num}")
            break

        output_file = os.path.join(output_dir, f"{consonant}_{vowel}_page_{page_num}.txt")
        print(f"Output file path: {output_file}")
        print(f"Writing to {output_file}")

        with open(output_file, 'w', encoding='utf-8') as f:
            print(f"Creating file: {output_file}")

            for element in expression_elements:
                expression = element.text.strip()
                expression_url = 'https://ko.dict.naver.com' + element['href']

                # Visit the expression page
                driver.get(expression_url)
                time.sleep(3)

                # Parse the expression page
                expression_soup = BeautifulSoup(driver.page_source, 'html.parser')
                print(soup.prettify())

                # Extract the meanings
                meanings = expression_soup.select('.mean_list .mean_item .mean')

                f.write(f"{expression}:\n")
                for idx, meaning in enumerate(meanings, 1):
                    f.write(f"{idx}. {meaning.text.strip()}\n")
                f.write("\n")
                print(f"Finished writing to file: {output_file}")

        page_num += 1

driver.quit()
