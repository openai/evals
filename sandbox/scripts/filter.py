import os
import re
import time

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from tqdm import tqdm

from hangul_utils import join_jamos

driver_path = 'C:/webdrivers/chromedriver.exe'
options = Options()
options.add_argument('--headless')
options.add_argument('--log-level=3')

service = Service(executable_path=driver_path)
driver = webdriver.Chrome(service=service, options=options)

consonants = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
vowels = ['ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ']

results = []


def write_results_to_file(results, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(f"{result['consonant']}, {result['vowel']}, {result['amount']}\n")


total_combinations = len(consonants) * len(vowels)
with tqdm(total=total_combinations) as pbar, tqdm(total=0, position=0, bar_format='{desc}') as msg_bar:
    for consonant in consonants:
        for vowel in vowels:
            combined_char = join_jamos(consonant + vowel)
            url = f"https://ko.dict.naver.com/#/topic/search?category1=phrase&consonant={consonant}&vowel={vowel}&page=1"
            driver.get(url)
            time.sleep(1)
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            amount_element = soup.select_one('.result .amount')
            if amount_element:
                amount_text = amount_element.text
                match = re.search(r'\d+(?=건)', amount_text)
                if match:
                    amount = int(match.group(0))
                else:
                    amount = -1
                result = {'consonant': consonant, 'vowel': vowel, 'amount': amount}
                results.append(result)
                msg_bar.desc = f"Processing: {len(results)} / {total_combinations} ({len(results) / total_combinations * 100:.2f}%)"
                tqdm.write(f"{consonant}, {vowel}, {amount}")
            pbar.update(1)

driver.quit()

write_results_to_file(results, 'results.txt')

msg_bar.write(f"Done! Total results processed: {len(results)}")

os.startfile('results.txt')
