import json

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


# Function to scrape an idiom page
def scrape_idiom_page(url):
    result = requests.get(url)
    soup = BeautifulSoup(result.text, 'html.parser')
    sections = soup.find_all('section')

    idiom_data = []
    for section in sections:
        idiom_section = {}
        idiom_section['title'] = section.find('h2', {'class': 'title paddding'}).text.strip()
        idiom_section['meaning'] = section.find('div',
                                                {'class': 'content-box contents_area meaning_area p10'}).text.strip()
        idiom_section['extra'] = {}

        for dt, dd in zip(section.find_all('dt'), section.find_all('dd')):
            idiom_section['extra'][dt.text.strip()] = dd.text.strip()

        idiom_data.append(idiom_section)

    return idiom_data


# Iterate through hiragana characters and pages
hiragana_list = ['あ', 'い', 'う', 'え', 'お',
                 'か', 'き', 'く', 'け', 'こ',
                 'さ', 'し', 'す', 'せ', 'そ',
                 'た', 'ち', 'つ', 'て', 'と',
                 'な', 'に', 'ぬ', 'ね', 'の',
                 'は', 'ひ', 'ふ', 'へ', 'ほ',
                 'ま', 'み', 'む', 'め', 'も',
                 'や', 'ゆ', 'よ',
                 'ら', 'り', 'る', 'れ', 'ろ',
                 'わ', 'を', 'ん']

MAX_PAGES = 5  # Maximum number of pages to check before giving up

hiragana_tqdm = tqdm(hiragana_list, desc="Hiragana characters", position=0)

for hiragana in hiragana_tqdm:
    page = 1
    while True:
        list_page_url = f'https://dictionary.goo.ne.jp/idiom/index/{hiragana}/{page}/'
        list_page_result = requests.get(list_page_url)
        list_page_soup = BeautifulSoup(list_page_result.text, 'html.parser')

        idiom_links = list_page_soup.find_all('a', href=True)
        if not idiom_links or page > MAX_PAGES:
            break

        idiom_links_tqdm = tqdm(idiom_links, desc=f"Processing idioms for {hiragana}, page {page}", position=1,
                                leave=False)
        for link in idiom_links_tqdm:
            if '/word/' in link['href']:
                idiom_page_url = 'https://dictionary.goo.ne.jp' + link['href']
                idiom_data = scrape_idiom_page(idiom_page_url)
                idiom_name = link['href'].split('/')[-2]
                with open(f'{idiom_name}.json', 'w', encoding='utf-8') as f:
                    json.dump(idiom_data, f, ensure_ascii=False, indent=4)
            idiom_links_tqdm.update()
        idiom_links_tqdm.close()

        page += 1
    hiragana_tqdm.update()
hiragana_tqdm.close()
