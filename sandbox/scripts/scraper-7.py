import json
from urllib.parse import unquote

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


# Function to scrape an idiom page
def scrape_idiom_page(idiom, url):
    result = requests.get(url)
    soup = BeautifulSoup(result.text, 'html.parser')
    sections = soup.find_all('section')

    idiom_data = []
    for section in sections:
        idiom_section = {}
        title_element = section.find('h2', {'class': 'title paddding'})
        meaning_element = section.find('div', {'class': 'content-box contents_area meaning_area p10'})

        if title_element and meaning_element:
            title = title_element.text.replace("の解説 -", "").strip()
            title = " ".join(title.split())  # Remove excessive whitespaces
            idiom_section['source'] = title.replace(f"{idiom} ", "")  # Remove idiom from the title
            idiom_section['meaning'] = meaning_element.text.strip()
            idiom_section['extra'] = {}

            for dt, dd in zip(section.find_all('dt'), section.find_all('dd')):
                idiom_section['extra'][dt.text.strip()] = dd.text.strip()

            idiom_data.append(idiom_section)

    return {'idiom': idiom, 'definitions': idiom_data}


def find_last_page_number(soup):
    nav_paging = soup.find('div', {'class': 'nav-paging cx bt_e5'})
    if nav_paging:
        page_numbers = nav_paging.find_all('a')
        last_page_number = max([int(a.text) for a in page_numbers if a.text.isdigit()])
        return last_page_number
    return 1


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

hiragana_tqdm = tqdm(hiragana_list, desc="Hiragana characters", position=0)

for hiragana in hiragana_tqdm:
    list_page_url = f'https://dictionary.goo.ne.jp/idiom/index/{hiragana}/1/'
    list_page_result = requests.get(list_page_url)
    list_page_soup = BeautifulSoup(list_page_result.text, 'html.parser')

    last_page_number = find_last_page_number(list_page_soup)

    for page in range(1, last_page_number + 1):  # Use range instead of while loop
        idiom_links = list_page_soup.find_all('a', href=True)
        if not idiom_links:
            break

        idiom_links_tqdm = tqdm(idiom_links, desc=f"Processing idioms for {hiragana}, page {page}", position=1,
                                leave=False)
        for link in idiom_links_tqdm:
            if '/word/' in link['href']:
                idiom_page_url = 'https://dictionary.goo.ne.jp' + link['href']
                idiom_name = link['href'].split('/')[-2]
                idiom_data = scrape_idiom_page(unquote(idiom_name), idiom_page_url)
                with open(f'./output/{unquote(idiom_name)}.json', 'w', encoding='utf-8') as f:
                    json.dump(idiom_data, f, ensure_ascii=False, indent=4)
            idiom_links_tqdm.update()
        idiom_links_tqdm.close()

        page += 1
    hiragana_tqdm.update()
hiragana_tqdm.close()
