import concurrent.futures
import json
import os
import random
import time

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from tqdm import tqdm

verbose = False

# Load the first set of JSON files
folder_1 = "E:/data/stdict/JSON_20230208"
json_files_1 = [os.path.join(folder_1, file) for file in os.listdir(folder_1) if file.endswith(".json")]

# Read progress file, if available
progress_file = "progress.json"
if os.path.exists(progress_file):
    with open(progress_file, "r", encoding="utf-8") as f:
        progress = json.load(f)
else:
    progress = {}

# Continue from the last processed file
start_index = progress.get("last_processed_file", 0)


# Function to save progress
def save_progress():
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


def search_chengyu_on_website(driver, chengyu, max_retries=3):
    for attempt in range(max_retries):
        try:
            # Navigate to the search URL
            search_url = f"https://dict.idioms.moe.edu.tw/idiomList.jsp?idiom={chengyu}&qMd=0&qTp=1&qTp=2"
            driver.get(search_url)

            # Wait for the results to load and find the elements with the <mark> tag
            wait = WebDriverWait(driver, 1)
            elements = wait.until(EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'mark')))

            # Extract the text from the <mark> elements
            result_list = [element.text for element in elements]

            return result_list

        except Exception as e:
            if verbose:
                tqdm.write(f"Error searching for {chengyu}: {e}")
            if attempt < max_retries - 1:
                wait_time = random.uniform(1, 5)  # Random wait time between 1 and 5 seconds
                time.sleep(wait_time)
        # No need to quit the driver here, as it will be reused


output_folder = "E:/data/temp"
os.makedirs(output_folder, exist_ok=True)


def process_entry(entry):
    if "original_language_info" in entry["word_info"] and "original_language" in \
            entry["word_info"]["original_language_info"][0]:
        return entry["word_info"]["original_language_info"][0]["original_language"], entry
    return None, None


def process_search_results(driver, chengyu, entry):
    if chengyu and entry:
        search_result = search_chengyu_on_website(driver, chengyu)
        if search_result:
            output_file = os.path.join(output_folder, f"{entry['word_info']['word']}.json")
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(entry, f, indent=2, ensure_ascii=False)


def process_file(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = [process_entry(item) for item in data["channel"]["item"]]

    for chengyu, entry in results:
        process_search_results(driver, chengyu, entry)

    return len(data["channel"]["item"])


chrome_options = Options()
chrome_options.add_argument("--headless")
chrome_options.add_argument("--log-level=3")
chrome_options.add_argument("--disable-dev-shm-usage")
chrome_options.add_argument("--remote-debugging-port=0")
service = Service(executable_path="C:/webdrivers/chromedriver.exe")
driver = webdriver.Chrome(service=service, options=chrome_options)

pbar1 = tqdm(json_files_1[start_index:], desc="Processing files", position=0)

with concurrent.futures.ProcessPoolExecutor() as executor:
    for i, processed_items in enumerate(executor.map(process_file, json_files_1[start_index:])):
        # Update progress
        progress["last_processed_file"] = start_index + i + 1
        save_progress()

        # Update the first progress bar
        pbar1.update(1)
        pbar1.set_postfix({"Processed JSON items": processed_items})
        pbar1.refresh()

driver.quit()  # Quit the driver after processing all files
