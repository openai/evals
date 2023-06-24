import requests


def get_related_words(word: str, api_url: str ="https://api.datamuse.com/words") -> list[str]:
    response = requests.get(api_url, params={"rel_syn": word, "max": 5})
    return [result["word"] for result in response.json()]
