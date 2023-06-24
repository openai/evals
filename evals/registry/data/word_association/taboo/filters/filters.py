import nltk


def parts_of_speech_filter(words: list[str], parts_of_speech: list[str]) -> list[str]:
    return [word for word in words if word[1] in parts_of_speech]


def frequency_filter(words: list[str], frequency_distribution: nltk.FreqDist, threshold: int) -> list[str]:
    return [
        word
        for word in words
        if frequency_distribution[word] >= threshold
    ]


def subword_filter(words: list[str], subword: str) -> list[str]:
    return [word for word in words if subword not in word]
