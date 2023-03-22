from collections import Counter
from typing import Dict, Text

from evals.plugin.base import Plugin, api_description, namespace


@namespace(
    "TextAnalyzer",
    description="Analyze a given text and provide insights like word count, character count, most common words, and average word length.",
)
class TextAnalyzer(Plugin):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    @api_description(
        name="analyzeText",
        description="API for analyzing the input text.",
        api_args={
            "text": {
                "type": "string",
                "optional": False,
                "description": "The text to be analyzed.",
            }
        },
    )
    def analyze_text(self, text: Text) -> Dict:
        words = text.split()
        word_count = len(words)
        character_count = len(text)
        avg_word_length = sum(len(word) for word in words) / word_count
        most_common_words = dict(Counter(words).most_common(5))

        return {
            "word_count": word_count,
            "character_count": character_count,
            "average_word_length": avg_word_length,
            "most_common_words": most_common_words,
        }
