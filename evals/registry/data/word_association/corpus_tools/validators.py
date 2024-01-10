import os
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Dict, List, NamedTuple, Tuple, Union

import numpy as np
from logger_config import logger
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

CORRELATION_PROMPT_TEMPLATE = """Task: Estimate the degree of correlation between
 two provided strings. In your evaluation, consider not just direct links, but also indirect and subtle correlations.
 As an illustration, if 'watch' appears in the first string and 'tower' in the second,
 you could consider the combined term 'watchtower'. Similarly, for 'watch' and 'warning',
 think in terms of phrases like 'watch out for warnings'.
 You should score the correlation using the following scale from 0.00 to 1.00, where:
 0.00 signifies no correlation whatsoever.
 0.50 indicates a moderate level of correlation. This means there are several significant
 connections between the terms in the two strings, but these are not overwhelming.
 1.00 is reserved for ONLY two strings that are completely identical.

Strings to Correlate:
string_one: {word}
string_two: {related_words}"""

ANSWER_PROMPT_TEMPLATE = """\nYour final output should be in the following format(NOTE: Include the square brackets):
\nReasoning: <your reasoning>\nFinal Answer: [<float, rounded to the 100th place>]"""


class Embedding(NamedTuple):
    """A named tuple representing a string and its corresponding embedding."""

    string: str
    vector: List[float]


class RelatedWordsPair(NamedTuple):
    """A named tuple containing a word and its related words."""

    word: str
    related_words: str


class EmbeddingPair(NamedTuple):
    """A named tuple representing a pair of related words and their embeddings."""

    related_words_pair: RelatedWordsPair
    vectors: Tuple[Embedding]


class SimilarityTuple(NamedTuple):
    """A named tuple representing the result of a similarity analysis."""

    related_words_pair: RelatedWordsPair
    similar: bool
    similarity_score: float


class QualityValidator(ABC):
    """Abstract base class for implementing quality validators."""

    def __init__(self, target_score: int) -> None:
        self.target_score = target_score

    @abstractmethod
    def validate(self, related_words_pair: List[RelatedWordsPair]) -> List[SimilarityTuple]:
        raise NotImplementedError


class EmbeddingsValidator(QualityValidator):
    """
    An implementation of QualityValidator that validates the similarity of embeddings for pairs of related words.
    """

    def validate(
        self,
        related_words_pairs: List[RelatedWordsPair],
        similarity_function: Callable[[List[float], List[float]], float] = None,
    ) -> List[SimilarityTuple]:
        """
        Validates a list of related words pairs by comparing their embeddings.

        Args:
            related_words_pairs: a list of related word pairs to validate.
            similarity_function: a function that calculates similarity between two embeddings.
                                 Defaults to cosine similarity.

        Returns:
            A list of SimilarityTuple each containing a RelatedWordsPair, a boolean indicating if they're similar,
            and the similarity score.
        """
        logger.info(f"Validating {len(related_words_pairs)} related strings.")
        if similarity_function is None:
            similarity_function = self.calculate_cosine_similarity

        # flatten all strings
        all_strings = [
            string for pair in related_words_pairs for string in (pair.word, pair.related_words)
        ]
        logger.debug(f"{all_strings} flattened.")
        # get embeddings
        embeddings = self.get_embeddings(all_strings)
        logger.info(f"{len(embeddings)} embeddings processed.")
        # form EmbeddingPairs
        embedding_pairs = [
            EmbeddingPair(related_string, (embeddings[i], embeddings[i + 1]))
            for i, related_string in enumerate(related_words_pairs)
        ]

        results = []
        # for each EmbeddingPair, compare their embeddings and form SimilarityTuple
        for pair in embedding_pairs:
            similarity_score = round(
                similarity_function(pair.vectors[0].vector, pair.vectors[1].vector), 3
            )
            similar = similarity_score > self.target_score
            similarity_tuple = SimilarityTuple(pair.related_words_pair, similar, similarity_score)
            results.append(similarity_tuple)
            logger.info(f"{pair.related_words_pair}: {similar} score:({similarity_score})")
        return results

    @staticmethod
    def calculate_cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        Calculates cosine similarity between two vectors.

        Args:
            vec1: First vector.
            vec2: Second vector.

        Returns:
            The cosine similarity between the two vectors.
        """
        vec1_norm = np.linalg.norm(vec1)
        vec2_norm = np.linalg.norm(vec2)
        similarity = np.dot(vec1, vec2) / (vec1_norm * vec2_norm)
        logger.debug(f"vec1: {vec1}, vec2: {vec2}, similarity: {similarity}")
        return similarity

    @staticmethod
    def calculate_euclidean_distance(vec1: List[float], vec2: List[float]) -> float:
        """
        Calculates Euclidean distance between two vectors.

        Args:
            vec1: First vector.
            vec2: Second vector.

        Returns:
            The Euclidean distance between the two vectors.
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        difference = vec1 - vec2
        distance = np.linalg.norm(difference)
        logger.debug(f"vec1: {vec1}, vec2: {vec2}, distance: {distance}")
        return distance

    @staticmethod
    def get_embeddings(
        emb_input: Union[RelatedWordsPair, str, List[str], List[List[int]]]
    ) -> List[Embedding]:
        """
        Batches the process of getting embeddings from the API.

        Args:
            emb_input: an input which can be a single string, a list of strings or a list of lists of tokens.

        Returns:
            A list of Embedding namedtuples where each Embedding
            represents the input string and its corresponding vector.
        """
        response = client.embeddings.create(model="text-embedding-ada-002", input=emb_input)
        logger.debug(f"embeddings response: {response}")
        response_data = response["data"]
        emb_list = [data["embedding"] for data in response_data]
        embeddings = [
            Embedding(string=string, vector=vector) for string, vector in zip(emb_input, emb_list)
        ]
        return embeddings


class GPTValidator(QualityValidator):
    """Uses the GPT model to validate the similarities between pairs of related words."""

    def __init__(
        self, target_score: int, criteria: Dict[str, str] = None, model: str = "gpt-4"
    ) -> None:
        """
        Constructor for GPTValidator.

        Args:
            target_score: The minimum score threshold for two words to be considered similar.
            criteria: A dictionary containing any specific criteria to be used in the validation process.
            model: The identifier of the GPT model to use for the validation.
        """
        self._model = model
        self.criteria = criteria
        super().__init__(target_score)

    def validate(self, related_words_pairs: List[RelatedWordsPair]) -> List[SimilarityTuple]:
        """
        Validates a list of related word pairs by comparing the outputs of the GPT model.

        Args:
            related_words_pairs: A list of pairs of related words to validate.

        Returns:
            A list of tuples containing the original word pair, a boolean indicating whether they are similar
            according to the GPT model, and the similarity score.
        """
        similarity_tuples = []
        for related_words_pair in related_words_pairs:
            response = self.get_chat_completion(related_words_pair)
            similarity_score = self.extract_score(response)
            similarity = similarity_score > self.target_score
            similarity_tuple = SimilarityTuple(related_words_pair, similarity, similarity_score)
            similarity_tuples.append(similarity_tuple)
        return similarity_tuples

    def get_chat_completion(
        self,
        related_words_pair: RelatedWordsPair,
        correlation_prompt: str = None,
        answer_prompt: str = None,
    ) -> List[SimilarityTuple]:
        """
        Uses the GPT model to generate a completion based on a given prompt.

        Args:
            related_words_pair: The pair of related words to generate a completion for.
            correlation_prompt: An optional specific prompt for the correlation task.
            answer_prompt: An optional specific prompt for the answer format. If not provided, a default is used.

        Returns:
            The content of the message from the GPT model's response.
        """
        if correlation_prompt is None:
            correlation_prompt = CORRELATION_PROMPT_TEMPLATE.format(
                word=related_words_pair.word, related_words=related_words_pair.related_words
            )
        if answer_prompt is None:
            answer_prompt = ANSWER_PROMPT_TEMPLATE
        prompt = correlation_prompt + answer_prompt

        messages = [{"role": "user", "content": prompt}]
        logger.debug(
            f"Getting chat_completion using {self._model}.\nPrompting messages: {messages}"
        )
        response = client.chat.completions.create(
            model=self._model, messages=messages, temperature=0.0
        )
        logger.debug(f"response_message: {response}")
        response_message = response["choices"][0]["message"]["content"]
        logger.info(f"response_message: {response_message}")
        return response_message

    @staticmethod
    def extract_score(response_content: str) -> float:
        """
        Extracts the similarity score from the content of a GPT model's response.

        Args:
            response_content: The content of a GPT model's response.

        Returns:
            The similarity score as a float. If no score could be extracted, returns 0.0.
        """
        try:
            match = re.search(r"Final Answer: \[(.+?)]", response_content).group(1)
            score = float(match)
            logger.debug(f"response_content: {response_content}, score: {score}")
        except AttributeError:
            score = 0.0
            logger.warning(
                "Answer not found in response, score set to 0, will autofail validation scoring."
            )
        return score

    def set_model(self, model: str) -> None:
        """
        Changes the GPT model used for validation.

        Args:
            model: The identifier of the GPT model to use for the validation.
        """
        # Add logic to reject incorrect models
        self._model = model


if __name__ == "__main__":
    # Demonstration of Both Validators
    related_words_pairs = [
        RelatedWordsPair("stage", "point, level, present"),
        RelatedWordsPair("board", "point, level, present"),
    ]

    validator = EmbeddingsValidator(0.75)
    similarity_tuples: SimilarityTuple = validator.validate(related_words_pairs)
    print(similarity_tuples)

    gpt_validator = GPTValidator(0.75, model="gpt-4")
    similarity_tuples: SimilarityTuple = gpt_validator.validate(related_words_pairs)
    print(similarity_tuples)
