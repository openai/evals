from typing import Mapping

import jiwer
import spacy_universal_sentence_encoder

nlp = spacy_universal_sentence_encoder.load_model("en_use_lg")


def reconstruction_metrics(original: str, decompressed: str) -> Mapping:
    return {
        # Accuracies
        "exact_match": float(original == decompressed),
        "semantic_similarity": 1 - semantic_distance(original, decompressed),
        # Errors
        "semantic_distance": semantic_distance(original, decompressed),
        "character_error_rate_ignore_case_cap1": min(
            jiwer.cer(original.lower(), decompressed.lower()), 1.0
        ),
        "character_error_rate_ignore_case": jiwer.cer(original.lower(), decompressed.lower()),
        "character_error_rate_cap1": min(jiwer.cer(original, decompressed), 1.0),
        "character_error_rate": jiwer.cer(original, decompressed),
        "word_error_rate": jiwer.wer(original, decompressed),
        "match_error_rate": jiwer.mer(original, decompressed),
        "word_information_lost": jiwer.wil(original, decompressed),
    }


def semantic_distance(original: str, decompressed: str) -> float:
    doc1 = nlp(original)
    doc2 = nlp(decompressed)

    if decompressed.strip() == "" and original.strip() != "":
        # Recovered payload is empty
        return 1

    similarity = doc1.similarity(doc2)
    similarity = max(
        similarity, 0
    )  # Discard negative values, not clearly meaningful and rarely occurs

    dist = 1 - similarity
    return dist
