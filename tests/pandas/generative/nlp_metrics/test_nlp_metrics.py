import pandas as pd
import pytest
from arize.pandas.generative.nlp_metrics import bleu, google_bleu, meteor, rouge, sacre_bleu


def get_text_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "response": [
                "The cat is on the mat.",
                "The NASA Opportunity rover is battling a massive dust storm on Mars.",
            ],
            "references": [
                ["The cat is on the blue mat."],
                [
                    "The Opportunity rover is combating a big sandstorm on Mars.",
                    "A NASA rover is fighting a massive storm on Mars.",
                ],
            ],
        }
    )


def get_results_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bleu": [0.6129752413741056, 0.32774568052975916],
            "sacrebleu": [61.29752413741059, 32.774568052975916],
            "google_bleu": [0.6538461538461539, 0.3695652173913043],
            "rouge1": [0.923076923076923, 0.7272727272727272],
            "rouge2": [0.7272727272727272, 0.39999999999999997],
            "rougeL": [0.923076923076923, 0.7272727272727272],
            "rougeLsum": [0.923076923076923, 0.7272727272727272],
            "meteor": [0.8757427021441488, 0.7682980599647267],
        }
    )


def test_bleu_score() -> None:
    texts = get_text_df()
    results = get_results_df()

    try:
        bleu_scores = bleu(response_col=texts["response"], references_col=texts["references"])
    except Exception:
        assert False, "There should be no error"

    assert (bleu_scores == results["bleu"]).all(), "BLEU scores should match"  # type:ignore


def test_sacrebleu_score() -> None:
    texts = get_text_df()
    results = get_results_df()

    try:
        sacrebleu_scores = sacre_bleu(
            response_col=texts["response"], references_col=texts["references"]
        )
    except Exception:
        assert False, "There should be no error"

    assert (
        sacrebleu_scores == results["sacrebleu"]
    ).all(), "SacreBLEU scores should match"  # type:ignore


def test_google_bleu_score() -> None:
    texts = get_text_df()
    results = get_results_df()

    try:
        gbleu_scores = google_bleu(
            response_col=texts["response"], references_col=texts["references"]
        )
    except Exception:
        assert False, "There should be no error"

    assert (
        gbleu_scores == results["google_bleu"]
    ).all(), "Google BLEU scores should match"  # type:ignore


def test_rouge_score() -> None:
    texts = get_text_df()
    results = get_results_df()

    try:
        rouge_scores = rouge(response_col=texts["response"], references_col=texts["references"])
    except Exception:
        assert False, "There should be no error"

    # Check that only default rouge scores are returned, and they match
    assert isinstance(rouge_scores, dict)
    assert len(rouge_scores.keys()) == 1
    assert list(rouge_scores.keys())[0] == "rougeL"  # type:ignore
    assert (rouge_scores["rougeL"] == results["rougeL"]).all(), "ROUGE scores should match"  # type: ignore

    rouge_types = [
        "rouge1",
        "rouge2",
        "rougeL",
        "rougeLsum",
    ]
    try:
        rouge_scores = rouge(
            response_col=texts["response"],
            references_col=texts["references"],
            rouge_types=rouge_types,
        )
    except Exception:
        assert False, "There should be no error"

    # Check that all rouge scores are returned, and they match
    assert isinstance(rouge_scores, dict)
    assert list(rouge_scores.keys()) == rouge_types
    for rtype in rouge_types:
        assert (
            rouge_scores[rtype] == results[rtype]
        ).all(), f"ROUGE scores ({rtype}) should match"  # type: ignore


def test_meteor_score() -> None:
    texts = get_text_df()
    results = get_results_df()

    try:
        meteor_scores = meteor(response_col=texts["response"], references_col=texts["references"])
    except Exception:
        assert False, "There should be no error"

    assert (meteor_scores == results["meteor"]).all(), "METEOR scores should match"  # type:ignore


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
