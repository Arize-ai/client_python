from typing import Dict, List, Optional

import pandas as pd
from tqdm.auto import tqdm

from arize.utils.logging import logger

IMPORT_ERROR_MESSAGE = (
    "To enable NLP metrics calculations, the arize module must be installed with "
    "extra dependencies. Run: pip install 'arize[NLP_Metrics]'."
)

try:
    import evaluate
except ImportError:
    raise ImportError(IMPORT_ERROR_MESSAGE) from None


# BLEU
def bleu(
    response_col: pd.Series,
    references_col: pd.Series,
    max_order: int = 4,
    smooth: bool = False,
) -> List[float]:
    """
    BLEU (Bilingual Evaluation Understudy) is an algorithm used to evaluate the quality of machine
    translated text from one natural language to another. The quality of machine translation is measured
    based on its similarity to human translation, with the goal being to achieve high correspondence between
    the two.

    BLEU calculates scores for individual translated segments, typically sentences, by comparing
    them to a set of high-quality reference translations. These scores are then averaged over the entire
    corpus to obtain an estimate of the overall quality of the translation. Notably, BLEU does not consider
    factors such as intelligibility or grammatical correctness when computing the scores.

    NOTE: This metric is a wrapper around the Hugging Face's implementation of BLEU:
        https://github.com/huggingface/evaluate/blob/main/metrics/bleu/bleu.py

    Arguments:
    ---------
        response_col (pd.Series): Pandas Series containing the translations (as strings) to score.
        references_col (pd.Series): Pandas Series containing a reference, or list of several references,
            for each translation.
        max_order (int, optional):  Maximum n-gram order to use when computing BLEU score. Defaults to 4.
        smooth (bool, optional): Whether or not to apply Lin et al. 2004 smoothing. Defaults to False.

    Returns:
    -------
        List[float]: BLEU scores

    """
    logger.info("Loading metric: bleu")
    bleu = evaluate.load("bleu")
    logger.info("Computing bleu scores")
    return [
        bleu.compute(
            predictions=[preds],
            references=[refs],
            max_order=max_order,
            smooth=smooth,
        )[  # type:ignore
            "bleu"
        ]
        for (preds, refs) in tqdm(
            zip(response_col, references_col),
            total=len(response_col),
        )
    ]


# SACRE_BLEU
def sacre_bleu(
    response_col: pd.Series,
    references_col: pd.Series,
    smooth_method: str = "exp",
    smooth_value: Optional[float] = None,
    lowercase: bool = False,
    force: bool = False,
    use_effective_order: bool = False,
) -> List[float]:
    """
    SacreBLEU provides hassle-free computation of shareable, comparable, and reproducible BLEU scores.
    Inspired by Rico Sennrich's multi-bleu-detok.perl, it produces the official Workshop on Machine
    Translation (WMT) scores but works with plain text.

    NOTE: This metric is a wrapper around the Hugging Face's implementation of SacreBLEU:
        https://github.com/huggingface/evaluate/blob/main/metrics/sacrebleu/sacrebleu.py

    Arguments:
    ---------
        response_col (pd.Series): Pandas Series containing translations (as strings) to score.
        references_col (pd.Series): Pandas Series containing a reference, or list of several references,
            per prediction. Note that there must be the same number of references for each prediction
            (i.e. all sub-lists must be of the same length).
        smooth_method (str, optional): The smoothing method to use, defaults to 'exp'. Possible values are:
            - 'none': no smoothing
            - 'floor': increment zero counts
            - 'add-k': increment num/denom by k for n>1
            - 'exp': exponential decay
        smooth_value (Optional[float], optional): The smoothing value. Only valid when smooth_method='floor'
            (in which case smooth_value defaults to 0.1) or smooth_method='add-k' (in which case
            smooth_value defaults to 1). Defaults to None.
        lowercase (bool, optional): If True, lowercases the input, enabling case-insensitivity. Defaults to
            False.
        force (bool, optional): If True, insists that your tokenized input is actually de-tokenized. Defaults
            to False.
        use_effective_order (bool, optional): If True, stops including n-gram orders for which precision is 0.
            This should be True, if sentence-level BLEU will be computed. Defaults to False.

    Returns:
    -------
        List[float]: SacreBLEU scores

    """
    logger.info("Loading metric: sacrebleu")
    s_bleu = evaluate.load("sacrebleu")
    logger.info("Computing sacrebleu scores")
    return [
        s_bleu.compute(
            predictions=[preds],
            references=[refs],
            smooth_method=smooth_method,
            smooth_value=smooth_value,
            lowercase=lowercase,
            force=force,
            use_effective_order=use_effective_order,
        )[  # type:ignore
            "score"
        ]
        for (preds, refs) in tqdm(
            zip(response_col, references_col),
            total=len(response_col),
        )
    ]


# GOOGLE_BLEU
def google_bleu(
    response_col: pd.Series,
    references_col: pd.Series,
    min_len: int = 1,
    max_len: int = 4,
) -> List[float]:
    """
    The BLEU score was designed to be used as a corpus measure, and it has some limitations when applied to
    single sentences. To overcome this issue in RL experiments, there exists a variation called the GLEU
    score. For GLEU, we analyze all the possible sub-sequences of 1, 2, 3, or 4 tokens in both the generated
    output and target sequences. We then calculate the recall and precision, where recall is the number of
    matching n-grams divided by the total number of n-grams in the target sequence, and precision is the
    number of matching n-grams divided by the total number of n-grams in the generated output sequence.
    The GLEU score is the minimum of recall and precision.

    The GLEU score ranges between 0 (no matches) and 1 (all matches), and it is symmetrical when we switch
    the output and target sequences. The GLEU score is highly correlated with the BLEU metric at the corpus
    level but does not have the limitations of the BLEU score for our per sentence reward objective.

    NOTE: This metric is a wrapper around the Hugging Face's implementation of Google BLEU:
        https://github.com/huggingface/evaluate/blob/main/metrics/google_bleu/google_bleu.py

    Arguments:
    ---------
        response_col (pd.Series): Pandas Series containing translations (as strings) to score.
        references_col (pd.Series): Pandas Series containing a reference, or list of several references,
            for each translation.
        min_len (int, optional): The minimum order of n-gram this function should extract. Defaults to 1.
        max_len (int, optional): The maximum order of n-gram this function should extract. Defaults to 4.

    Returns:
    -------
        List[float]: google-BLEU scores

    """
    logger.info("Loading metric: google_bleu")
    g_bleu = evaluate.load("google_bleu")
    logger.info("Computing google_bleu scores")
    return [
        g_bleu.compute(predictions=[preds], references=[refs])["google_bleu"]  # type:ignore
        for (preds, refs) in tqdm(
            zip(response_col, references_col), total=len(response_col)
        )
    ]


# ROUGE
def rouge(
    response_col: pd.Series,
    references_col: pd.Series,
    rouge_types: Optional[List[str]] = None,
    use_stemmer: bool = False,
) -> Dict[str, List[float]]:
    """
    ROUGE, which stands for Recall-Oriented Understudy for Gisting Evaluation, is a software package and a
    set of metrics commonly used to evaluate machine translation and automatic summarization software in
    natural language processing. These metrics involve comparing a machine-produced summary or translation
    with a reference or set of references that have been human-produced. It's worth noting that ROUGE treats
    uppercase and lowercase letters as equivalent, making the evaluation case-insensitive. By providing a
    standardized way to assess the quality of machine-generated summaries and translations, ROUGE is a useful
    tool in the field of natural language processing.

    NOTE: This metric is a wrapper around the Hugging Face's implementation of ROUGE:
        https://github.com/huggingface/evaluate/blob/main/metrics/rouge/rouge.py

    Arguments:
    ---------
        response_col (pd.Series): Pandas Series containing predictions (as strings) to score.
        references_col (pd.Series): Pandas Series containing a reference, or list of several references,
            per prediction.
        rouge_types (List[str], optional): A list of rouge types to calculate. If None is passed, it will
        default to ['rougeL']. Valid rouge types:
                - "rouge1": unigram (1-gram) based scoring
                - "rouge2": bigram (2-gram) based scoring
                - "rougeL": Longest common subsequence based scoring.
                - "rougeLSum": splits text using '\n'
        use_stemmer (bool, optional): If True, uses Porter stemmer to strip word suffixes. Defaults to False.

    Returns:
    -------
        Dict[str, List[float]]: The output is a dictionary with one entry for each rouge type in the input
            list rouge_types.

    """
    if rouge_types is None:
        rouge_types = ["rougeL"]
    data: Dict[str, List[float]] = {
        k: [0] * len(response_col) for k in rouge_types
    }
    logger.info("Loading metric: rouge")
    rouge = evaluate.load("rouge")
    logger.info("Computing rouge scores")
    for idx, (preds, refs) in tqdm(
        enumerate(zip(response_col, references_col)),
        total=len(response_col),
    ):
        r: Dict[str, List[float]] = rouge.compute(
            predictions=[preds],
            references=[refs],
            rouge_types=rouge_types,
            use_stemmer=use_stemmer,
            use_aggregator=False,
        )  # type:ignore
        for r_type in rouge_types:
            data[r_type][idx] = r[r_type][0]
    return data


# METEOR
def meteor(
    response_col: pd.Series,
    references_col: pd.Series,
    alpha: float = 0.9,
    beta: float = 3,
    gamma: float = 0.5,
) -> List[float]:
    """
    METEOR is an automatic metric used to evaluate machine translation, which is based on a generalized
    concept of unigram matching between the machine-produced translation and the reference human-produced
    translations. METEOR has the capability to match unigrams based on their surface forms, stemmed forms,
    and meanings. Moreover, advanced matching strategies can be easily integrated into METEOR. After finding
    all generalized unigram matches between the two strings, METEOR calculates a score for this matching using
    a combination of unigram-precision, unigram-recall, and a measure of fragmentation, which is specifically
    designed to assess the order of the matched words in the machine translation compared to the reference.

    NOTE: This metric is a wrapper around the Hugging Face's implementation of METEOR:
        https://github.com/huggingface/evaluate/blob/main/metrics/meteor/meteor.py

    Arguments:
    ---------
        response_col (pd.Series): Pandas Series containing predictions (as strings) to score.
        references_col (pd.Series): Pandas Series containing a reference, or list of several references,
            per prediction.
        alpha (float, optional): Parameter for controlling relative weights of precision and recall.
            Default is 0.9.
        beta (float, optional): Parameter for controlling shape of penalty as a function of fragmentation.
            Default is 3.
        gamma (float, optional): The relative weight assigned to fragmentation penalty. Default is 0.5.

    Returns:
    -------
        List[float]: METEOR scores

    """
    logger.info("Loading metric: meteor")
    meteor = evaluate.load("meteor")
    logger.info("Computing meteor scores")
    return [
        meteor.compute(
            predictions=[preds],
            references=[refs],
            alpha=alpha,
            beta=beta,
            gamma=gamma,
        )[  # type:ignore
            "meteor"
        ]
        for (preds, refs) in tqdm(
            zip(response_col, references_col), total=len(response_col)
        )
    ]
