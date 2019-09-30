"""
Un intento con subsampling, parece que es la luz
"""
import sys
import logging
from logging import info
from collections import Counter
import numpy as np

from nltk.tokenize import TweetTokenizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, cohen_kappa_score, accuracy_score
from sklearn.model_selection import (cross_val_score,
                                     cross_validate,
                                     StratifiedKFold,
                                     KFold,
                                     RepeatedStratifiedKFold)

from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.pipeline import make_pipeline

sys.path.append("..")
from utils import get_data  # noqa:E402


logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s [%(asctime)s] %(message)s',
    datefmt='%H:%M:%S'
)


def main():
    info("Leyendo datos")
    train, _ = get_data("../data")
    dataset = train["anger"]

    # X, y = CountVectorizer().fit_transform(dataset.tweet), dataset.sentiment_intensity

    # print(Counter(y))
    # X_res, y_res = RandomUnderSampler("majority").fit_resample(X, y)
    # print(Counter(y_res))

    clf = make_pipeline(
        CountVectorizer(
            # tokenizer=add_negation(
            #     TweetTokenizer(
            #         strip_handles=True,
            #         # reduce_len=True,
            #     ).tokenize
            # )
            tokenizer=TweetTokenizer(
                preserve_case=False,
                strip_handles=True,
                reduce_len=False,
            ).tokenize
        ),
        RandomUnderSampler(
            sampling_strategy="majority",
        ),
        SVC(
            C=100.,
            kernel='rbf',
            gamma='scale',
            # probability=True,
        ),
        # verbose=False
    )

    info("Calculando cross-validation")
    scores = cross_validate(
        clf,
        dataset.tweet,
        dataset.sentiment_intensity,
        cv=RepeatedStratifiedKFold(
            n_splits=5,
            n_repeats=10,
            # shuffle=True,
        ),
        scoring=make_scorer(cohen_kappa_score),
        verbose=0,
        n_jobs=-1,
        return_estimator=True,
    )
    # print(list(sorted(scores.keys())))
    # print(len(scores["estimator"]))
    info(f"Scores obtenidos: {scores['test_score']}")
    info(
        f"Score promedio con subsampling: "
        f"{scores['test_score'].mean():0.2f} "
        f"(+/- {scores['test_score'].std()*2:0.2f})"
    )

    best = scores["estimator"][np.argmax(scores["test_score"])]

    high = dataset[dataset.sentiment_intensity == "medium"]
    print(accuracy_score(best.predict(high.tweet), high.sentiment_intensity))


if __name__ == '__main__':
    main()
