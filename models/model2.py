"""
Aca voy a probar distintas weas con SVM
Hasta ahora este ha sido el mejorcito
"""

import sys
import logging
from logging import info

from nltk.tokenize import TweetTokenizer
from nltk.sentiment.util import mark_negation

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import NuSVC, SVC
from sklearn.model_selection import (cross_val_score,
                                     StratifiedKFold,
                                     KFold,
                                     RepeatedStratifiedKFold)
from sklearn.metrics import make_scorer, cohen_kappa_score

from representation1 import my_preprocessor, without_unwanted

sys.path.append("..")
from utils import get_data  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s [%(asctime)s] %(message)s',
    datefmt='%H:%M:%S'
)


def add_negation(tokenizer):
    def aux(tweet):
        return mark_negation(tokenizer(tweet))
    return aux


def main():
    # Leer datos
    info("Leyendo datos")
    train, target = get_data("../data")
    anger_train, _ = train["anger"], target["anger"]

    clf = make_pipeline(
        CountVectorizer(
            # preprocessor=my_preprocessor,
            # tokenizer=add_negation(
            #     TweetTokenizer(
            #         strip_handles=True,
            #         # reduce_len=True,
            #     ).tokenize
            # )
            # tokenizer=without_unwanted(TweetTokenizer(
            #     preserve_case=True,
            #     strip_handles=True,
            #     reduce_len=False,
            # ).tokenize),
            tokenizer=TweetTokenizer(
                preserve_case=True,
                strip_handles=True,
                reduce_len=False,
            ).tokenize,
        ),
        SVC(
            C=100.,
            kernel='rbf',
            gamma='scale',
            # probability=True,
        ),
        verbose=False
    )
    info("Calculando cross-validation")
    scores = cross_val_score(
        clf,
        anger_train.tweet,
        anger_train.sentiment_intensity,
        cv=RepeatedStratifiedKFold(
            n_splits=5,
            n_repeats=10,
            # shuffle=True,
        ),
        scoring=make_scorer(cohen_kappa_score),
        verbose=0,
        n_jobs=-1
    )
    info(f"Scores obtenidos: {scores}")
    info(f"Mean score: {scores.mean():0.2f} (+/- {scores.std()*2:0.2f})")


if __name__ == '__main__':
    main()
