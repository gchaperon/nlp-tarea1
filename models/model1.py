"""
Modelo con naive-bayes y usando algun otro vectorizador
Lo voy a usar para probar cross validation y los clasificadores ensemble
"""
import sys
import logging
from logging import info
from functools import partial

from nltk.tokenize import TweetTokenizer
from nltk.sentiment.util import mark_negation

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import (cross_val_score,
                                     StratifiedKFold,
                                     KFold,
                                     RepeatedStratifiedKFold)
from sklearn.metrics import make_scorer, cohen_kappa_score


sys.path.append("..")
from utils import get_data  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s [%(asctime)s] %(message)s',
    datefmt='%H:%M:%S'
)

# WIP (?)
UNWANTED = ['.', ',', ';', ':', '\n', '%']


tokenizer = TweetTokenizer(
                # preserve_case=False,
                strip_handles=True,
                reduce_len=True,
            )


def my_tokenizer(tweet, tokenizer=tokenizer):
    """
    - Parece que algo mejora sacando las palabras muy cortas
    """
    return [word for word in tokenizer.tokenize(tweet) if len(word) > 1]


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
            # tokenizer=add_negation(
            #     TweetTokenizer(
            #         strip_handles=True,
            #         # reduce_len=True,
            #     ).tokenize
            # )
            # tokenizer=TweetTokenizer(
            #     # preserve_case=False,
            #     strip_handles=True,
            #     reduce_len=True,
            # ).tokenize
            tokenizer=my_tokenizer
        ),
        MultinomialNB(
            alpha=0.15,
            fit_prior=False,
        ),
        verbose=False
    )
    info("Calculando cross-validation")
    scores = cross_val_score(
        clf,
        anger_train.tweet,
        anger_train.sentiment_intensity,
        cv=RepeatedStratifiedKFold(
            n_splits=10,
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
