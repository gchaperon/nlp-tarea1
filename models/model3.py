import sys
import logging
from logging import info

from nltk.tokenize import TweetTokenizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.ensemble import BaggingClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (cross_val_score, RepeatedStratifiedKFold)
from sklearn.metrics import make_scorer, cohen_kappa_score

sys.path.append("..")
from utils import get_data  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s [%(asctime)s] %(message)s',
    datefmt='%H:%M:%S'
)


def main():
    info("Leyendo datos")
    train, _ = get_data("../data")
    data = train["anger"]

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
        BaggingClassifier(
            base_estimator=SVC(
                C=100.,
                kernel='rbf',
                gamma='scale',
                probability=True,
            ),
            n_estimators=10,
            max_samples=1.,
            bootstrap=True,
            n_jobs=-1,
            verbose=0,
        )
        # SVC(
        #     C=100.,
        #     kernel='rbf',
        #     gamma='scale',
        #     probability=True,
        # )
    )
    info("Calculando cross-validation para el bagging classifier")
    scores = cross_val_score(
        clf,
        data.tweet,
        data.sentiment_intensity,
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
