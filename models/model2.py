"""
Aca voy a probar distintas weas con SVM
Hasta ahora este ha sido el mejorcito
"""

import os
import sys
import copy 
import shutil
import logging
import numpy as np
from logging import info

from nltk.tokenize import TweetTokenizer
from nltk.sentiment.util import mark_negation

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import NuSVC, SVC
from sklearn.model_selection import (cross_val_score,
                                     cross_validate,
                                     StratifiedKFold,
                                     KFold,
                                     RepeatedStratifiedKFold)
from sklearn.metrics import make_scorer, cohen_kappa_score

from representation1 import my_preprocessor, without_unwanted

sys.path.append("..")
from utils import get_data, predict_target  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s [%(asctime)s] %(message)s',
    datefmt='%H:%M:%S'
)


def add_negation(tokenizer):
    def aux(tweet):
        return mark_negation(tokenizer(tweet))
    return aux


def fair_sampling(train, rs):
    new_train = copy.deepcopy(train)
    intensities = ['high', 'medium', 'low']
    for key in new_train:
        cants = np.array([new_train[key]['id'][new_train[key].sentiment_intensity == tens].count()
                          for tens in intensities])
        max_who = intensities[list(cants).index(cants.max())]
        new_max = int((cants.mean() - cants.max()/3) * 3/2)
        # Aqui estan los reducidos reducidos
        max_sample = new_train[key][new_train[key].sentiment_intensity == max_who].sample(new_max, random_state = rs)
        the_rest = new_train[key][new_train[key].sentiment_intensity != max_who].copy()
        all_data = the_rest.append(max_sample, ignore_index=True)
        new_train[key] = all_data
    return new_train


def main():
    # Leer datos
    info("Leyendo datos")
    train, target = get_data("../data")
    new_train = fair_sampling(train, 8080)

    clf = make_pipeline(
        CountVectorizer(
            preprocessor=my_preprocessor,
            # tokenizer=add_negation(
            #     TweetTokenizer(
            #         strip_handles=True,
            #         # reduce_len=True,
            #     ).tokenize
            # )
            tokenizer=without_unwanted(
                TweetTokenizer(
                    preserve_case=True,
                    reduce_len=False,
                    strip_handles=False,
                ).tokenize
            ),
            # tokenizer=TweetTokenizer(
            #     preserve_case=True,
            #     strip_handles=True,
            #     reduce_len=False,
            # ).tokenize,
            stop_words="english",
            min_df=5,
        ),
        SVC(
            C=100.,
            kernel='linear',
            gamma='scale',
            probability=True,
        ),
        verbose=False
    )
    # crear directorio
    if (not os.path.isdir('../predictions')):
        os.mkdir('../predictions')

    else:
        # Eliminar predicciones anteriores:
        shutil.rmtree('../predictions')
        os.mkdir('../predictions')

    sentiments = ["anger"]  # , "fear", "joy", "sadness"]
    for sentiment in sentiments:
        dataset_train, dataset_target = train[sentiment], target[sentiment]
        info(f"Calculando cross-validation para {sentiment}")
        scores = cross_validate(
            clf,
            dataset_train.tweet,
            dataset_train.sentiment_intensity,
            cv=RepeatedStratifiedKFold(
                n_splits=5,
                n_repeats=20,
                # shuffle=True,
            ),
            scoring=make_scorer(cohen_kappa_score),
            verbose=0,
            n_jobs=-1,
            return_estimator=True,
        )
        info(f"Scores obtenidos: {scores['test_score']}")
        info(f"Mean score: {scores['test_score'].mean():0.2f} (+/- {scores['test_score'].std()*2:0.2f})")
        best = scores["estimator"][np.argmax(scores["test_score"])]
        # info(best.classes_)

        predicted_target = predict_target(dataset_target, best, best.classes_)
        predicted_target.to_csv(
            f'../predictions/{sentiment}-pred.txt',
            sep='\t',
            header=False,
            index=False
        )
    # Crear archivo zip
    shutil.make_archive('../predictions', 'zip', '../predictions')


if __name__ == '__main__':
    main()
