# import os
# import random
# import shutil
import numpy as np
import pandas as pd
from functools import partial

from sklearn.metrics import (confusion_matrix,
                             cohen_kappa_score,
                             classification_report,
                             accuracy_score,
                             roc_auc_score)
from sklearn.model_selection import train_test_split


def auc(test_set, predicted_set):
    high_predicted = np.array([prediction[2] for prediction in predicted_set])
    medium_predicted = np.array(
        [prediction[1] for prediction in predicted_set])
    low_predicted = np.array([prediction[0] for prediction in predicted_set])

    high_test = np.where(test_set == 'high', 1.0, 0.0)
    medium_test = np.where(test_set == 'medium', 1.0, 0.0)
    low_test = np.where(test_set == 'low', 1.0, 0.0)

    auc_high = roc_auc_score(high_test, high_predicted)
    auc_med = roc_auc_score(medium_test, medium_predicted)
    auc_low = roc_auc_score(low_test, low_predicted)

    auc_w = (low_test.sum() * auc_low + medium_test.sum() * auc_med +
             high_test.sum() * auc_high) / (
                 low_test.sum() + medium_test.sum() + high_test.sum())
    return auc_w


def get_data(base_path):
    col_names = ['id', 'tweet', 'class', 'sentiment_intensity']
    sentiments = ['anger', 'fear', 'joy', 'sadness']
    split_names = ['train', 'target']
    return [
        {
            sentiment: pd.read_csv(
                f"{base_path}/{split_name}/{sentiment}-{split_name}.txt",
                sep='\t',
                names=col_names
            )
            for sentiment in sentiments
        }
        for split_name in split_names
    ]


def get_group_dist(group_name, train):
    print(group_name, "\n",
          train[group_name].groupby('sentiment_intensity').count())


def split_dataset(dataset, seed=8080):
    # Dividir el dataset en train set y test set
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.tweet,
        dataset.sentiment_intensity,
        shuffle=True,
        test_size=0.33,
        random_state=seed,
    )
    return X_train, X_test, y_train, y_test


def evaulate(predicted, y_test, labels, key):
    # Importante: al transformar los arreglos de probabilidad a clases,
    # entregar el arreglo de clases aprendido por el clasificador.
    # (que comunmente, es distinto a ['low', 'medium', 'high'])
    predicted_labels = [labels[np.argmax(item)] for item in predicted]

    # Confusion Matrix
    print('Confusion Matrix for {}:\n'.format(key))

    # Classification Report
    print(
        confusion_matrix(y_test,
                         predicted_labels,
                         labels=['low', 'medium', 'high']))

    print('\nClassification Report')
    print(
        classification_report(y_test,
                              predicted_labels,
                              labels=['low', 'medium', 'high']))

    # AUC
    print("auc: ", auc(y_test, predicted))

    # Kappa
    print("kappa:", cohen_kappa_score(y_test, predicted_labels))

    # Accuracy
    print("accuracy:", accuracy_score(y_test, predicted_labels), "\n")

    print('------------------------------------------------------\n\n')


def _classify(dataset, key, pipeline):

    X_train, X_test, y_train, y_test = split_dataset(dataset)
    text_clf = pipeline

    # Entrenar el clasificador
    text_clf.fit(X_train, y_train)

    # Predecir las probabilidades de intensidad de cada elemento del set de prueba.
    predicted = text_clf.predict_proba(X_test)

    # Obtener las clases aprendidas.
    learned_labels = text_clf.classes_

    # Evaluar
    evaulate(predicted, y_test, learned_labels, key)
    return text_clf, learned_labels


def classify_hof(pipeline):
    return partial(_classify, pipeline=pipeline)


def do_the_magic(pipeline):
    """
    Por ahora esta funcion solo printea resultados, no guarda na
    """
    train, _ = get_data("../data")
    classifiers = []
    learned_labels_array = []

    classify = classify_hof(pipeline)
    # Por cada llave en train ('anger', 'fear', 'joy', 'sadness')
    for key in train:
        classifier, learned_labels = classify(train[key], key)
        classifiers.append(classifier)
        learned_labels_array.append(learned_labels)

    # TODO TERMINAR ESTA WEA Y GUARDAR DATOS Y ETC