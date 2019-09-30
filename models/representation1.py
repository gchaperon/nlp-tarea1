"""
Aca voy a realizar un filtro terrible a mano de los datos
"""

import sys
import logging
import re
import html
from logging import info

from nltk.tokenize import TweetTokenizer
from nltk.sentiment.util import mark_negation

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

sys.path.append("..")
from utils import get_data  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s [%(asctime)s] %(message)s',
    datefmt='%H:%M:%S'
)

UNWANTED = ["#", "$", "%", "&", "'", "(", ")", "*", ",", "-", ".", "..", "/",
            "<", ">", "@", "{", "}", "~", "£", "óg", "–", "—", "’", "“", "”",
            "•", "…", "™", "←", "→"]


def my_filter(token):
    return not (
        token in UNWANTED or
        token.startswith("@") or
        token[0].isdigit()  # revisar este
    )


def my_preprocessor(tweet):
    """
    Aca quiza vaya mas logica, es un comienzo
    """
    return html.unescape(tweet).replace(r"\n", " ")


def without_unwanted(base_tokenizer):
    def clean(tweet):
        dirty = base_tokenizer(tweet)
        return [token for token in dirty if my_filter(token)]
    return clean


def main():
    train, _ = get_data("../data")
    sentiment = "anger"
    dataset = train[sentiment]

    vec = CountVectorizer(
        preprocessor=my_preprocessor,
        tokenizer=without_unwanted(TweetTokenizer(
            preserve_case=False,
            reduce_len=True,
            strip_handles=True,
        ).tokenize),
    )

    vec.fit(dataset.tweet)
    print(len(vec.get_feature_names()))

    with open(f"features_{sentiment}.txt", "w") as f:
        for feature in sorted(vec.get_feature_names()):
            f.write(feature + "\n")


if __name__ == '__main__':
    main()
