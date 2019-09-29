import sys

from nltk.tokenize import TweetTokenizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import make_pipeline

sys.path.append("..")
from utils import get_data  # noqa: E402


def main():
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
                gamma='scale'
            ),
            
        )
    )


if __name__ == '__main__':
    main()
