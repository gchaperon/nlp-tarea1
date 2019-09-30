import sys
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize.casual import TweetTokenizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
# Esta wea es pa poder importar de utils
sys.path.append("..")

# from utils import auc  # noqa: E402
from utils import do_the_magic  # noqa: E402



def main():
    vectorizer = CountVectorizer()

    # Inicializamos el Clasificador.
    naive_bayes = MultinomialNB()

    # Establecer el pipeline.
    pipeline = Pipeline([('vect', vectorizer), ('clf', naive_bayes)])

    do_the_magic(pipeline)


if __name__ == '__main__':
    main()
