import pickle
import re
import sys

import nltk
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sqlalchemy import create_engine
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download(['punkt', 'wordnet'])
nltk.download(['stopwords'])


class LengthsCalculator(BaseEstimator, TransformerMixin):
    """
    A custom transformer which calculates the character count, word
    count, sentence count, average word length and average sentence length.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_counts = pd.Series(X, name="char_count").apply(
            lambda x: sum(len(word) for word in str(x).split(" ")))
        X_out = pd.DataFrame(X_counts)
        X_out['word_count'] = pd.Series(X).apply(
            lambda x: len(str(x).split(" ")))
        X_out['sent_count'] = pd.Series(X).apply(
            lambda x: len(str(x).split(".")))
        X_out['word_length'] = X_out['char_count'] / X_out['word_count']
        X_out['sent_length'] = X_out['word_count'] / X_out['sent_count']
        return X_out


class SentimentExtractor(BaseEstimator, TransformerMixin):
    """
    A custom transformer which returns the sentiment of the messages using
    Vader Sentiment. Sentiment is binned into positive, negative and neutral.
    """

    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def get_sentiment(self, message):
        vs = self.analyzer.polarity_scores(message)
        if vs['compound'] >= 0.05:
            return 2  # positive
        elif vs['compound'] <= -0.05:
            return 0  # negative
        else:
            return 1  # neutral

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_sentiment = pd.Series(X, name='sentiment').apply(self.get_sentiment)
        return pd.DataFrame(X_sentiment)


def load_data(database_filepath):
    """
    Load the data from the database

    Args:
        database_filepath (str): filepath to the database file

    Returns:
        pandas.DataFrame: The messages in a dataframe
        pandas.DataFrme: The output categories in a dataframe
        list: The output column labels
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('Messages', con=engine)

    X = df['message']
    Y = df.drop(columns=['id', 'message', 'genre', 'original'])

    return X, Y, list(Y.columns)


def tokenize(text):
    """
    Tokenizer function for use with CountVectorizer. Removes all special
    characters, converts to lower case, tokenizes, lemmatizes and removes
    stopwords.

    Args:
        text (str): Individual message

    Returns:
        list: The cleaned and processed tokens for the message
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")
    clean_tokens = [lemmatizer.lemmatize(token).lower(
    ).strip() for token in tokens if token not in stop_words]

    return clean_tokens


def build_model():
    """
    Constructs the model which will be used for classification, using a Pipeline.

    Returns:
        GridSearchCV: The model
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer()),
            ])),
            ('lengths', LengthsCalculator()),
            ('sentiment', SentimentExtractor())
        ])),
        ('clf', MultiOutputClassifier(RandomForestClassifier(
            n_jobs=-1, random_state=42), n_jobs=-1))
    ])

    # commented out these options for performance. They don't really seem to make a big
    # difference in terms of the accuracy score of the output
    parameters = {
        # 'tfidf__use_idf': [True, False],
        # 'vect__max_df': (0.5, 0.75, 1.0),
        # 'vect__max_features': (None, 5000, 10000),
        # 'vect__ngram_range': ((1, 1), (1, 2)),
        # 'clf__estimator__n_estimators': [10, 100, 500, 1000, 1500, 2000],
        # 'clf__estimator__max_depth': [None, 50, 100, 500],
        'clf__estimator__max_depth': [500],  # found to be best parameter
        # 'clf__estimator__min_samples_leaf': [1, 3, 5],
        # 'clf__estimator__max_features': ['auto', 'log2', None],
        # 'clf__estimator__min_samples_split': [2, 4],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the given trained model against the test data set. Outputs the 
    classification report for each target classification and also the best found
    parameters from the model.

    Args:
        model (GridSearchCV): Trained model
        X_test (DataFrame): Test dataset inputs
        Y_test (DataFrame): Test dataset outputs
        category_names (list): output column names
    """
    Y_pred = model.predict(X_test)
    Y_pred_df = pd.DataFrame(Y_pred, columns=category_names)
    for column in category_names:
        print(f'Classification Report for: {column}')
        print(classification_report(Y_test[column], Y_pred_df[column]))
        print(accuracy_score(Y_test[column], Y_pred_df[column]))
    print('Best parameters for model:')
    print(model.best_params_)


def save_model(model, model_filepath):
    """
    Save the trained model into a pickle file for future use.

    Args:
        model (GridSearchCV): Trained model
        model_filepath (str): Filepath to save to
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
