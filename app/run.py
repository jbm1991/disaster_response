import json
import re

import joblib
import nltk
import pandas as pd
import plotly
from flask import Flask, jsonify, render_template, request
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from plotly.graph_objs import Bar
from sklearn.base import BaseEstimator, TransformerMixin
from sqlalchemy import create_engine
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

nltk.download(['punkt', 'wordnet'])
nltk.download(['stopwords'])

app = Flask(__name__)


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


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Messages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    category_counts = df.drop(
        columns=['id', 'message', 'original', 'genre']).sum()
    category_names = list(category_counts.index)

    related_means = df.groupby('genre').mean()['related'] * 100

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(x=category_names, y=category_counts)
            ],
            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(x=genre_names, y=related_means)
            ],
            'layout': {
                'title': 'Proportion of Related Messages by Genre',
                'yaxis': {
                    'title': "Mean"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
