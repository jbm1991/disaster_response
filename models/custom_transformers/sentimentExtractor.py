import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


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
