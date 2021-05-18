import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


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
