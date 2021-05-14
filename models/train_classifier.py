import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import re
import pandas as pd
from sqlalchemy import create_engine
import sys
import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download(['stopwords'])


def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('Messages', con=engine)

    X = df['message']
    Y = df.drop(columns=['id', 'message', 'genre', 'original'])

    return X, Y, list(Y.columns)


def tokenize(text):
    # TODO part of speech tagging
    # TODO named entity recognition
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = stopwords.words("english")
    clean_tokens = [lemmatizer.lemmatize(token).lower(
    ).strip() for token in tokens if token not in stop_words]

    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1), n_jobs=-1))
    ])

    parameters = {
        # 'tfidf__use_idf': [True, False],
        # 'vect__max_df': (0.5, 0.75, 1.0),
        # 'vect__max_features': (None, 5000, 10000),
        # 'vect__ngram_range': ((1, 1), (1, 2)),
        # 'clf__estimator__random_state': [42],
        # 'clf__estimator__n_estimators': [10, 100, 500, 1000, 1500, 2000],
        # 'clf__estimator__max_depth': [None, 10, 50, 100, 500],
        # 'clf__estimator__min_samples_leaf': [1, 3, 5],
        # 'clf__estimator__max_features': ['auto', 'log2', None],
        # 'clf__estimator__bootstrap': [True, False],
        # 'clf__estimator__min_samples_split': [2, 4],
    }

    cv = GridSearchCV(pipeline, param_grid=parameters, verbose=3)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    Y_pred_df = pd.DataFrame(Y_pred, columns=category_names)
    for column in category_names:
        print(f'Classification Report for: {column}')
        print(classification_report(Y_test[column], Y_pred_df[column]))
        print(accuracy_score(Y_test[column], Y_pred_df[column]))
    print('Best parameters for model:')
    print(model.best_params_)


def save_model(model, model_filepath):
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
