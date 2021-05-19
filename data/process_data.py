import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load the data from the CSV files and merge into 1 DataFrame.

    Args:
        messages_filepath (str): Messages CSV file
        categories_filepath (str): Categories CSV file

    Returns:
        DataFrame: Combined data from the CSVs
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages.merge(categories, on='id')


def clean_data(df):
    """
    Clean the data by separating into separate columns where appropriate
    and removing duplicates.

    Args:
        df (DataFrame): The data in a DataFrame

    Returns:
        DataFrame: Cleaned data in a DataFrame
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x.split('-')[0])
    categories.columns = category_colnames
    for column in categories:
        # remove the column name from the value, so we are left with 0 or 1
        categories[column] = categories[column].str.split('-').str[1]
        categories[column] = pd.to_numeric(categories[column])
    df = df.drop(columns=['categories'])
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates()
    df = df.drop(index=df[df['related'] == 2].index)
    return df


def save_data(df, database_filename):
    """Save the supplied DataFrame into a db file for querying

    Args:
        df (DataFrame): Data to be saved
        database_filename (str): Path to save the file at
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('Messages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
