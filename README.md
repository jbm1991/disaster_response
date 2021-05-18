# Disaster Response Pipeline Project

## Installation

Please use Python 3.\*. You can install the necessary dependencies with the command `pip install -r requirements.txt`.

## Instructions

1. Run the following commands in the project's root directory to set up your database and model.

   - To run ETL pipeline that cleans data and stores in database
     `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
   - To run ML pipeline that trains classifier and saves
     `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
   `python run.py`

3. Go to [http://0.0.0.0:3001/](http://0.0.0.0:3001/)

## Project Motiviation

This project is an assignment in the Udacity Data Science Nanodegree. The aim is use an ETL pipeline and an ML pipeline to train a model which can categorise messages received by disaster response teams. Additionally there is a web application which shows visualisations of the dataset, and uses the trained model to predict the categorisations of new messages inputted into the website.

## File Descriptions

- `app/*` contains the files for the Flask web application
- `data/*` contains both the dataset files and the ETL pipeline file
- `models/*` contains ML pipeline file

## Licensing, Authors and Acknowledgements

The dataset was provided by [Figure Eight](https://www.figure-eight.com/). Some of the code was provided by [Udacity](https://www.udacity.com), and the rest was written by me.
