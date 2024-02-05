# Sentiment Analysis on Restaurant Reviews

## Overview
This project aims to perform sentiment analysis on restaurant reviews using Support Vector Machines (SVM) combined with Principal Component Analysis (PCA) for feature reduction. The project utilizes Natural Language Processing (NLP) techniques to preprocess the text data, extract features, and classify reviews into positive or negative sentiments.

## Dataset
The dataset consists of 1000 restaurant reviews collected from a variety of sources. Each review is labeled with a sentiment value: 1 for positive and 0 for negative. The dataset is structured as a tab-separated file (`Restaurant_Reviews.tsv`) with two columns: `Review` and `Liked`.

## Requirements
- Python 3.x
- NumPy
- pandas
- Matplotlib
- NLTK
- scikit-learn

## Installation
To run this project, you need to install the required Python packages. You can install them using pip:

    pip install numpy pandas matplotlib nltk scikit-learn
2.NLTK further requires downloading stop words and other resources. You can do this by running the following Python commands:

python

    import nltk
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
## Usage

To execute the sentiment analysis:

Ensure the Restaurant_Reviews.tsv file is in the same directory as your script.
Run the script via a terminal or an IDE that supports Python.
The script will preprocess the text data, perform feature extraction using CountVectorizer, reduce dimensionality with PCA, fit the SVM model, and finally evaluate the model's performance on test data.

## Project Structure

Data Preprocessing: Cleansing and preparing the text data for analysis.
Feature Extraction: Transforming text data into numerical vectors using the Bag of Words model.
Dimensionality Reduction: Applying PCA to reduce the number of features while retaining essential information.
Model Training and Evaluation: Fitting an SVM model to the processed data and evaluating its performance on unseen test data.
Results

The model achieved an accuracy score of 94% on the test dataset and 94.63% on the training dataset, indicating a high level of effectiveness in classifying the sentiment of restaurant reviews.

## Acknowledgments

This project utilizes data and techniques from various sources, including public datasets and the scikit-learn, pandas, and NLTK Python libraries. Special thanks to the developers and contributors of these tools for making them available to the community.
