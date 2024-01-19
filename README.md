# Amazon Movie Review Sentiment Analysis
This project is primarily a personal learning exercise, maintained on GitHub for documentation.

## Overview
This project aims to identify the most critically-acclaimed movies by training a model that deduces the sentiment of Amazon reviews. Specifically, the model will determine whether the review author believes the movie is worth watching. This project is particularly designed to assist in automating the movie selection process, allowing for more efficient and informed choices.

## Dataset
The dataset provided for this project is a comprehensive collection of reviews and ratings from Amazon Prime Video’s vast catalog of movies. It consists of thousands of reviews from various users, offering a rich resource for training our models.

## Objective
The primary goal is to train various Support Vector Machines (SVMs) to classify the sentiment of a movie review accurately. This automated sentiment analysis will help in quickly identifying movies that are highly regarded by viewers, thereby enhancing the movie selection process.

## Tools and Techniques

- Data Preprocessing: Cleaning and preparing the review data for analysis.
- Word Embedding: Utilizing Word2Vec to convert text data into numerical form that can be fed into machine learning models.
- Model Training: Using SVMs to classify review sentiments.
- Model Evaluation: Assessing the performance of our models.

### Development Environment
1. Python (https://www.python.org/downloads/), with a Python 3.11 virtual environment.
2. scikit-learn (1.3.0): documentation available at https://scikit-learn.org/stable/
3. numpy (1.25.2): documentation available at https://numpy.org/doc/stable/
4. pandas (2.1.0): documentation available at https://pandas.pydata.org/docs/
5. matplotlib (3.7.2): documentation available at https://matplotlib.org/stable/
6. gensim (4.3.2): documentation available at https://radimrehurek.com/gensim/auto examples/

To install the correct versions of the required packages, run the command ```pip install -r requirements.txt``` in your virtual environment.

### File Structure
- `data/`: Directory containing the dataset files.
  - `dataset.csv`: Amazon movie reviews with binary labels.
  - `heldout.csv`: Amazon movie reviews with multiclass labels, using for prediction.
  - `debug.csv`: A samll dataset for debug.
- `helper.py`: Script for data accessing and prediction generating.
- `project.py`: Skeleton code, including data cleaning, model training, model evaluation, and output.
- `challenge.py`: Script for more parameter selecting and model attempt.
- `test_output.py`: Script for testing output format.
- `debug_output.txt`: Output of debug dataset, use for check the correctness of some functions before appling on whole dataset.
- `requirements.txt`: List of all the dependencies with their versions.
- `README.md`: Project overview.
