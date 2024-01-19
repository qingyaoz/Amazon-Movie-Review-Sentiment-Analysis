import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import project


def load_data(fname):
    """
    Reads in a csv file and return a dataframe.
    """
    return pd.read_csv(fname)


def get_split_binary_data(fname="data/dataset.csv", n=None):
    """
    Reads in the data from fname and returns it using
    extract_dictionary and generate_feature_matrix split into training and test sets.
    The binary labels take two values:
        -1: poor/average
         1: good
    Also returns the dictionary used to create the feature matrices.
    Input:
        fname: name of the file to be read from.
    """
    dataframe = load_data(fname)
    dataframe = dataframe[dataframe["label"] != 0]
    positiveDF = dataframe[dataframe["label"] == 1].copy()
    negativeDF = dataframe[dataframe["label"] == -1].copy()
    if n != None:
        class_size = n
    else:
        class_size = 2 * positiveDF.shape[0] // 3
    X_train = (
        pd.concat([positiveDF[:class_size], negativeDF[:class_size]])
        .reset_index(drop=True)
        .copy()
    )
    dictionary = project.extract_dictionary(X_train)
    X_test = (
        pd.concat([positiveDF[class_size:], negativeDF[class_size:]])
        .reset_index(drop=True)
        .copy()
    )
    Y_train = X_train["label"].values.copy()
    Y_test = X_test["label"].values.copy()
    X_train = project.generate_feature_matrix(X_train, dictionary)
    X_test = project.generate_feature_matrix(X_test, dictionary)

    return (X_train, Y_train, X_test, Y_test, dictionary)


def get_imbalanced_data(dictionary, fname="data/dataset.csv", ratio=0.25):
    """
    Reads in the data from fname and returns imbalanced dataset using
    extract_dictionary and generate_feature_matrix split into training and test sets.
    The binary labels take two values:
        -1: poor/average
         1: good
    Input:
        dictionary: dictionary to create feature matrix from
        fname: name of the file to be read from.
        ratio: ratio of positive to negative samples
    """
    dataframe = load_data(fname)
    dataframe = dataframe[dataframe["label"] != 0]
    positiveDF = dataframe[dataframe["label"] == 1].copy()
    negativeDF = dataframe[dataframe["label"] == -1].copy()
    negativeDF = negativeDF[: int(ratio * positiveDF.shape[0])]
    positive_class_size = 2 * positiveDF.shape[0] // 3
    negative_class_size = 2 * negativeDF.shape[0] // 3
    positiveDF = positiveDF.sample(frac=1, random_state=445)
    negativeDF = negativeDF.sample(frac=1, random_state=445)
    X_train = (
        pd.concat([positiveDF[:positive_class_size], negativeDF[:negative_class_size]])
        .reset_index(drop=True)
        .copy()
    )
    X_test = (
        pd.concat([positiveDF[positive_class_size:], negativeDF[negative_class_size:]])
        .reset_index(drop=True)
        .copy()
    )
    Y_train = X_train["label"].values.copy()
    Y_test = X_test["label"].values.copy()
    X_train = project.generate_feature_matrix(X_train, dictionary)
    X_test = project.generate_feature_matrix(X_test, dictionary)

    return (X_train, Y_train, X_test, Y_test)


def get_multiclass_training_data(class_size=750):
    """
    Reads in the data from data/dataset.csv and returns it using
    extract_dictionary and generate_feature_matrix as a tuple
    (X_train, Y_train) where the labels are multiclass as follows
        -1: poor
         0: average
         1: good
    Also returns the dictionary used to create X_train.
    Input:
        class_size: Size of each class (pos/neg/neu) of training dataset.
    """
    fname = "data/dataset.csv"
    dataframe = load_data(fname)
    neutralDF = dataframe[dataframe["label"] == 0].copy()
    positiveDF = dataframe[dataframe["label"] == 1].copy()
    negativeDF = dataframe[dataframe["label"] == -1].copy()
    X_train = (
        pd.concat(
            [positiveDF[:class_size], negativeDF[:class_size], neutralDF[:class_size]]
        )
        .reset_index(drop=True)
        .copy()
    )
    dictionary = project.extract_dictionary(X_train)
    Y_train = X_train["label"].values.copy()
    X_train = project.generate_feature_matrix(X_train, dictionary)

    return (X_train, Y_train, dictionary)


def get_heldout_reviews(dictionary):
    """
    Reads in the data from data/heldout.csv and returns it as a feature
    matrix based on the functions extract_dictionary and generate_feature_matrix
    Input:
        dictionary: the dictionary created by get_multiclass_training_data
    """
    fname = "data/heldout.csv"
    dataframe = load_data(fname)
    X = project.generate_feature_matrix(dataframe, dictionary)
    return X


def generate_challenge_labels(y, uniqname):
    """
    Takes in a numpy array that stores the prediction of your multiclass
    classifier and output the prediction to held_out_result.csv. Please make sure that
    you do not change the order of the ratings in the heldout dataset since we will use
    this file to evaluate your classifier.
    """
    pd.Series(np.array(y)).to_csv(uniqname + ".csv", header=["label"], index=False)
    return


def filter_actors_and_actresses(fname):
    """
    The input fname is the path to the csv file containing the dataframe. Example: "data/dataset.csv"
    df_actor should contain all rows of the original dataframe where the review text contains the words 'actor' and/or 'actors' (not case sensitive).
    df_actress should contain all rows of the original dataframe where the review text contains the words 'actress' and/or 'actresses' (not case sensitive).
    Reviews mentioning both actor(s) and actress(es) should be in both dataframes.
    """
    df = load_data(fname)
    df_actor = df.loc[df["reviewText"].str.contains(r"\bactors?\b", case=False)]
    df_actress = df.loc[
        df["reviewText"].str.contains(r"\bactress(?:es)?\b", case=False)
    ]
    return df_actor, df_actress


def count_actors_and_actresses(fname):
    """
    The input fname is the path to the csv file containing the dataframe. Example: "data/dataset.csv"
    Returns the number of reviews in df_actor and df_actress from the filter_actors_and_actresses() function
    """
    df_actor, df_actress = filter_actors_and_actresses(fname)
    return df_actor["reviewText"].count(), df_actress["reviewText"].count()


def plot_actors_and_actresses(fname, x_label):
    """
    Inputs:
        - fname: The path to the csv file containing the dataframe. Example: "data/dataset.csv"
        - x_label: The name of the dataframe column we are plotting. Either 'label' or 'rating'
    Save a figure showing the distribution of labels or ratings across reviews mentioning actors and actresses.
    """
    df_actor, df_actress = filter_actors_and_actresses(fname)

    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    fig.supxlabel(x_label)
    fig.supylabel("proportion")

    ax1.set_title("Actor")
    ax2.set_title("Actress")

    num_bins = 3 if x_label == "label" else 5
    weights1 = np.ones_like(df_actor[x_label]) / float(df_actor[x_label].count())
    _, _, bars1 = ax1.hist(df_actor[x_label], bins=num_bins, weights=weights1)

    weights2 = np.ones_like(df_actress[x_label]) / float(df_actress[x_label].count())
    _, _, bars2 = ax2.hist(df_actress[x_label], bins=num_bins, weights=weights2)

    ax1.locator_params(axis="x", nbins=num_bins)
    ax2.locator_params(axis="x", nbins=num_bins)

    ax1.bar_label(bars1, fmt="%.2f")
    ax2.bar_label(bars2, fmt="%.2f")

    plt.savefig(f"plot_actor_{x_label}")
