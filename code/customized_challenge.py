import itertools
import string
import warnings

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from helper import *
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import TimeSeriesSplit
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC, LinearSVC
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)

np.random.seed(445)


def get_data(class_size=750):
    """
    Reads in the data from data/dataset.csv, returns it using
    extract_dictionary2, and may use [lemmatize, stop word, POS Tagging] for every sentence
    Do tfidf, and trian it. 
    The labels are multiclass as follows
        -1: poor
         0: average
         1: good
    Also returns the dictionary used to create X_train.
    Input:
        class_size: Size of each class (pos/neg/neu) of training dataset.
    """
    fname = "data/dataset.csv"
    dataframe = pd.read_csv(fname) 
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
    # Y_train = X_train["label"].values.copy()
    return X_train


def clean_sentence(input_string):
    """Preprocess review text into list of tokens.

    Convert input string to lowercase, replace punctuation with spaces
    Return the resulting array.

    E.g.
    > extract_word("I love EECS 445. It"s my favorite course!")
    > ["i love eecs 445 it s my favorite course "]
    """
    for letter in input_string:
        if letter in string.punctuation:
            input_string = input_string.replace(letter, " ")
    input_string = input_string.lower()
    return input_string


def preprocess(text, lem = False, stop = False, pos = False):  # 用来对document中的每个句子做lemmatize去停用词判断词性等操作，得到变换后的句子。
    tokens = text.split()
    
    # stemmer = PorterStemmer()
    # tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words] # lemmatize and stop_word
    if lem == True:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    if stop == True:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if not token in stop_words]
    if pos == True:
        tagged_tokens = pos_tag(tokens)
        weighted_tokens = []
        for word, tag in tagged_tokens:
            # if tag == 'JJR':                      # have effect, but not the best
            #     weighted_tokens.extend([word]*2)
            # elif tag == 'JJS': 
            #     weighted_tokens.extend([word]*3) 
            if tag in ['JJR', 'JJS']: # "JJ", 'JJR', 'JJS', 'RBR', 'RBS'
                weighted_tokens.extend([word]*3) # may consider * or not
            else:
                weighted_tokens.append(word)
        tokens = weighted_tokens

    return ' '.join(tokens)


def feature_engineering():
    train = get_data()
    # ---- new part ----
    docs = [] # Store train data, each element is a sentence
    test_docs = [] # Store test data
    train, test_data, label, test_lebal = train_test_split(train['reviewText'], train['label'], 
                                                           test_size=0.2, random_state=445, stratify=train['label']) # random_state=445
    for text in train:
        docs.append(clean_sentence(text))
    for text in test_data:
        test_docs.append(text)
    
    processed_docs = [preprocess(text, lem = True, pos = True) for text in docs] # lem = True, stop = True, pos = True
    processed_test_docs = [preprocess(text, lem = True, pos = True) for text in test_docs] # lem = True, stop = True, pos = True
    # processed_docs = [text for text in docs]  # origin sentences with out preprocess (only use clean_sentence)
    # processed_test_docs = [text for text in test_docs]

    vectorizer = TfidfVectorizer(ngram_range=(1,3)) # top_words='english'
    X_train = vectorizer.fit_transform(processed_docs).toarray()
    y_train = label.values
    X_test = vectorizer.transform(processed_test_docs).toarray()
    y_test = test_lebal.values
    
    return X_train, y_train, X_test, y_test


def time_series():
    train = get_data()
    sorted_train = train.sort_values(by='unixReviewTime', ascending=True)
    # -----------------
    docs = [clean_sentence(text) for text in sorted_train['reviewText']]
    label_docs = sorted_train['label'].values

    tscv = TimeSeriesSplit(n_splits=5)
    for train_idx, test_idx in tscv.split(docs):
        X_train, X_test = [docs[i] for i in train_idx], [docs[i] for i in test_idx]
        y_train, y_test = [label_docs[i] for i in train_idx], [label_docs[i] for i in test_idx]
        
    processed_X_train = [preprocess(text, lem=True, pos=True) for text in X_train]
    processed_X_test = [preprocess(text, lem=True, pos=True) for text in X_test]
        
    vectorizer = TfidfVectorizer(ngram_range=(1,3))
    X_train = vectorizer.fit_transform(processed_X_train).toarray()
    X_test = vectorizer.transform(processed_X_test).toarray()
    
    return X_train, y_train, X_test, y_test

# ---- another implementation----
    # scores = []
    # for train_idx, test_idx in tscv.split(docs):
    #     X_train, X_test = [docs[i] for i in train_idx], [docs[i] for i in test_idx]
    #     y_train, y_test = label_docs[train_idx], label_docs[test_idx]
        
    #     processed_X_train = [preprocess(text, lem=True, pos=True) for text in X_train]
    #     processed_X_test = [preprocess(text, lem=True, pos=True) for text in X_test]
    #     vectorizer = TfidfVectorizer(ngram_range=(1,3))
    #     X_train = vectorizer.fit_transform(processed_X_train).toarray()
    #     X_test = vectorizer.transform(processed_X_test).toarray()

    #     clf.fit(X_train, y_train)
    #     y_pred = clf.predict(X_test)

    #     score = performance(y_test, y_pred)
    #     scores.append(score)

    # return np.array(scores).mean()


# def time_series_select():
#     best_c = None
#     best_score = -float("inf")

#     clf = LinearSVC(penalty="l2", loss="hinge", dual=True, C=1, random_state=445)
#     new_score = cv_performance(clf, X, y, k=5)
#     if new_score > best_score:
#         best_score = new_score
#     return best_c, best_score


def svm_ovo(X_train, y_train, X_test, y_test, model, normalized=False, C=0.01):
    if normalized:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train_normalized = scaler.transform(X_train)
        X_test_normalized = scaler.transform(X_test)
    else:
        X_train_normalized = X_train
        X_test_normalized = X_test
    if model == 'ovr':
        ovo = LinearSVC(multi_class='ovr',C=C,penalty='l2', loss='hinge', dual=True)
    if model == 'ovo':
        ovo = SVC(decision_function_shape='ovo',kernel='rbf',gamma='scale', C = C) # ply is worse
    ovo.fit(X_train_normalized, y_train)
    predicted = ovo.predict(X_test_normalized)
    return performance(y_test, predicted, metric="accuracy")


def performance(y_true, y_pred, metric="accuracy"):
    if metric == "accuracy":
        return metrics.accuracy_score(y_true, y_pred)
    elif metric == "f1_macro":
        return metrics.f1_score(y_true, y_pred, average="macro")

    else:
        raise ValueError("Unsupported metric. Supported metrics are: \"accuracy\", \"f1_macro\"")


def cv_performance(clf, X, y, k=5, metric="accuracy"):
    skf = StratifiedKFold(n_splits=k)
    scores = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        score = performance(y_test, y_pred, metric)
        scores.append(score)

    return np.array(scores).mean()


def select_linear(X_train, y_train, ply=["l2", "l1"], metric="accuracy"):
    C = [0.001, 0.01, 0.1, 1, 10, 100]
    if ply == "l2":
        ls = "hinge"
        dl = True
    else:
        ls = "squared_hinge"
        dl = False
    best_c = None
    best_score = -float("inf")
    best_c_cs = None
    best_score_cs = -float("inf")
    best_c_ovo = None
    best_score_ovo = -float("inf")
    for c in C:
        clf = LinearSVC(penalty=ply, loss=ls, dual=dl, C=c, random_state=445)
        clf_cs = LinearSVC(penalty=ply, loss=ls, dual=dl, C=c, multi_class='crammer_singer', random_state=445)
        clf_ovo = OneVsOneClassifier(LinearSVC(penalty=ply, loss=ls, dual=dl, C=c, random_state=445))
        new_score = cv_performance(clf, X_train, y_train, k=5, metric=metric)
        new_score_cs = cv_performance(clf_cs, X_train, y_train, k=5, metric=metric)
        new_score_ovo = cv_performance(clf_ovo, X_train, y_train, k=5, metric=metric)
        if new_score > best_score:
            best_c = c
            best_score = new_score
        if new_score_cs > best_score_cs:
            best_c_cs = c
            best_score_cs = new_score_cs
        if new_score_ovo > best_score_ovo:
            best_c_ovo = c
            best_score_ovo = new_score
    return best_c, best_score, best_c_cs, best_score_cs, best_c_ovo, best_score_ovo


def test_perf(clf, X, y, X_test, y_test, metrics = ["accuracy", "f1_macro"]):
    clf.fit(X, y)
    output = {"Performance Measures": [], "Performance": []}
    for metric in metrics:
        y_pred = clf.predict(X_test)

        score = performance(y_test, y_pred, metric)
        output["Performance Measures"].append(metric)
        output["Performance"].append(score)
        
    output_df = pd.DataFrame(output)
    print(output_df)
    print("\n")


def feature_engineering_all():
    all_train = get_data()

    docs = [] # Store train data, each element is a sentence

    all_y = all_train['label']

    for text in all_train['reviewText']:
        docs.append(clean_sentence(text))
    
    processed_docs = [preprocess(text, lem = True, pos = True) for text in docs]

    vectorizer = TfidfVectorizer(ngram_range=(1,3))
    all_X = vectorizer.fit_transform(processed_docs).toarray()
    
    return all_X, all_y, vectorizer

def get_heldout_reviews_mine(vectorizer):
    fname = "data/heldout.csv"
    dataframe = load_data(fname)

    docs = [] # Store train data, each element is a sentence
    for text in dataframe["reviewText"]:
        docs.append(clean_sentence(text))
    
    processed_docs = [preprocess(text, lem = True, pos = True) for text in docs] 

    X_transformed = vectorizer.transform(processed_docs)
    return X_transformed


def main():
#     # I'll use new one
#     heldout_features = get_heldout_reviews(multiclass_dictionary) 
    (
        multiclass_features,
        multiclass_labels,
        multiclass_dictionary,
    ) = get_multiclass_training_data()

# ----------------------------------------------------------------------
    X_train, y_train, X_test, y_test = feature_engineering()
    # X_train, y_train, X_test, y_test = time_series()

# ---------- LinearSVC: select para with CV ----------
    # best_c, best_score, best_c_cs, best_score_cs, best_c_ovo, best_score_ovo = select_linear(X_train, y_train, "l2", "f1_macro")
    # best_c1, best_score1, best_c_cs1, best_score_cs1, best_c_ovo1, best_score_ovo1 = select_linear(X_train, y_train, "l1", "f1_macro")
    # op = {"Classifier": ["ovr", "crammer_singer", "ovo", "ovr", "crammer_singer", "ovo"], 
    #     "Best C": [best_c, best_c_cs, best_c_ovo, best_c1, best_c_cs1, best_c_ovo1], 
    #     "Penatly": ["l2", "l2", "l2", "l1", "l1", "l1"], 
    #     "CV score": [best_score, best_score_cs, best_score_ovo, best_score1, best_score_cs1,best_score_ovo1]}
    # op = pd.DataFrame(op)
    # print(op)

# ---------- LinearSVC: test para ----------
    clf = LinearSVC(penalty="l2", loss="hinge", dual=True, C=1, random_state=445) # multi_class='crammer_singer'
    test_perf(clf, X_train, y_train, X_test, y_test)                              # OneVsOneClassifier
# # ----------------------------------------------------------------------

# ---------------------------- Final Model -----------------------------
    # all_X, all_y, vectorizer = feature_engineering_all()
    # X_transformed = get_heldout_reviews_mine(vectorizer)

    # best_clf = LinearSVC(penalty="l2", loss="hinge", dual=True, C=1, random_state=445)
    # best_clf.fit(all_X, all_y)

    # y_pred = best_clf.predict(X_transformed)
    # generate_challenge_labels(y_pred, "qingyaoz")
# ----------------------------------------------------------------------

if __name__ == "__main__":
    main()

