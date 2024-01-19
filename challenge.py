import project
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
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

import nltk
# nltk.download('stopwords')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download()

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)

np.random.seed(445)

def performance(y_true, y_pred, metric="accuracy"):
    """Calculate performance metrics.

    Performance metrics are evaluated on the true labels y_true versus the
    predicted labels y_pred.

    Input:
        y_true: (n,) array containing known labels
        y_pred: (n,) array containing predicted scores
        metric: string specifying the performance metric (default="accuracy"
                 other options: "f1-score", "auroc", "precision", "sensitivity",
                 and "specificity")
    Returns:
        the performance as an np.float64
    """
    # See the sklearn.metrics documentation for pointers on how to implement
    # the requested metrics.
    if metric == "accuracy":
        return metrics.accuracy_score(y_true, y_pred)
    elif metric == "f1_macro":
        return metrics.f1_score(y_true, y_pred, average="macro") # Zero_division = 1?
    else:
        raise ValueError("Unsupported metric. Supported metrics are: \"accuracy\", \"f1_macro\"")


# -------------------- Function for Challenge --------------------
def linear_select_l2(X, y):
    # ---- penalty = l2, loss = hinge ----
    svc = LinearSVC()
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                  'penalty': ['l2'],
                  'loss': ['hinge'],
                  'multi_class': ['ovr', 'crammer_singer']}
    output = {"metrix": [], "Best cv score": [], "Parameters": []}
    models = dict()
    metrix = ["accuracy", "f1_macro"] # "precision_weighted", "recall_weighted"
    for m in metrix:
        grid_search = GridSearchCV(svc, param_grid, scoring=m, refit=m, cv=5)
        grid_search.fit(X, y)
        output["metrix"].append(m)
        output["Best cv score"].append(grid_search.best_score_)
        output["Parameters"].append(grid_search.best_params_)
        models[m] = grid_search.best_estimator_

    output_df = pd.DataFrame(output)
    pd.set_option('display.max_colwidth', None)
    print(output_df)
    return models
    
def linear_select_l1(X, y):
    # ---- penalty = l1, loss = squared hinge ----
    svc = LinearSVC()
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                  'penalty': ['l1'],
                  'loss': ['squared_hinge'], 
                  'dual': [False], 
                  'multi_class': ['ovr', 'crammer_singer']}
    output = {"metrix": [], "Best cv score": [], "Parameters": []}
    models = dict()
    metrix = ["accuracy", "f1_macro"]
    for m in metrix:
        grid_search = GridSearchCV(svc, param_grid, scoring=m, refit=m, cv=5)
        grid_search.fit(X, y)
        output["metrix"].append(m)
        output["Best cv score"].append(grid_search.best_score_)
        output["Parameters"].append(grid_search.best_params_)
        models[m] = grid_search.best_estimator_

    output_df = pd.DataFrame(output)
    print(output_df)
    return models

def test_linear(X_test, y_test, models):
    output = dict()
    for m, estimator in models.items():
        y_pred = estimator.predict(X_test)
    
        score = performance(y_test, y_pred, m)
        print(m, score)

def cv_performance(clf, X, y, k=5, metric="accuracy"):
    skf = StratifiedKFold(n_splits=k)
    scores = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        clf.fit(X_train, y_train)
        if metric != "auroc":
            y_pred = clf.predict(X_test)
        else:
            y_pred = clf.decision_function(X_test)

        score = performance(y_test, y_pred, metric)
        scores.append(score)

    return np.array(scores).mean()

def select_p(X_train, y_train, ply=["l2", "l1"], metric="accuracy"):
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

# ---------- LinearSVC: go over all para to see general trend ----------
def go_over_all(X_train, y_train, X_test, y_test):
    C = [0.001, 0.01, 0.1, 1, 10, 100]
    output = {"C": [], "Performance Measures": [], "Performance": []}
    for c in C:
        clf = OneVsOneClassifier(LinearSVC(penalty="l2", loss="hinge", dual=True, C=c, random_state=445)) # multi_class='crammer_singer'
        clf.fit(X_train, y_train)
        metrics = ["accuracy", "f1_macro"]
        for metric in metrics:
            y_pred = clf.predict(X_test)
            score = performance(y_test, y_pred, metric)
            output["Performance Measures"].append(metric)
            output["Performance"].append(score)
            output["C"].append(c)

    output_df = pd.DataFrame(output)
    print(output_df)
    print("\n")

    C2 = [0.001, 0.01, 0.1, 1, 10, 100]
    output2 = {"C": [], "Performance Measures": [], "Performance": []}
    for c2 in C2:
        clf2 = OneVsOneClassifier(LinearSVC(penalty="l1", loss="squared_hinge", dual=False, C=c2, random_state=445)) # multi_class='crammer_singer'
        clf2.fit(X_train, y_train)
        metrics2 = ["accuracy", "f1_macro"]
        for metric2 in metrics2:
            y_pred2 = clf2.predict(X_test)
            score2 = performance(y_test, y_pred2, metric2)
            output2["Performance Measures"].append(metric2)
            output2["Performance"].append(score2)
            output2["C"].append(c2)

    output_df2 = pd.DataFrame(output2)
    print(output_df2)

def main():
    
    # Read multiclass data
    fname = "data/dataset.csv"
    (
        multiclass_features,
        multiclass_labels,
        multiclass_dictionary,
    ) = get_multiclass_training_data()
    X_train, X_test, y_train, y_test = train_test_split(multiclass_features, multiclass_labels,
                                                        test_size=0.2, random_state=445, stratify=multiclass_labels)
    heldout_features = get_heldout_reviews(multiclass_dictionary)

    # go_over_all(X_train, y_train, X_test, y_test)

    # ---------- LinearSVC: select para with CV ----------
    # best_c, best_score, best_c_cs, best_score_cs, best_c_ovo, best_score_ovo = select_p(X_train, y_train, "l2", "f1_macro")
    # best_c1, best_score1, best_c_cs1, best_score_cs1, best_c_ovo1, best_score_ovo1 = select_p(X_train, y_train, "l1", "f1_macro")
    # op = {"Classifier": ["ovr", "crammer_singer", "ovo", "ovr", "crammer_singer", "ovo"], 
    #       "Best C": [best_c, best_c_cs, best_c_ovo, best_c1, best_c_cs1, best_c_ovo1], 
    #       "Penatly": ["l2", "l2", "l2", "l1", "l1", "l1"], 
    #       "CV score": [best_score, best_score_cs, best_score_ovo, best_score1, best_score_cs1,best_score_ovo1]}
    # op = pd.DataFrame(op)
    # print(op)
    # ---------- LinearSVC: test para ----------
    clf = LinearSVC(penalty="l2", loss="hinge", dual=True, C=0.1, multi_class='crammer_singer', random_state=445) # multi_class='crammer_singer'
    test_perf(clf, X_train, y_train, X_test, y_test)
    # ----------------------------------------------------------------------

if __name__ == "__main__":
    main()