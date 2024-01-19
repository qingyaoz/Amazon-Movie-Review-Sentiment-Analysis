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
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
# nltk.download()

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)

np.random.seed(445)



def extract_word(input_string):
    """Preprocess review text into list of tokens.

    Convert input string to lowercase, replace punctuation with spaces, and split along whitespace.
    Return the resulting array.

    Input:
        input_string: text for a single review
    Returns:
        a list of words, extracted and preprocessed according to the directions
        above.
    """
    for letter in input_string:
        if letter in string.punctuation:
            input_string = input_string.replace(letter, " ")
    input_string = input_string.lower()
    return input_string.split()

def extract_dictionary(df):
    """Map words to index.

    E.g., with input:
        | reviewText                    | label | ... |
        | It was the best of times.     |  1    | ... |
        | It was the blurst of times.   | -1    | ... |

    The output should be a dictionary of indices ordered by first occurence in
    the entire dataset:
        {
           it: 0,
           was: 1,
           the: 2,
           best: 3,
           of: 4,
           times: 5,
           blurst: 6
        }
    The index is autoincrementing, starting at 0.

    Input:
        df: dataframe/output of load_data()
    Returns:
        a dictionary mapping words to an index
    """
    word_dict = {}
    word_list = []
    for review in df["reviewText"]:
        word_list.extend(extract_word(review))

    idx = 0
    for word in word_list:
        if word not in word_dict:
            word_dict[word] = idx
            idx += 1

    return word_dict


def generate_feature_matrix(df, word_dict):
    """Create matrix of feature vectors for dataset.
    Input:
        df: dataframe that has the text and labels
        word_dict: dictionary of words mapping to indices
    Returns:
        a numpy matrix of dimension (# of reviews, # of words in dictionary)
    """
    number_of_reviews = df.shape[0]
    number_of_words = len(word_dict)
    feature_matrix = np.zeros((number_of_reviews, number_of_words))
    for r in range(number_of_reviews):
        words_list = extract_word(df["reviewText"][r])
        for w in words_list:
            if w in word_dict:
                feature_matrix[r, word_dict[w]] = 1
    return feature_matrix


def performance(y_true, y_pred, metric="accuracy"):
    """Calculate performance metrics.
    
    Input:
        y_true: (n,) array containing known labels
        y_pred: (n,) array containing predicted scores
        metric: string specifying the performance metric (default="accuracy"
                 other options: "f1-score", "auroc", "precision", "sensitivity",
                 and "specificity")
    Returns:
        the performance as an np.float64
    """
    if metric == "accuracy":
        return metrics.accuracy_score(y_true, y_pred)
    elif metric == "f1-score":
        return metrics.f1_score(y_true, y_pred) # Zero_division = 1?
    elif metric == "auroc":
        return metrics.roc_auc_score(y_true, y_pred) # Zero_division = 1?
    elif metric == "precision":
        return metrics.precision_score(y_true, y_pred) # Zero_division?
    elif metric == "sensitivity":
        return metrics.recall_score(y_true, y_pred) # Zero_division?
    elif metric == "specificity":
        conf_matrix = metrics.confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = conf_matrix.ravel()
        return tn / (tn + fp) # Zero_division怎么处理？
    else:
        raise ValueError("Unsupported metric. Supported metrics are: \"accuracy\", \"f1-score\", \"auroc\", \"precision\", \"sensitivity\", \"specificity\"")


def cv_performance(clf, X, y, k=5, metric="accuracy"):
    """Split data into k folds and run cross-validation.

    Input:
        clf: an instance of SVC()
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default="accuracy"
             other options: "f1-score", "auroc", "precision", "sensitivity",
             and "specificity")
    Returns:
        average "test" performance across the k folds as np.float64
    """

    skf = StratifiedKFold(n_splits=k)
    # Put the performance of the model on each fold in the scores array
    scores = []

    # for i, (train_index, test_index) in enumerate(skf.split(X, y)):
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

    
def select_param_linear(
    X, y, k=5, metric="accuracy", C_range=[], loss="hinge", penalty="l2", dual=True
):
    """Search for hyperparameters from the given candidates of linear SVM with
    best k-fold CV performance.

    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default="accuracy",
             other options: "f1-score", "auroc", "precision", "sensitivity",
             and "specificity")
        C_range: an array with C values to be searched over
        loss: string specifying the loss function used (default="hinge",
             other option of "squared_hinge")
        penalty: string specifying the penalty type used (default="l2",
             other option of "l1")
        dual: boolean specifying whether to use the dual formulation of the
             linear SVM (set True for penalty "l2" and False for penalty "l1")
    Returns:
        the parameter value for a linear-kernel SVM that maximizes the
        average 5-fold CV performance.
    """
    best_c = None
    best_score = -float("inf")
    for c in C_range:
        clf = LinearSVC(penalty=penalty, loss=loss, dual=dual, C=c, random_state=445)
        new_score = cv_performance(clf, X, y, k=k, metric=metric)
        # print(new_score) # to see how cv performance vary accross different C
        if new_score > best_score:
            best_c = c
            best_score = new_score
    return best_c, best_score


def plot_weight(X, y, penalty, C_range, loss, dual):
    """Create a plot of the L0 norm learned by a classifier for each C in C_range.

    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
        and d is the number of features
        y: (n,) array of binary labels {1,-1}
        penalty: string for penalty type to be forwarded to the LinearSVC constructor
        C_range: list of C values to train a classifier on
        loss: string for loss function to be forwarded to the LinearSVC constructor
        dual: whether to solve the dual or primal optimization problem, to be
            forwarded to the LinearSVC constructor
    Returns: None
        Saves a plot of the L0 norms to the filesystem.
    """
    norm0 = []
    for c in C_range:
        clf = LinearSVC(penalty=penalty, loss=loss, dual=dual, C=c, random_state=445)
        clf.fit(X, y)
        theta = clf.coef_
        norm0.append(np.count_nonzero(theta))

    plt.plot(C_range, norm0)
    plt.xscale("log")
    plt.legend(["L0-norm"])
    plt.xlabel("Value of C")
    plt.ylabel("Norm of theta")
    plt.title("Norm-" + penalty + "_penalty.png")
    plt.savefig("Norm-" + penalty + "_penalty.png")
    plt.close()


def select_param_quadratic(X, y, k=5, metric="accuracy", param_range=[]):
    """Search for hyperparameters from the given candidates of quadratic SVM
    with best k-fold CV performance.

    Input:
        X: (n,d) array of feature vectors, where n is the number of examples
           and d is the number of features
        y: (n,) array of binary labels {1,-1}
        k: an int specifying the number of folds (default=5)
        metric: string specifying the performance metric (default="accuracy"
                 other options: "f1-score", "auroc", "precision", "sensitivity",
                 and "specificity")
        param_range: a (num_param, 2)-sized array containing the
            parameter values to search over. The first column should
            represent the values for C, and the second column should
            represent the values for r. Each row of this array thus
            represents a pair of parameters to be tried together.
    Returns:
        The parameter values for a quadratic-kernel SVM that maximize
        the average 5-fold CV performance as a pair (C,r)
    """
    best_C_val, best_r_val = 0.0, 0.0
    best_score = -float("inf")
    # best_C_val_r, best_r_val_r = 0.0, 0.0
    # best_score_r = -float("inf")

    # ---- i) Grid Search ---- #
    # pass-in param_range is already sorted
    for c, r in param_range:
        clf = SVC(kernel="poly", degree=2, C=c, coef0=r, 
                    gamma="auto", random_state=445)  # quadratic SVM
        new_score = cv_performance(clf, X, y, k=k, metric=metric)
        print(f"C = {c}, r = {r}, score = {new_score}") # for 3.3b
        if new_score > best_score:
            best_C_val = c
            best_r_val = r
            best_score = new_score
        if new_score == best_score:
            if c < best_C_val:
                best_C_val = c
            else:
                if r < best_r_val:
                    best_r_val = r

    # ---- ii) Random Search ---- # Older version, implement two in one function
    # C_values = np.sort(10 ** np.random.uniform( -2, 3, 25))  -- C_values[r_idx]
    # r_values = np.sort(10 ** np.random.uniform(-2, 3, 25))
    #selected_idx = np.random.choice(param_range.shape[0], 25, replace=False)
    #values = param_range[np.sort(selected_idx)]
    # for r_idx in range(25):
    #     C_value = 10 ** np.random.uniform(-2, 3)
    #     r_value = 10 ** np.random.uniform(-2, 3)
    #     clf_r = SVC(kernel="poly", degree=2, C=C_value, coef0=r_value, \
    #                   gamma="auto", random_state=445)  # quadratic SVM
    #     new_score_r = cv_performance(clf_r, X, y, k=k, metric=metric)
    #     if new_score_r > best_score_r:
    #         best_C_val_r = C_value
    #         best_r_val_r = r_value
    #         best_score_r = new_score_r
    #     if new_score_r == best_score_r:
    #         if C_value < best_C_val_r:
    #             best_C_val_r = C_value
    #         else:
    #             if r_value < best_r_val_r:
    #                 best_r_val_r = r_value

    return best_C_val, best_r_val


def train_word2vec(fname):
    """
    Train a Word2Vec model using the Gensim library.
    """
    df = load_data(fname)
    sentences = []
    for rv in df["reviewText"]:
        sentences.append(extract_word(rv)) 
    
    return Word2Vec(sentences, workers=1)


def compute_association(fname, w, A, B):
    """
    Inputs:
        - fname: name of the dataset csv
        - w: a word represented as a string
        - A and B: sets that each contain one or more English words represented as strings
    Output: Return the association between w, A, and B as defined in the spec
    """
    model = train_word2vec(fname)

    # First, find a numerical representation for the English language words in A and B
    def words_to_array(set):
        num_words = len(set)
        dim = model.vector_size
        arr = np.zeros((num_words, dim))
        for idx, word in enumerate(set):
            # ensure no repeat words?
            arr[idx] = model.wv[word]
        return arr

    def cosine_similarity(set):
        array = words_to_array(set)
        w_ebd = model.wv[w]
        # cos_sml = np.zeros(len(set))
        cos_sml = (np.dot(array, w_ebd) / (np.linalg.norm(array, axis=1) * np.linalg.norm(w_ebd)))
        return cos_sml

    asso = np.mean(cosine_similarity(A)) - np.mean(cosine_similarity(B))
    return asso


# -------------------- Function for Output and visualization --------------------

#   ---------- helper Function- ---------
def test_perf(clf, X, y, X_test, y_test, 
              metrics = ["accuracy", "f1-score", "auroc", "precision", "sensitivity", "specificity"]):
    
    clf.fit(X, y)
    output = {"Performance Measures": [], "Performance": []}
    for metric in metrics:
        if metric != "auroc":
            y_pred = clf.predict(X_test)
        else:
            y_pred = clf.decision_function(X_test)

        score = performance(y_test, y_pred, metric)
        output["Performance Measures"].append(metric)
        output["Performance"].append(score)
        
    output_df = pd.DataFrame(output)
    print(output_df)
    print("\n")
#   ---------- helper Function- ---------

def q2(feature_matrix, dicts):
    word_list = extract_word("It’s a test sentence! Does it look CORRECT?")
    avg_nz = np.count_nonzero(feature_matrix)/feature_matrix.shape[0]
    max_word_idx = np.argmax(np.sum(feature_matrix, axis=0))
    max_word = ""
    for word, index in dicts.items():
        if index == max_word_idx:
            max_word = word
            break
    
    print("Reporting dataset statistics")
    print("The input sentence is \"It's a test sentence! Does it look CORRECT?\"")
    print(f"The processed sentence is {word_list}")
    print(f"d: {len(dicts)}")
    print(f"Average number of nonzero features: {avg_nz}")
    print(f"Most common word: {max_word}")
    print("\n")
    return None

def q3_1b(X, y):
    c_range = np.array([10**(-3), 10**(-2), 10**(-1), 10**0, 10**1, 10**2, 10**3])
    metrics = ["accuracy", "f1-score", "auroc", "precision", "sensitivity", "specificity"]
    output = {"Performance Measures": [], "C": [], "CV Performance": []}
    for metric in metrics:
        # print(metric)
        best_c,best_score = select_param_linear(X, y, k=5, metric=metric, C_range=c_range)

        output["Performance Measures"].append(metric)
        output["C"].append(best_c)
        output["CV Performance"].append(best_score)
    
    print("Linear SVM with grid search")
    output_df = pd.DataFrame(output)
    print(output_df)
    print("\n")

def q3_1c(X, y, X_test, y_test, best_c):
    clf = LinearSVC(penalty="l2", loss="hinge", dual=True, C=best_c, random_state=445)
    test_perf(clf, X, y, X_test, y_test)


def q3_1d(X, y):
    c_range = np.array([10**(-3), 10**(-2), 10**(-1), 10**0])
    plot_weight(X, y, "l2", c_range, "hinge", True)

def q3_1e(X, y, word_dict):
    c = 0.1
    clf = LinearSVC(loss="hinge", C=c, random_state=445)
    model = clf.fit(X, y)
    coefs = model.coef_[0]

    # find the five most negative and five most positive coefficients
    index = np.concatenate([np.argsort(coefs)[:5], np.argsort(coefs)[-5:]])
    mm_coefs = coefs[index]
    mm_words = np.array(list(word_dict.keys()))[index] 

    print("Displaying the most positive and negative words")
    #as debug output
    for i in range(len(mm_coefs)):
        print(f"coeff: {mm_coefs[i]} word: {mm_words[i]}")
    print("\n")

    # 创建bar chart
    colors = ['green', 'green', 'green', 'green', 'green', 'red', 'red', 'red', 'red', 'red']
    plt.bar(mm_words, mm_coefs, color=colors)
    plt.xticks(rotation = 25)
    # 添加标题和标签
    plt.title("The Most Positive & Most Negative Coefficients")
    plt.xlabel("Words")
    plt.ylabel("Coefficients")
    plt.savefig("The Most Pos & Neg Coefficients")
    plt.close()

def q3_2a(X, y, X_test, y_test): 
    print("Linear SVM with l1-penalty, grid search and auroc")
    # print("By using training data, the C value with the best CV performance could be found as")
    c_range = np.array([10**(-3), 10**(-2), 10**(-1), 10**0])
    best_c, best_score = select_param_linear(X, y, k=5, metric="auroc", C_range=c_range, \
                                            loss = "squared_hinge", penalty="l1", dual=False)
    para_score = {"Best C": [best_c], "CV AUROC Score": [best_score]}    
    df = pd.DataFrame(para_score)
    print(df)

    # print("Performance of Prediction on the test set with this BEST C")
    clf = LinearSVC(penalty="l1", dual=False, C=best_c, random_state=445) # default loss="squared hinge"
    test_perf(clf, X, y, X_test, y_test, metrics=["auroc"])


def q3_2b(X, y): 
    c_range = np.array([10**(-3), 10**(-2), 10**(-1), 10**0])
    plot_weight(X, y, "l1", c_range, "squared_hinge", dual=False)
    

def q3_3a(X, y, X_test, Y_test): 
    print("i) Grid Search")
    # ------------ (i) ------------
    # ---- (i) param_range ----
    powers = np.linspace(-2, 3, 6)  # [-2., -1.,  0.,  1.,  2.,  3.]
    values = 10 ** powers
    C_values, r_values = np.meshgrid(values, values)
    p_range = np.vstack([C_values.ravel(), r_values.ravel()]).T
    p_range[:, [0, 1]] = p_range[:, [1, 0]]
    # ---- (i) param_range ----

    best_C_val_g, best_r_val_g = select_param_quadratic(X, y, k=5, metric="auroc", param_range=p_range)
    clf_g = SVC(kernel="poly", degree=2, C=best_C_val_g, coef0=best_r_val_g, gamma="auto", random_state=445)
    clf_g.fit(X, y)
    y_pred_g = clf_g.decision_function(X_test)
    test_auroc_g = metrics.roc_auc_score(Y_test, y_pred_g)
    # ------------ (i) ------------
    print(f"Tuning Scheme: Grid Search, C: {best_C_val_g}, r: {best_r_val_g}, AUROC: {test_auroc_g}")


    # output = {"Tuning Scheme": ["Grid Search", "Random Search"], "C": [best_C_val_g, best_C_val_r], \
    #           "r": [best_r_val_g, best_r_val_r], "AUROC": [test_auroc_g, test_auroc_r]}
    # output_df = pd.DataFrame(output)
    # print("Question 3.3(a)")
    # print(output_df)
    # print("\n")

# may improve
def q3_3aii(X, y, X_test, Y_test):
    # np.random.uniform(-2, 3)
    print("2) Random Search")
    # ------------ (ii) ------------
    p_range_rand = []
    for idx in range(25):
        epsilon = np.finfo(float).eps
        C_value = np.random.uniform(-2, 3+ epsilon)
        r_value = np.random.uniform(-2, 3+ epsilon)
        p_range_rand.append([10**C_value, 10**r_value])

    c, r = select_param_quadratic(X, y, 5, metric="auroc", param_range=p_range_rand)
    clf= SVC(kernel="poly", degree=2, C=c, coef0=r, gamma="auto", random_state=445)
    clf.fit(X, y)
    y_pred = clf.decision_function(X_test)
    test_auroc = metrics.roc_auc_score(Y_test, y_pred)
    print(f"Tuning Scheme: Random Search, C: {c}, r: {r}, AUROC: {test_auroc}")
    # ------------ (ii) ------------

def q4_1c(X, y, X_test, y_test):
    print("Linear SVM with imbalanced class weights")
    clf = LinearSVC(loss="hinge", penalty="l2", C=0.01, class_weight={-1: 1, 1: 10}, random_state=445) # Wn = 1 and Wp = 10
    test_perf(clf, X, y, X_test, y_test) 


def q4_2a(X, y, X_test, y_test):
    print("Linear SVM on an imbalanced data set")
    print("---- With Class Weights: Wn = 1, Wp = 1 ----")
    clf = LinearSVC(loss="hinge", penalty="l2", C=0.01, class_weight={-1: 1, 1: 1}, random_state=445) # Wn = 1 and Wp = 10
    test_perf(clf, X, y, X_test, y_test) 

def q4_3a(X, y):
    print("Finding appropriate class weights")
    output = {"Wn": [], "Wp": [], "Performance Measures": [], "CV Performance": []}
    metrics = ["accuracy", "f1-score", "auroc", "precision", "sensitivity", "specificity"]
    # metrics = ["f1-score", "auroc"]
    for metric in metrics:
        best_Wn = None
        best_Wp = None
        best_score = -float("inf")
        for i in np.arange(10, 0, -0.5):
            for j in np.arange(10, 0, -0.5):
                clf = LinearSVC(loss="hinge", penalty="l2", C=0.01, class_weight={-1: i, 1: j}, random_state=445)
                new_score = cv_performance(clf, X, y, k=5, metric=metric)
                if new_score >= best_score:
                    best_Wn = i
                    best_Wp = j
                    best_score = new_score
        output["Performance Measures"].append(metric)
        output["Wn"].append(best_Wn)
        output["Wp"].append(best_Wp)
        output["CV Performance"].append(best_score)

    output_df = pd.DataFrame(output)
    print(output_df)
    print("\n")
    

def q4_3b(X, y, X_test, y_test, Wn, Wp):
    print("Choosing appropriate class weights")
    print(f"---- With Class Weights: Wn = {Wn}, Wp = {Wp} ----")
    clf = LinearSVC(loss="hinge", penalty="l2", C=0.01, class_weight={-1: Wn, 1: Wp}, random_state=445) # Wn = 1 and Wp = 10
    test_perf(clf, X, y, X_test, y_test)


def q4_4(X, y, X_test, y_test, Wn, Wp):
    clf = LinearSVC(loss="hinge", penalty="l2", C=0.01, class_weight={-1: 1, 1: 1}, random_state=445)
    clf.fit(X, y)
    y_scores = clf.decision_function(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_scores)
    roc_auc = metrics.roc_auc_score(y_test, y_scores)

    clf_w = LinearSVC(loss="hinge", penalty="l2", C=0.01, class_weight={-1: Wn, 1: Wp}, random_state=445)
    clf_w.fit(X, y)
    y_scores_w = clf_w.decision_function(X_test)
    fpr_w, tpr_w, thresholds = metrics.roc_curve(y_test, y_scores_w)
    roc_auc_w = metrics.roc_auc_score(y_test, y_scores_w)

    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label="Wn = 1, Wp = 1, AUC = %0.5f" % roc_auc)
    plt.plot(fpr_w, tpr_w, color="green", lw=2, label="ROC curve (Custom Wn, Wp) (AUC = %0.5f)" % roc_auc_w)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.title("ROC Curve of IMB Data")
    plt.legend(loc="lower right")
    plt.savefig("ROC Curve of IMB Data")


def q5_1a(fname):
    print("Number of reviews containing the words \"actor(s)\" and \"actress(es)\"")
    actor, actress = count_actors_and_actresses(fname)
    output_df = pd.DataFrame({"Containing Words": ["Actor(s)", "Actress(s)"], "Number of Reviews": [actor, actress]})
    print(output_df)
    print("\n")


def q5_1b(fname):
    plot_actors_and_actresses(fname, "label")


def q5_1c(fname):
    plot_actors_and_actresses(fname, "rating")


def q5_1d(X, y, word_dict):
    print("Report the theta vector’s coefficients for the words \"actor\" and \"actress\"")
    clf = LinearSVC(loss="hinge", penalty="l2", C=0.1, random_state=445)
    model = clf.fit(X, y)
    m_coef = model.coef_[0, word_dict["actor"]]
    f_coef = model.coef_[0, word_dict["actress"]]
    print(f"Actor's Coefficient: {m_coef}")
    print(f"Actress's Coefficient: {f_coef}")
    print("\n")


def q5_2(fname):
    print("Report the dimensionality of word embedding.")
    edb_model = train_word2vec(fname)
    word_ebd = edb_model.wv["actor"]
    dim = edb_model.vector_size
    print(f"Word Embedding for 'actor': {word_ebd}")
    print(f"Dimensionality of Word Embedding: {dim}")

    print("Report the five most similar words to the word \"plot\"")
    words = edb_model.wv.most_similar("plot", topn=5)
    for item in words:
        print(f"Word: {item[0]}, Score: {item[1]}")
    print("\n")


def q5_3a(fname):
    she = {"her", "woman", "women"}
    he = {"him", "man", "men"}
    asso = compute_association(fname, "talented", she, he)
    print(f"Association between \"talented\" and the sets {{her, woman, wome\}} and {{him, man, men}}: {asso}")
    print("\n")
# -------------------- Function for Output and visualization --------------------

def main():

    # Read binary data
    fname = "data/dataset.csv"
    X_train, Y_train, X_test, Y_test, dictionary_binary = get_split_binary_data(
        fname="data/dataset.csv"
    )
    IMB_features, IMB_labels, IMB_test_features, IMB_test_labels = get_imbalanced_data(
        dictionary_binary, fname="data/dataset.csv"
    )


    q2(X_train, dictionary_binary)
    q3_1b(X_train, Y_train)
    best_c = 0.1
    q3_1c(X_train, Y_train, X_test, Y_test, best_c)
    q3_1d(X_train, Y_train)
    q3_1e(X_train, Y_train, word_dict = dictionary_binary)
    q3_2a(X_train, Y_train, X_test, Y_test)
    q3_2b(X_train, Y_train)
    # ---- 3.3(a) ----
    q3_3a(X_train, Y_train, X_test, Y_test)
    q3_3aii(X_train, Y_train, X_test, Y_test)
    # ---- 3.3(a) ----
    q4_1c(X_train, Y_train, X_test, Y_test)
    q4_2a(IMB_features, IMB_labels, IMB_test_features, IMB_test_labels)
    q4_3a(IMB_features, IMB_labels)
    q4_3b(IMB_features, IMB_labels, IMB_test_features, IMB_test_labels, Wn=10, Wp=5)
    q4_4(IMB_features, IMB_labels, IMB_test_features, IMB_test_labels, Wn=10, Wp=5)
    q5_1a(fname)
    q5_1b(fname)
    q5_1c(fname)
    q5_1d(X_train, Y_train, dictionary_binary)
    q5_2(fname)
    q5_3a(fname)

    # Read multiclass data
    (
        multiclass_features,
        multiclass_labels,
        multiclass_dictionary,
    ) = get_multiclass_training_data()

    heldout_features = get_heldout_reviews(multiclass_dictionary)


if __name__ == "__main__":
    main()
