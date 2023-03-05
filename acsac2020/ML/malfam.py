import argparse, sys
import pandas as pd
import numpy as np
import random
from time import time

#from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score

from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Prepare data for training and testing
def prepareData():
    # 0-2: full dataset; 3-5: remove families of sample size < 200
    datasets = ["../Dataset/MalFamTrainDataset.csv",
                "../Dataset/MalFamTestDataset.csv",
                "../Dataset/malicious_dataset_ALL_2020.csv",
                "../Dataset/MalFamTrainDatasetRemoveless200.csv",
                "../Dataset/MalFamTestDatasetRemoveless200.csv",
                "../Dataset/MalFamDatasetRemoveless200.csv"]

    #le = LabelEncoder()
    train_df = pd.read_csv(datasets[0], encoding="utf8", error_bad_lines=True)
    print("Train dataset length: {}".format(len(train_df)))
    x_train = train_df["abstracted_api_info"]
    y_train = train_df["familyname"]
    #le.fit(train_df["familyname"])
    #y_train = le.transform(train_df["familyname"])

    test_df = pd.read_csv(datasets[1], encoding="utf8", error_bad_lines=True)
    print("Test dataset length: {}".format(len(test_df)))
    x_test = test_df["abstracted_api_info"]
    y_test = test_df["familyname"]
    #y_test = le.transform(test_df["familyname"])

    full_df = pd.read_csv(datasets[2], encoding="utf8", error_bad_lines=True)
    print("Full dataset length: {}".format(len(full_df)))
    x = full_df["abstracted_api_info"]
    y = full_df["familyname"]

    return x_train, x_test, y_train, y_test, x, y

# Method 1: Tokenizer for api calls and arguments
def apiTokenizer1(s):
    return list(filter(lambda x:x!='', s.split(" ")))

# Method 2: Tokenizer for api calls and arguments
def apiTokenizer2(s):
    s = s.replace(';', ' ')
    return list(filter(lambda x:x!='', s.split(" ")))

# Run model training and testing for malware family classification
def classification(args):
    ## Prepare data
    x_train, x_test, y_train, y_test, x, y = prepareData()
    num_classes = len(set(y))
    print("Total number of malware families in dataset: {}".format(num_classes))

    ## Initialise Hashing Vectorizer wrt method-type
    if args.method == 1:
        hash_vec = HashingVectorizer(input="content",
                                     lowercase=True,
                                     ngram_range=(1,5),
                                     analyzer="word",
                                     n_features=2**20,
                                     binary=False,
                                     norm=None,
                                     token_pattern=None,
                                     tokenizer=apiTokenizer1)
    elif args.method == 2:
        hash_vec = HashingVectorizer(input="content",
                                     lowercase=True,
                                     ngram_range=(1,5),
                                     analyzer="word",
                                     n_features=2**20,
                                     binary=False,
                                     norm=None,
                                     token_pattern=None,
                                     tokenizer=apiTokenizer2)
        
    ## Vectorize the features; api calls
    x_train = hash_vec.fit_transform(x_train)
    x_test = hash_vec.fit_transform(x_test)
    x = hash_vec.fit_transform(x)

    ## Start supervised training with x_train and y_train
    print("Start training with classifier {}, using method {} features"\
          .format(args.classifier, args.method))

    ## Get classifier
    if args.classifier == "LSVC":
        clf = LinearSVC()
        cv_clf = LinearSVC()
    elif args.classifier == "SVC":
        clf = SVC(kernel="linear")
        cv_clf = SVC(kernel="linear")
    elif args.classifier == "RFC":
        clf = RandomForestClassifier(n_estimators=1000,
                                     max_depth=100,
                                     n_jobs=-1)
        cv_clf = RandomForestClassifier(n_estimators=1000,
                                        max_depth=100,
                                        n_jobs=-1)
    elif args.classifier == "PAC":
        clf = PassiveAggressiveClassifier(loss="squared_hinge")
        cv_clf = PassiveAggressiveClassifier(loss="squared_hinge")
    elif args.classifier == "DTC":
        clf = DecisionTreeClassifier(max_depth=100)
        cv_clf = DecisionTreeClassifier(max_depth=100)
    elif args.classifier == "XGBC":
        clf = XGBClassifier(booster=args.booster,
                            objective="multi:softmax",
                            num_class=num_classes,
                            n_jobs=-1)
        cv_clf = XGBClassifier(booster=args.booster,
                               objective="multi:softmax",
                               num_class=num_classes,
                               n_jobs=-1)
        assert (clf.get_xgb_params()["booster"]) == (cv_clf.get_xgb_params()["booster"])
        print("Booster used: {}".format(clf.get_xgb_params()["booster"]))
    else:
        print("[ERROR] Invalid classifier chosen: {}".format(args.classifier))
        sys.exit(0)

    if args.crossval:
        ## Cross validation start
        T0 = time()
        crossval_scores = cross_val_score(cv_clf, x, y, cv=5, n_jobs=-1)
        print("5 folds cross-validation done in {}s".format(round(time()-T0, 2)))
        print("Cross validation scores: {}".format(crossval_scores))
        print("Mean Cross-Val score: %0.3f (+/- %0.3f)" % (crossval_scores.mean(), crossval_scores.std() * 2))

    ## Training start
    T0 = time()
    clf.fit(x_train, y_train)
    print("Training done in {}s".format(round(time()-T0, 2)))

    # Get results
    T0 = time()
    ## Get accuracy score
    acc_score = clf.score(x_test, y_test)
    print("Accuracy Score: {}".format(acc_score))
    ## Do prediction on test set 
    pred_results = clf.predict(x_test)
    print("Prediction done in {}s".format(round(time()-T0, 2)))
    ## Get classification report
    print("Classification Report:\n{}".format(classification_report(y_test,
                                                                    pred_results,
                                                                    #digits=4,
                                                                    target_names=clf.classes_,
                                                                    zero_division=0)))
                                                                    
    # Get confusion matrix
    cfmx = confusion_matrix(y_test, pred_results)
    print("Confusion Matrix :\n{}".format(cfmx))

# Main function
if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Classification of Malware Families")
    args.add_argument("--method", type=int, default=1,
                      help="Type of feature extraction method to be used; 1(default) or 2")
    args.add_argument("--classifier", type=str, default="XGBC",
                      help="Type of classifier to be used;\
                            LSVC: Linear Support Vector Classifier,\
                            SVC: Support Vector Classifier,\
                            RFC: Random Forest Classifier,\
                            PAC: Passive Aggressive Classifier,\
                            DTC: Decision Tree Classifier,\
                            XGBC: XGBoosting Classifier(default)")
    args.add_argument("--booster", type=str, default="dart",
                      help="Type of booster to be used for XGBC; gbtree, gblinear, dart(default)")
    args.add_argument("--crossval", action="store_true",
                      help="Run 5-folds cross validation")
    # Do malware family classification
    classification(args.parse_args())    



    
