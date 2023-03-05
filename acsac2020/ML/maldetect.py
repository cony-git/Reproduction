import argparse, sys
import pandas as pd
import numpy as np
import random
from time import time
from matplotlib import pyplot as plt

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
    datasets=["../Dataset/benign_dataset_ALL_2020.csv",
              "../Dataset/malicious_dataset_ALL_2020.csv"]
    ## Get datasets as data frames
    benign_df = pd.read_csv(datasets[0], encoding="utf-8")
    mal_df = pd.read_csv(datasets[1], encoding="utf-8", error_bad_lines=False)
    print("Benign Dataset Len: {}, Malicious Dataset Len: {}".format(len(benign_df),
                                                                     len(mal_df)))

    ## Add <label> to data frames; 0 for benign, 1 for malicious
    benign_df["label"] = np.zeros(len(benign_df), dtype=np.int8)
    mal_df["label"] = np.ones(len(mal_df), dtype=np.int8)

    ## Concatate both dataframes into one dataframe
    combine_df = pd.concat([benign_df, mal_df], ignore_index=True)
    print("Full Dataset Len: {}".format(len(combine_df)))

    return combine_df

# Method 1: Tokenizer for api calls and arguments
def apiTokenizer1(s):
    return list(filter(lambda x:x!='', s.split(" ")))

# Method 2: Tokenizer for api calls and arguments
def apiTokenizer2(s):
    s = s.replace(";", " ")
    return list(filter(lambda x:x!='', s.split(" ")))

# Run model training and testing for malware detection
def detection(args):
    # Prepare data
    df = prepareData()

    # Get hash values of data
    data_hash = list(df["sha1"])

    # Get labels
    y = df["label"]

    # Initialise Hashing Vectorizer wrt method-type
    if args.method == 1:
        hash_vec = HashingVectorizer(input="content",
                                    lowercase=True,
                                    ngram_range=(1,5),
                                    analyzer="word",
                                    n_features=2**20,
                                    binary=True,
                                    norm=None,
                                    token_pattern=None,
                                    tokenizer=apiTokenizer1)
    elif args.method == 2:
        hash_vec = HashingVectorizer(input="content",
                                    lowercase=True,
                                    ngram_range=(1,5),
                                    analyzer="word",
                                    n_features=2**20,
                                    binary=True,
                                    norm=None,
                                    token_pattern=None,
                                    tokenizer=apiTokenizer2)
        
    # Vectorize the features; api calls
    x = hash_vec.fit_transform(df["abstracted_api_info"])

    # Split the features and labels into training sets and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=args.testSize,
                                                        random_state=args.randomSeed,
                                                        stratify=y)
    print("Dataset shuffled by random seed of {}".format(args.randomSeed))
    print("Training and test sets split by a ratio of {}:{}".format(int((1-args.testSize)*10),
                                                                    int(args.testSize*10)))

    test_set = list(y_test.items())
    with open("{}{}_testset_{}.txt".format(args.classifier, args.randomSeed, len(y_test)), "w+") as txtfile:
        for test_sample in test_set:
            txtfile.write("{}\n".format(test_sample))
    
    # Start supervised training with x_train and y_train
    print("Start training with classifier {}, using method {} features"\
          .format(args.classifier, args.method))
    
    ## Get classifier
    if args.classifier == "LSVC":
        clf = LinearSVC() #random_state=1,max_iter=10000)
        cv_clf = LinearSVC() #random_state=1,max_iter=10000)
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
                            objective="binary:logitraw",
                            n_jobs=-1)
        cv_clf = XGBClassifier(booster=args.booster,
                               objective="binary:logitraw",
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
                                                                    digits=4,
                                                                    target_names=["Benign",
                                                                                  "Malicious"])))
    # Get confusion matrix
    cfmx = confusion_matrix(y_test, pred_results)
    print("Confusion Matrix(1st Row:TP,FP; 2nd Row:FN,TN):\n{}".format(cfmx))

    # Get misclassified samples
    fp = []
    fn = []
    misclassified = []
    for i in range(len(y_test)):
        if pred_results[i] != test_set[i][1]:
            misclassified.append(i)
            # False positives; misclassified benign sample
            if int(pred_results[i]) == 1: 
                fp.append([test_set[i][0], data_hash[test_set[i][0]]])
            # False negatives; misclassified mal sample
            elif int(pred_results[i]) == 0:
                fn.append([test_set[i][0], data_hash[test_set[i][0]]])
    print("Misclassified samples: Total={}, FP={}, FN={}".format(len(fp)+len(fn),
                                                                 len(fp),
                                                                 len(fn)))
    
    # Save misclassified samples into text files(for analysis)
    with open("{}{}_fp_{}.txt".format(args.classifier, args.randomSeed, len(fp)), "w+") as txtfile:
        for sample in fp:
            txtfile.write("{}\n".format(sample))
    with open("{}{}_fn_{}.txt".format(args.classifier, args.randomSeed, len(fn)), "w+") as txtfile:
        for sample in fn:
            txtfile.write("{}\n".format(sample))
            
    if args.graph:
        # Plot figures, showing misclassified samples
        x1 = np.linspace(0, len(y_test), len(y_test))
        x2 = np.linspace(0, len(y_test), len(y_test))
        plt.plot(x1, y_test.values, "o", color="grey", markersize=10, label="Real")
        plt.plot(x2, pred_results, "x", color="black", markersize=10, label="Expected")
        plt.plot(x2, pred_results, "|",color="red", markersize=20,
                 markevery=misclassified, label="Misclassified (%s)" %len(misclassified))
        plt.yticks([0,1])
        plt.legend(loc='best')
        plt.xlabel("Sample No.",fontsize=12)
        plt.ylabel("Benign=0, Malicious=1",fontsize=12)
        plt.savefig("{}{}_Method{}.png".format(args.classifier, args.randomSeed, args.method))
        plt.show()

# Main function
if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Detection of Malicious Windows Applications")
    args.add_argument("--testSize", type=float, default=0.2,
                      help="Size of test set to be split; default 0.2")
    args.add_argument("--randomSeed", type=int, default=random.randint(0, 100),
                      help="Option to set random seed")
    args.add_argument("--graph", action="store_true",
                      help="Option to show graph")
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
                      help="Run cross validation")
    # Do malware detection
    detection(args.parse_args())    
