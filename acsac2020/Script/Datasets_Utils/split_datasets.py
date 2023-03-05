import argparse, sys, os
from time import time
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
import collections

CLASS_OPT = ["maldetect", "maltype", "malfamily"]
SHUFFLE_SEED = 0

# Count sample size of each label
def count_labels(labels_list):
    labels_dict = {}
    for label in labels_list:
        if label not in labels_dict.keys():
            labels_dict[label] = 1
        else: labels_dict[label]+=1
    labels_ord_dict = collections.OrderedDict(sorted(labels_dict.items()))
    return labels_ord_dict

# Get the dataframe wrt indexes given and save the dataframe
def get_datasets(fold_no, df_full, folder_dir, index_set, typ_class, typ_set):
    df_set = df_full.loc[list(index_set)]
    assert len(df_set) == len(index_set)
    labels_dict = count_labels(list(df_set['label']))
    print("[Fold {}] {}: {}\n{}".format(fold_no, typ_set, len(df_set), labels_dict))
    csvfilename = os.path.join(folder_dir, "{}_{}_fold{}.csv".format(typ_class, typ_set, fold_no))
    df_set.to_csv(csvfilename, encoding="utf-8", index=False)
    print("{} dataset as {}".format(typ_set, csvfilename))

# Open full datasets as dataframes and add labels
def process_datasets(maldata, benigndata, classtyp):
    # Get malicious dataset as dataframe
    df_mal = pd.read_csv(maldata, encoding="utf-8", error_bad_lines=False)
    print("[Mal] Dataset len = {}, Headers = {}".format(len(df_mal), df_mal.columns))
    # For malware detection
    if classtyp == "maldetect":
        if "maltype" in df_mal.columns:
            df_mal = df_mal.drop(columns=["maltype"])
        ## Get benign dataset as dataframe
        df_benign = pd.read_csv(benigndata, encoding="utf-8")
        print("[Benign] Dataset len = {}, Headers = {}".format(len(df_benign), df_benign.columns))
        ## Assign label, 0 to benign dataset
        df_benign['label'] = np.zeros(len(df_benign), dtype=np.int8)
        ## Assign label, 1 to malicious dataset
        df_mal['label'] = np.ones(len(df_mal), dtype=np.int8)
        ## Concatenate benign and malicious datasets
        df_concat = pd.concat([df_benign, df_mal], ignore_index=True, sort=False)
        print("[Benign+Mal] Dataset len = {}".format(len(df_concat)))
        ## Get features as x and labels as y
        x = df_concat['abstracted_api_info']
        y = df_concat['label']
        return x, y, df_concat

    # For malware type/family classification
    else:
        ## Sort dataframe accordingly to type/family in ascending order
        df_mal = df_mal.sort_values(by=[classtyp])
        ## Convert malware type/family name into number labels
        ## And, get sample size of each type/family
        mal_dict = dict(df_mal.pivot_table(index=[classtyp], aggfunc="size"))
        mal_size = "[{}] ".format(classtyp)

        label=0
        labels = []
        for cat in mal_dict.keys():
            mal_size+="{},{}: {} | ".format(cat, label, mal_dict[cat])
            for i in range(mal_dict[cat]):
                labels.append(label)
            label+=1
        df_mal['label'] = labels
        print(mal_size)

        ## Get features as x and labels as y
        x = df_mal['abstracted_api_info']
        y = df_mal['label']
        return x, y, df_mal

# Split datasets into different folds of train and test data and save the data
def split_datasets(x, y, cv, df_final, class_typ):
    folder_dir = "{}_datasets".format(class_typ)
    if not os.path.exists(folder_dir):
        os.mkdir(folder_dir)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=SHUFFLE_SEED)
    fold_no = 1
    for train_index, test_index in skf.split(x,y):
        # Get and save train datasets
        get_datasets(fold_no, df_final, folder_dir, train_index, class_typ, "train")
        # Get and save test datasets
        get_datasets(fold_no, df_final, folder_dir, test_index, class_typ, "test")
        fold_no += 1

# Main function
def main(args):
    ## Check if valid classification option chosen
    if args.classopt > 2:
        print("ERROR: Invalid classification option; {}! Exiting program...".format(args.classopt))
        sys.exit(0)
    classtyp = CLASS_OPT[args.classopt]
    T0 = time()
    x, y, df_final = process_datasets(args.maldata, args.benigndata, classtyp)
    split_datasets(x, y, args.cv, df_final, classtyp)
    print("Done in {}s".format(time()-T0))

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Split datasets into specified number of folds")
    args.add_argument("--maldata", type=str, required=True,
                      help="Absolute path to malicious dataset")
    args.add_argument("--benigndata", type=str, default=None,
                      help="Absolute path to benign dataset")
    args.add_argument("--cv", type=int, default=5,
                      help="Number of folds to split dataset; default: 5 folds")
    args.add_argument("--classopt", type=int, default=0,
                      help="Classification options; 0(default): maldetect, 1: maltype, 2: malfamily")
    main(args.parse_args())




