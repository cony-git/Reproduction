import sys
import pandas as pd

# Check if there are any duplicate samples in dataframe, based on hash value
def check_dup(df):
    ori_len = len(df)
    new_len = len(df.drop_duplicates(subset=["sha1"]))
    if not ori_len == new_len:
        print("Ori len: {}, New len: {}, Dup: {}".format(ori_len,
                                                         new_len,
                                                         ori_len-new_len))
        return False
    else:
        return True

# Check if user agree to proceed, based on query asked
def check_proceed(reply):
    if reply.upper() == "N":
        print("User chose not to proceed; exiting program...")
        sys.exit(0)

# Main function to concatenate input1-dataset with input2-dataset
if __name__ == "__main__":
    if not len(sys.argv) == 3:
        print("ERROR: Expecting 2 inputs but got {} inputs; exiting program...".format(len(sys.argv)-1))
        sys.exit(0)

    # Get datasets as dataframes
    dataset1 = sys.argv[1]
    d1_df = pd.read_csv(dataset1, encoding="utf-8")
    dataset2 = sys.argv[2]
    d2_df = pd.read_csv(dataset2, encoding="utf-8")

    # Print out the length of each dataset
    print("Dataset 1: {} Len: {}, Dataset 2: {} Len: {}".format(dataset1,
                                                                len(d1_df),
                                                                dataset2,
                                                                len(d2_df)))

    # Check if there are any duplicates in each dataset
    check1 = check_dup(d1_df)
    check2 = check_dup(d2_df)
    if not check1 or not check2:
        print("ERROR: Duplicates found; exiting program...")
        sys.exit(0)

    # Get and check headers of datasets 1 and 2
    d1_headers = list(d1_df.columns.values)
    d2_headers = list(d2_df.columns.values)
    print("Dataset 1 header: {}\nDataset 2 header: {}".format(d1_headers,
                                                              d2_headers))

    # Remove 'filename' column if exists
    if "filename" in d1_headers:
        d1_df.drop(columns=["filename"], inplace=True)
    if "filename" in d2_headers:
        d2_df.drop(columns=["filename"], inplace=True)
    print("Proceed with concatenating both datasets? Y/N")
    check_proceed(input())

    # Concatenate the two datasets
    print("Concatenating {} with {}".format(dataset1, dataset2))
    concat_df = pd.concat([d1_df, d2_df], ignore_index=True, sort=False)

    # Check if there are any duplicates
    check_concat = check_dup(concat_df)
    if not check_concat:
        print("ERROR: Duplicates found; exiting program...")
        sys.exit(0)

    # Check the concatenated dataset
    print("Concatenated dataset:\n{}".format(concat_df))
    print("Proceed with saving concatenated dataset? Y/N")
    check_proceed(input())

    # Save the concatenated dataset
    concat_csvname = "concat_dataset_{}.csv".format(len(concat_df))
    concat_df.to_csv(concat_csvname, index=False, encoding="utf-8")
    print("Saved concatenated dataset as {}".format(concat_csvname))
    
    print("Done!")
