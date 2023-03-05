# Description
This folder contains supporting scripts for datasets configuration. The following will provide details of each script and guide on how to run them.

# concat_datasets.py

## Details
This script concatenates input <dataset1>* with input <dataset2>*. The script works in a step-by-step method; it will ask user whether to proceed in each step. There are a total of 2 steps; 1) whether to concatenate datasets 2) whether to save the concatenated dataset
> *Both datasets must be in the format of csv files.

### How-to-run 
		
`$ python concat_datasets.py <path to dataset1 csv file> <path to dataset2 csv file>` 

# split_datasets.py

## Details
This script split input dataset(s) (in the form of csv file) for specified folds of cross validation and save each fold of training and test sets as csv files.

## How-to-run

#### To get a list of input options
	
`$ python split_datasets.py --help` 

#### Example of splitting datasets for malware detection for 5-folds cross validation

`$ python split_datasets.py --maldata <path to malicious dataset> --benigndata <path to benign dataset> --cv 5 --classopt 0`

#### Example of splitting dataset for malware type classification for 5-folds cross validation

`$ python split_datasets.py --maldata <path to malicious dataset> --cv 5 --classopt 1`

#### Example of splitting dataset for malware family classification for 5-folds cross validation

`$ python split_datasets.py --maldata <path to malicious dataset> --cv 5 --classopt 2`
	

