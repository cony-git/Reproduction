# Description
This folder contains supporting scripts for feature preprocessing. The following will provide details of each script and guide on how to run them.

# getReport.py

## Details
This script will check 1)if there is cuckoo report and 2)if cuckoo report has >=2 api calls before copying the report to another folder for feature extraction.
Configuration is to be done in the script (placed at the top) before running. 

### Configuration options
- START_ID 		        : Cuckoo report ID to start copying
- END_ID   		        : Cuckoo report ID to stop copying
- cuckoo_analyses_dir	: Directory to cuckoo analyses folder
- dir_to_cp_to		    : Directory to copy the cuckoo report to

### How-to-run 
		
`$ python getReport.py` 
> Only run after configuration is done! 

# getSysmon.py

## Details
This script will check 1)if there is sysmon logs 2)if sysmon logs are not empty before copying the logs to another folder.
Configuration is to be done in the script (placed at the top) before running. 

### Configuration Options
- START_ID 		        : Cuckoo report ID to start copying 
- END_ID   		        : Cuckoo report ID to stop copying 
- cuckoo_analyses_dir	: Directory to cuckoo analyses folder
- dir_to_cp_to		    : Directory to copy the cuckoo report to

## How-to-run

`$ python getSysmon.py`
> Only run after configuration is done!

# Newextract_apiBitVector.py

## Details
This script will check if sample type of the report is PE32, PE32+ or ASCII before doing feature extraction and saving the extracted features in a csv file. 
Configuration is to be done in the script (placed at the top) before running.

### Configuration options
- csvfile               : Name of csv file to store and save features 
- where_api_dir         : Directory to copied cuckoo report to do feature extraction

### How-to-run 
		
`$ python Newextract_apiBitVector.py`
> Only run after configuration is done! 

# newutils.py

## Details
This script does the feature extraction; `Newextract_apiBitVector.py` will run this script to do feature extraction.
> This script is to be placed with Newextract_apiBitVector.py in the same folder!