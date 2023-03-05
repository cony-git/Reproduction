#This code will check the following before copying report.json
## 1) if there is report.json
## 2) if report.json has API calls and their arguments
import os
import shutil

START_ID = 3581 # Sample ID to start collecting report.json 
END_ID = 4500 # Last sample ID to stop collecting report.json(+1)

cuckoo_analyses_dir = "/home/dima/.cuckoo/storage/analyses/"  #dir to cuckoo analyses folder
#exe_dir = "/home/dima/Desktop/executables_16092020/"          #dir to the analysed executables
dir_to_cp_to = "/home/dima/Desktop/Cuckoo_Docs/Json_reports/" #dir to copy cuckoo report to 

def check_num_calls(report_file):
    with open(report_file,"r") as f:
        num_of_calls=0
        for line in f:
            words = line.split()
            for word in words:
                if(word=='\"calls\":'): #"calls": [
                    num_of_calls+=1

    return num_of_calls

def get_report():
    HasReport = []
    NoApi = 0
    NoApi_file = open(os.path.join(dir_to_cp_to, "without_api.txt"), "w+")

    for sample_id in range(START_ID, END_ID+1):
        report_folder = os.path.join(cuckoo_analyses_dir, "{}/reports/".format(sample_id))
        if os.path.exists(report_folder):
            if os.listdir(report_folder):
                report_file = os.listdir(report_folder)[0]
                #HasReport.append(report_file.replace(".json", ".exe"))
                HasReport.append(report_file.replace(".json", ""))
                report_filename = os.path.join(report_folder, report_file)

                # Check for calls in report
                calls_no = check_num_calls(report_filename)

                # Based on observation:
                ##If num of calls in the file is >=2, there are api calls & arguments
                if calls_no>=2:
                    if ".json" not in report_file:
                        report_file = "{}.json".format(report_file)
                    new_filename = os.path.join(dir_to_cp_to, report_file)
                    shutil.copy(report_filename, new_filename)
                    print("{} is copied as {}".format(report_filename, new_filename))
                
                else:
                    print("[NO API] Sample ID: {}, Sample Name: {}, Calls No: {}".format(sample_id,
                                                                                         report_file.split(".")[0],
                                                                                         calls_no))
                    NoApi_file.write("{}: {}\n".format(sample_id,
                                                       report_file.split(".")[0]))
                    NoApi+=1
            # report.json does not exist         
            else:
                print("[NO REPORT] Sample ID: {}".format(sample_id))
    NoApi_file.close()
    print("Number of files with 0 or 1 API calls : {}".format(NoApi))

    return HasReport

if __name__ == "__main__":
    samples_with_reports = get_report()
    # To check report with the analysed executables
    '''
    samples_dir = exe_dir
    total_samples = []
    for root, dirs, files in os.walk(samples_dir):
        for f in files:
            total_samples.append(f)
    print("Number of files without reports: {}".format(len(total_samples)-len(samples_with_reports)))
    NoReport_file = open(os.path.join(dir_to_cp_to, "without_report.txt"), "w+")
    for sample in total_samples:
        if sample not in samples_with_reports:
            NoReport_file.write(str(sample.split(".")[0])+"\n")
    NoReport_file.close()    
    for sample in samples_with_reports:
        if sample not in total_samples:
            print("{} does not belong in folder {}".format(sample, samples_dir))'''
    
