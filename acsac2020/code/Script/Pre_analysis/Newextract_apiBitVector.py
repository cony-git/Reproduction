import datetime 
import os
import csv
import json
import newutils

csvfile = "mal_features.csv" #Filename of csv file to contain the features
where_api_dir = "/home/dima/Desktop/Cuckoo_Docs/Json_reports/" # Directory path to copied cuckoo reports

def obtain_abstracted_api(target_sample, ListOfWrongFiles):
    LIST = ListOfWrongFiles
    current_row = []
    target_json = target_sample
    try:
        with open(target_json) as read_file:
            data = json.load(read_file)
    # Error opening json file
    except:
        LIST = target_json + ';'
        return None, LIST
    sample_sha1 = data['target']['file']['sha1']
    sample_type = data['target']['file']['type'].split(' ')[0]
    print('sample_sha: ', sample_sha1)
    print('sample_type: ', sample_type)

    # For now, target only samples of type 'PE32', 'PE32+' and 'ASCII'
    if (sample_type != 'PE32+' and sample_type != 'PE32' and 'ASCII' not in sample_type):
        LIST = target_json + ';'
        return None, LIST

    current_row.append(target_json) #filename of sample
    current_row.append(sample_sha1) #sha1 hash of sample
    current_row.append(sample_type) #type of sample

    # Some statistics for about API call sequences in the report
    print ("# of processes: ", len(data['behavior']['processes']))

    # Length of API in each process
    api_set = set()
    total_api_seq_len = 0
    for i in range(len(data['behavior']['processes'])):
        current_api_list = data['behavior']['processes'][i]['calls']

        len_api_seq = len(current_api_list)
        print ("# of invoked API calls: ", len_api_seq)
        total_api_seq_len += len_api_seq
        #print(current_api_list)
        for api in current_api_list:
            abstracted_api = newutils.abstract_api_VD(api)
            if abstracted_api == "ERROR":
                LIST = target_json + ";"
                return None, LIST
            # there may be some empty tuple due to possessing only not meaningful arguments
            # Change the abstracted_api in tuple to a string
            abstracted_api = ';'.join(abstracted_api)
            if len(abstracted_api) > 0:
                api_set.add(abstracted_api)

    current_row.append(total_api_seq_len)

    len_sample_set = len(api_set)
    print("length of api_set: ",len_sample_set)
    current_row.append(len_sample_set)

    current_row.append(' '.join(api_set))
    
    if len_sample_set < 1:
        LIST = target_json + ';'
        return None, LIST

    return current_row, LIST

if __name__ == "__main__":
    WrongFiles = ''
    Wrong = ''
    ListWrong = 'ListWrong: '
    fNameERROR = 'fNameERROR:  '

    # Check if csv file exists
    if os.path.exists(csvfile):
        file_CSV = open(csvfile, "a")
        writer = csv.writer(file_CSV)
    else:
        file_CSV = open(csvfile, "w+")
        writer = csv.writer(file_CSV)
        # Headers of csv file
        #writer.writerow(["sha1", "type", "seq_length", "set_length", "abstracted_api_info"])
        writer.writerow(["filename", "sha1", "type", "seq_length", "set_length", "abstracted_api_info"])


    for report in os.listdir(where_api_dir):
        if report.endswith(".json"):
            report_path = os.path.join(where_api_dir, report)
            print(report_path)
            try:
                current_row, Wrong = obtain_abstracted_api(report_path, WrongFiles)
                ListWrong = ListWrong + Wrong
                if current_row != None:
                    writer.writerow(current_row)
            except IOError:
                fNameERROR = fNameERROR + report_path
                print("Could not read file:", fNameERROR)

    file_CSV.close()
    print(ListWrong)

    listwrong_filename = "listwrong_{}_{}.txt".format(len(ListWrong.split(";")),
                                                      datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    with open(listwrong_filename, "w+") as txtfile:
        for w in ListWrong.split(";"):
            txtfile.write("{}\n".format(w)) 
    print(fNameERROR)
