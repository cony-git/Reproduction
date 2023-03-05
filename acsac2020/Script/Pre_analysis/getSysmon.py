import os, shutil

start_id = 3581 # Sample ID to start collecting sysmon logs
end_id = 4500 # Last sample ID to stop collecting sysmon logs

dir_to_cp_to = "/home/dima/Desktop/JiaYi/Cuckoo_Docs/Sysmon_logs/"

def get_sample_hash(sample_id):
    cuckoo_report_dir = "/home/dima/.cuckoo/storage/analyses/{}/reports/".format(sample_id)
    if os.path.exists(cuckoo_report_dir):
        if os.listdir(cuckoo_report_dir):
            sample_hash = os.listdir(cuckoo_report_dir)[0].split(".")[0]
            return sample_hash
    return str(sample_id)

def check_if_empty(sysmon_log_file):
    with open(sysmon_log_file, "r", errors="ignore") as f:
            contents = f.read()
    if contents:
        return 1
    else:
        return 0

def get_sysmon_logs():
    no_logs = []
    no_files = []
    has_logs = []
    no_files_file = open(os.path.join(dir_to_cp_to, "no_files.txt"), "w+")
    no_logs_file = open(os.path.join(dir_to_cp_to, "no_logs.txt"), "w+")

    if not os.path.exists(dir_to_cp_to):
        os.mkdir(dir_to_cp_to)

    for sample_id in range(start_id, end_id+1):
        cuckoo_sysmon_dir = "/home/dima/.cuckoo/storage/analyses/{}/simpledima/".format(sample_id)
        
        if os.path.exists(cuckoo_sysmon_dir):
            sample_hash = get_sample_hash(sample_id)
            if os.listdir(cuckoo_sysmon_dir):
                new_folder = os.path.join(dir_to_cp_to, sample_hash)
                if not os.path.exists(new_folder):
                    os.mkdir(new_folder)
                for sysmon_logs in os.listdir(cuckoo_sysmon_dir):
                    old_file = os.path.join(cuckoo_sysmon_dir, sysmon_logs)
                    not_empty = check_if_empty(old_file)
                    if not_empty:
                        new_file = os.path.join(new_folder, sysmon_logs)
                        if not os.path.exists(new_file):
                            shutil.copy(old_file, new_file)
                        if sample_hash not in has_logs:
                            has_logs.append(sample_hash)
                if not os.listdir(new_folder):
                    os.rmdir(new_folder)
                    no_logs_file.write("{}: {}\n".format(sample_id, sample_hash))
                    if sample_hash not in no_logs:
                        no_logs.append(sample_hash)
            else:
                #print("[NO SYSMON FILE] Sample ID: {}".format(sample_id))
                no_files_file.write("{}: {}\n".format(sample_id, sample_hash))
                if sample_hash not in no_files:
                    no_files.append(sample_hash)
    no_logs_file.close()
    no_files_file.close()
    print("Number of samples without sysmon logs: {}".format(len(no_logs)))
    print("Number of samples without sysmon files: {}".format(len(no_files)))
    print("Number of samples with sysmon logs: {}".format(len(has_logs)))

if __name__ == "__main__":
    get_sysmon_logs()

