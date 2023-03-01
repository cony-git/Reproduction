import os

import lief
import numpy as np

import pandas as pd
import pickle
print(lief.PE.SECTION_CHARACTERISTICS.MEM_EXECUTE)
def extract_one(file_path):
    pointerto_raw_data_list = []
    virtual_size_list = []
    binary = lief.parse(file_path)
    has_exe_mem = False
    if binary is None:
        return None
    try:
        for section in binary.sections:
            if lief.PE.Section.has_characteristic(section,lief.PE.SECTION_CHARACTERISTICS.MEM_EXECUTE):
            #if lief.PE.SECTION_CHARACTERISTICS.MEM_EXECUTE in section.characteristics_lists:
                print(section.name,section.pointerto_raw_data,section.virtual_size,section.characteristics_lists)
                has_exe_mem = True
                pointerto_raw_data_list.append(str(section.pointerto_raw_data))
                virtual_size_list.append(str(section.virtual_size))
    except Exception as e:
        print("ERROR"+str(e))
        return None
    if has_exe_mem is False:
        return None
    return {
        'mw_file_directory':binary.name,
        'pointerto_raw_data': ','.join(pointerto_raw_data_list),
        'virtual_size': ','.join(virtual_size_list)
    }

def extract_path(dir,label_true=True):
    feature_list = []
    label_list = []
    for name in os.listdir(dir):
        file_path = os.path.join(dir,name)
        if os.path.isfile(file_path):
            ret = extract_one(file_path)
            if(ret):
                feature_list.append(ret)
                if label_true:
                    label_list.append(1)
                else:
                    label_list.append(0)

    feature_array = pd.DataFrame(feature_list)
    label_array = np.array(label_list)

    if label_true:
        with open("feature_true.pkl", 'wb') as file:
            pickle.dump(feature_array, file)
        with open("label_true.pkl", 'wb') as file:
            pickle.dump(label_array, file)
    else:
        with open("feature_false.pkl", 'wb') as file:
            pickle.dump(feature_array, file)
        with open("label_false.pkl", 'wb') as file:
            pickle.dump(label_array, file)

    #return pd.DataFrame(feature_list)

if __name__ == "__main__":
    #extract_path(r"E:\pure_win10_64\benign",label_true=True)
    #extract_path(r"E:\viruses-2010-05-18\malware-pe", label_true=False)

    # 要用时加载文件即可
    with open("feature_true.pkl", 'rb') as file:
        feature_true = pickle.load(file)
    with open("label_true.pkl", 'rb') as file:
        label_true = pickle.load(file)

    with open("feature_false.pkl", 'rb') as file:
        feature_false = pickle.load(file)
    with open("label_false.pkl", 'rb') as file:
        label_false = pickle.load(file)
    print(feature_true.loc[2])
    print(label_true[2])
    #print(extract_one(r"G:\v2rayN-Core\v2ray.exe"))

'''
 1024 356352 {<SECTION_CHARACTERISTICS.CNT_UNINITIALIZED_DATA: 128>, <SECTION_CHARACTERISTICS.MEM_EXECUTE: 536870912>, <SECTION_CHARACTERISTICS.MEM_READ: 1073741824>, <SECTION_CHARACTERISTICS.MEM_WRITE: 2147483648>}
 1024 376832 {<SECTION_CHARACTERISTICS.CNT_INITIALIZED_DATA: 64>, <SECTION_CHARACTERISTICS.MEM_EXECUTE: 536870912>, <SECTION_CHARACTERISTICS.MEM_READ: 1073741824>, <SECTION_CHARACTERISTICS.MEM_WRITE: 2147483648>}
'''