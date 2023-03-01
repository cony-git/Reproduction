import re
from math import log10
from sys import getsizeof

# Process api name in the API list to remove suffixes 
# i.e., 'ExW', 'ExA', 'W', 'A', 'Ex'
# to resove the multi-version problem of the same API call
def remove_api_suffix(api_name):
    
    suffix_to_remove = ['ExW', 'ExA', 'W', 'A', 'Ex']
    for item in suffix_to_remove:
        if re.search(item+'$', api_name):
            return re.sub(item+'$', '', api_name)
    return api_name

###
# Some utility functions to process the argument values
# For auguments with integer values, we convert then to logarithic bins
# I choose to base 10 instead of base e or 2, so as to have better stability...
def bin_int(x):
    # assert (type(x) == int)
    # sometimes the value may be of hex
    x = int(x)
    prefix = ''
    if x < 0:
        prefix = 'neg_'
    if x == 0:
        # to distinguish 0 and num in the range of (0, 10)
        # 0 is tagged as num0 instead of numB0
        tag = 'num0'
    else:
        tag = 'numB' + str(int(log10(abs(x))))
    return prefix + tag

# Common malware file types
# if any of them is found in arguments of file_path, they will be highlighted in output
# https://sensorstechforum.com/popular-windows-file-types-used-malware-2018/
common_malware_types = ['exe', 'doc', 'docx', 'docm', 'hta', 'html', 'htm', 'js', 'jar', 'vbs',\
                         'vb', 'pdf', 'sfx', 'bat', 'dll', 'tmp', 'py']


# common registry keys exploited by malware to establish persistance
# # Run includes Run and RunOnce, RunServices, RunServicesOnce           
common_registry_keyword = {'BootExecute':'BootExecute',
                            'Winlogon' : 'Winlogon',
                            r'Windows\CurrentVersion\Run' : 'Run',
                            r'SYSTEM\CurrentControlSet\Services' : 'Service',
                            r'SOFTWARE\Microsoft\Windows NT\CurrentVersion\Svchost' : 'Service',
                            r'Explorer\Browser Helper Objects' : 'BHO',
                            r'CurrentVersion\Windows\AppInit_DLLs' : 'AppInit_DLL',
                            r'CurrentVersion\Schedule\TaskCache\Tasks' : 'Scheduled_tasks',
                            r'CurrentVersion\Schedule\TaskCache\Tree' : 'Scheduled_tasks'}

def process_api_argu(api_argu_dict):
    
    processed_api = {}
    for key, value in api_argu_dict.items():
        # print(key, value)
      
        if re.search('(handle|address|parameter|pointer)', key):
            # To do - For now, no good idea of how to abstract these pointers
            # print('do nothing')
            continue
        elif re.search('_r', key):
            # to ignore filepath_r, and dirpath_r, which is duplicated with filepath & dirpath:
            # regkey_r will also be removed
            continue
        
        # elif re.search('(index|identifier)', key):
        #     # For now, think that arguments containing these two keyword provides limited information
        #     continue

        elif re.search('size', key):
            # for these parameters with concrete values, we use logarithmic buckets
            
            processed_api[key] = str(value)
        elif re.search('name|module', key):
            ##
            ## For now think that the string value under this argument is not so important
            ## Maybe later we can take advantage of it
            continue
        elif key == 'algorithm_identifier':
            processed_api[key] = str(value)
        elif key == 'command_line':
            # commonly appear in CreateProcessInternalW
            # currently I just try to extract the common malware file types from CL
            # if no common malware file types, return other_CL

            # Note: It is possible to retrieve more useful information from CL
            c_l_tag = ''
            for file_type in common_malware_types:
                pattern = '.'+file_type+'\"'
                if re.search(re.escape(pattern), value):
                    if len(c_l_tag) == 0:
                        c_l_tag += file_type
                    else:
                        c_l_tag += ('_' + file_type)
            
            if len(c_l_tag) >0:
                processed_api[key] = c_l_tag
            else:
                processed_api[key] = 'other_CL'

        elif re.search('type|library', key):
            # type argument commonly appear in FindResource API
            # Think both ok to represent it as it is
            # or simply ignore it as such API is not really malware analysis related
            #
            # 'library' argument is added for API like LdrUnloadDLL
            # libraries to be loaded

            # As network api may also have type argument, which may provide 
            # very useful information, choose to keep type and library value as it is
            # continue
            #
            processed_api[key] = str(value)
        
        elif key == 'af' or key == 'protocol':
            # specially for network related API
            # e.g., socket
            # these information are useful for my task
            processed_api[key] = str(value)

        elif re.search('path|directory', key):
            # to deal with filepath and dirpath
            # to extract file or dir location

            # added directory for API like CreateProcessInternal, in category Process
            # it has an argument of current_directory
            loc = 'sys_dir' if re.search('System32|system32', value) else 'other_dir'
            # to extract file type
            file_type = value.split('.')[-1] if re.search(r'.', value) else None
            processed_api[key] = loc
            if file_type != None:
                if file_type in common_malware_types:
                    processed_api[key] += ("_" + file_type)
        elif re.search('regkey', key):
            # to identify whether the regkey value is one of the common regkey for malware
            is_common_regkey = False
            for common_key in common_registry_keyword.keys():
                if re.search(re.escape(common_key), value):
                    processed_api[key] = common_registry_keyword[common_key]
                    is_common_regkey = True
                    break
            if is_common_regkey == False:
                processed_api[key] = 'other_reg'
        
        elif re.search('buf|string|stacktrace', key):
            # To convert the buffer, input_buffer, output_buffer arguments to firstly length
            # and then log10 based bins
            # stacktrace argument is added for __exception__api
            # as think that this api may probably add useful information for my task

            #changed buffer keyword to buf, as 'buf' is an argument for 
            # SetInformationJobObject API
            processed_api[key] = str(len(value))
        
        elif key == 'registers' or key == 'exception':
            # these two arguments are specifically for __exception__ api
            # in the form of dictionary, think that they are of limited use for my task
            continue

        elif key == 'desired_access':
            # access value is a 32-bit number
            # for now, just keep it as it is
            # Found some value is represented in decimal integer instead of hex in string
            # added hex()
            processed_api[key] = str(hex(value)) if type(value) == int else value
        elif key == 'share_access':
            # the value is typically a small integer, just keep as it is
            processed_api[key] = str(value)
        
        elif type(value) == int:
            # For remaining arguments with integer values, convert them into 
            # bin-based tags
            processed_api[key] = str(value)
        elif key == 'value':
            # if the argument is 'value' and the value of 'value' is not integer
            # represent it with the length

            processed_api[key] = str(len(value))
        elif key == 'clsid' or key == 'iid':
            # these two arguments were found for CoCreateInstance API
            # Cuckoo indicates that sizeof(clsid) and sizeof(iid) are interesting
            # decide to use such size first and then convert the size to log10 bins
            processed_api[key] = str(getsizeof(value))

        # Choose to omit all the remaining argument types
        # Remember that it is possible to make the list above longer 
        # to include more information 
        #else:#Damdoom dont forget to omit the followinf 5 sept 2019
        #     # keep as it is for remaining arguments
        #     # Use it to confirm any missing necessary processing  
             #processed_api[key] = value
    return processed_api

def abstract_api_VD(org_api):
    # org_api is a dictionary
    processed_api = []
    api_name = org_api['api']
    # processed_api['name'] = api_name
    # remove suffix like ExW, ExA, W, A, Ex
    api_name = remove_api_suffix(api_name)

    try: 
        api_arguments = process_api_argu(org_api['arguments']) # again, a dictionary
    except:
        return "ERROR"
    #print(api_name)
    #print(api_arguments)
    arg_idx = 0
    # we use sorted() to remove the ambiguous order of dict keys
    for key in sorted(api_arguments):
        api_item = api_name + ":" + str(arg_idx)+ "=" + (api_arguments[key])
        processed_api.append(api_item)
        arg_idx += 1
    # processed_api['arguments'] = api_arguments
    return tuple(processed_api)
