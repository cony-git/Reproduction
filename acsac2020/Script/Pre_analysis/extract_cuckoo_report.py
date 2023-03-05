from asyncio.windows_events import NULL
import json
import numpy as np
import re
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import FeatureHasher

# TODO 判断这个文件有没有运行成功

def read_json(file):
    data = None
    with open(file) as data_file:
        data = json.load(data_file)
    return data


class CuckooFeatureType(object):
    ''' Base class from which each feature type may inherit '''

    name = ''
    dim = 0

    def __repr__(self):
        return '{}({})'.format(self.name, self.dim)

    def raw_features(self, json_data):
        ''' Generate a JSON-able representation of the file '''
        raise (NotImplemented)

    def process_raw_features(self, raw_obj):
        ''' Generate a feature vector from the raw features '''
        raise (NotImplemented)

    def feature_vector(self, json_data):
        ''' Directly calculate the feature vector from the sample itself. This should only be implemented differently
        if there are significant speedups to be gained from combining the two functions. '''
        return self.process_raw_features(self.raw_features(json_data))


class BasicInfo(CuckooFeatureType):
    # TODO 进程树深度？
    name = 'info'
    dim = 3

    def __init__(self):
        super(CuckooFeatureType, self).__init__()

    def raw_features(self, json_data):
        if json_data is None:
            return {
                'duration': 0,
                'score': 0,
                'dropped': 0,
            }

        return {
            'duration': json_data['info']['duration'],
            'score': json_data['info']['score'],
            'dropped': len(json_data['dropped']) if 'dropped' in json_data else 0,
        }

    def process_raw_features(self, raw_obj):
        return np.asarray(
            [
                raw_obj['duration'], raw_obj['score'], raw_obj['dropped'],
            ],
            dtype=np.float32)


class NetworkInfo(CuckooFeatureType):
    name = 'Network'
    dim = 36

    def __init__(self):
        super(CuckooFeatureType, self).__init__()

    def raw_features(self, json_data):
        if json_data is None or 'network' not in json_data:
            return {
                'domains': [],
                'tls': [],
                'irc': [],
                'udp': [],
                'dns_servers': [],
                # 解析一下每个记录类型的总数
                'dns': [],
                'http': [],
                'http_ex': [],
                'icmp': [],
                'smtp': [],
                'smtp_ex': [],
                'hosts': [],
            }

        return {
            'domains': json_data['network']['domains'],
            'tls': json_data['network']['tls'],
            'irc': json_data['network']['irc'],
            'udp': json_data['network']['udp'],
            'dns_servers': json_data['network']['dns_servers'],
            'dns': json_data['network']['dns'],
            'http': json_data['network']['http'],
            'http_ex': json_data['network']['http_ex'],
            'icmp': json_data['network']['icmp'],
            'smtp': json_data['network']['smtp'],
            'smtp_ex': json_data['network']['smtp_ex'],
            'hosts': json_data['network']['hosts'],
        }

    def process_raw_features(self, raw_obj):
        dns_a, dns_aaaa, dns_ns, dns_cname = 0, 0, 0, 0
        if raw_obj['dns']:
            for dns_cap in raw_obj['dns']:
                if dns_cap['type'] == 'A':
                    dns_a += 1
                elif dns_cap['type'] == 'AAAA':
                    dns_aaaa += 1
                elif dns_cap['type'] == 'NS':
                    dns_ns += 1
                elif dns_cap['type'] == 'NS':
                    dns_cname += 1
        # 提取域名和IP字符串转为FeatureHasher
        domians_ip = [each['ip'] if 'ip' in each else '' for each in raw_obj['domains']]
        domians_domain = [each['domain'] if 'domain' in each else '' for each in raw_obj['domains']]
        ip_hashed = FeatureHasher(10, input_type="string").transform([domians_ip]).toarray()[0]
        domain_hashed = FeatureHasher(10, input_type="string").transform([domians_domain]).toarray()[0]
        return np.hstack(
            [
                len(raw_obj['domains']), len(raw_obj['tls']), len(raw_obj['irc']), len(raw_obj['udp']),
                len(raw_obj['dns_servers']),
                len(raw_obj['dns']), dns_a, dns_aaaa, dns_ns, dns_cname, len(raw_obj['http']), len(raw_obj['http_ex']),
                len(raw_obj['icmp']), len(raw_obj['smtp']), len(raw_obj['smtp_ex']), len(raw_obj['hosts']),
                ip_hashed,
                domain_hashed
            ]).astype(np.float32)


class SignatureInfo(CuckooFeatureType):
    name = 'Signature'
    # TODO 提取签名信息，这块不知道怎样提取较好
    dim = 0

    def __init__(self):
        super(CuckooFeatureType, self).__init__()

    def raw_features(self, json_data):
        if json_data is None or 'signature' not in json_data:
            return []
        return []

    def process_raw_features(self, raw_obj):
        return None


class ProcMemoryInfo(CuckooFeatureType):
    # TODO 有待挖掘，但很多文件没有这个字段

    name = 'procmemory'
    dim = 0

    def __init__(self):
        super(CuckooFeatureType, self).__init__()

    def raw_features(self, json_data):
        if json_data is None or 'procmemory' not in json_data:
            return {

            }

        return {

        }

    def process_raw_features(self, raw_obj):
        return None


class BehaviorGenericInfo(CuckooFeatureType):
    name = 'BehaviorGeneric'
    dim = 60

    def __init__(self):
        super(CuckooFeatureType, self).__init__()

    def raw_features(self, json_data):
        summary_feature_list = ['directory_created', 'directory_removed', 'directory_enumerated', 'dll_loaded',
                                'file_created', 'file_recreated', 'file_opened', 'file_read', 'file_written',
                                'file_exists', 'file_failed',
                                'file_deleted', 'file_moved', 'file_copied',
                                'regkey_opened', 'regkey_read', 'regkey_written', 'regkey_deleted',
                                'connects_host', 'fetches_url', 'resolves_host', 'connects_ip',
                                'command_line', 'mutex', 'guid', 'wmi_query']
        generic_feature_list = ['pid', 'ppid', 'process_name', 'process_path']
        raw_obj = {
            key: [] for key in summary_feature_list
        }
        raw_obj.update({key2: [] for key2 in generic_feature_list})
        if json_data is None or 'behavior' not in json_data or 'generic' not in json_data['behavior']:
            return raw_obj
        behavior_generic = json_data['behavior']['generic']
        for proc in behavior_generic:
            for key in summary_feature_list:
                raw_obj[key].extend(proc['summary'].get(key, []))
            for key in generic_feature_list:
                raw_obj[key].append(proc.get(key, ))
        return raw_obj

    def process_raw_features(self, raw_obj):

        proc_name_pid = [(raw_obj['process_name'][i], raw_obj['pid'][i]) for i in range(len(raw_obj['pid']))]
        pid_str = [str(pid) for pid in raw_obj['pid']]
        raw_obj['pid'].sort()  # 对pid列表排下序
        # 这些特征与API的统计特征很多都重复了，有没有必要保留？
        return np.hstack(
            [
                len(raw_obj['directory_created']), len(raw_obj['directory_removed']),
                len(raw_obj['directory_enumerated']),
                len(raw_obj['dll_loaded']), len(raw_obj['file_created']), len(raw_obj['file_recreated']),
                len(raw_obj['file_opened']), len(raw_obj['file_read']), len(raw_obj['file_written']),
                len(raw_obj['file_exists']), len(raw_obj['file_failed']), len(raw_obj['file_deleted']),
                len(raw_obj['file_moved']), len(raw_obj['file_copied']), len(raw_obj['regkey_opened']),
                len(raw_obj['regkey_read']), len(raw_obj['regkey_written']), len(raw_obj['regkey_deleted']),
                len(raw_obj['connects_host']), len(raw_obj['fetches_url']), len(raw_obj['resolves_host']),
                len(raw_obj['connects_ip']), len(raw_obj['command_line']), len(raw_obj['mutex']), len(raw_obj['guid']),
                len(raw_obj['wmi_query']),
                len(raw_obj['pid']),

                #FeatureHasher(10, input_type="pair").transform([proc_name_pid]).toarray()[0],
                FeatureHasher(10, input_type="string").transform([pid_str]).toarray()[0],
                FeatureHasher(10, input_type="string").transform([raw_obj['mutex']]).toarray()[0],
                FeatureHasher(10, input_type="string").transform([raw_obj['guid']]).toarray()[0],
                sum(raw_obj['pid']) / len(raw_obj['pid']) if len(raw_obj['pid'])!=0 else 0,  # mean
                # TODO 运行进程不成功raw_obj['pid']长度为0，下面这两句会报错，最后返回None
                raw_obj['pid'][0],  # min
                raw_obj['pid'][-1],  # max
            ]).astype(np.float32)



class ApiStatsGenericInfo(CuckooFeatureType):
    name = 'ApiStatsGeneric'
    dim = 2
    # TODO 在黑样本训练集中找到调用前20的API再统计次数
    # top20api=[]
    def __init__(self):
        super(CuckooFeatureType, self).__init__()
    def raw_features(self, json_data):
        if json_data is None or 'behavior' not in json_data or 'apistats' not in json_data['behavior']:
            return {}
        return json_data['behavior']['apistats']

    def process_raw_features(self, raw_obj):
        api_dict = {}
        api_cnt = 0
        for pid, proc_api_dict in raw_obj.items():  # 遍历外层，有可能有多个进程
            # print(len(proc_api_dict),pid,proc_api_dict)
            for each_api, cnt in proc_api_dict.items():
                api_cnt += cnt
                if (each_api in api_dict):
                    api_dict[each_api] += cnt
                else:
                    api_dict[each_api] = cnt

        return np.asarray(
            [
                len(api_dict), #api distinct
                api_cnt,# api count
            ],
            dtype=np.float32)

class ApiCallStatsInfo(CuckooFeatureType):
    # tid统计特征，api类别特征，返回值是0的总数和比例，有多少不同的返回值。线程调用的 最多/最少/平均 api数目？
    name = 'ApiCallStats'
    dim = 351

    def __init__(self):
        super(CuckooFeatureType, self).__init__()
    def raw_features(self, json_data):
        api_category_dict = {'__notification__': 0, 'certificate': 0, 'crypto': 0, 'exception': 0, 'file': 0,
                             'iexplore': 0, 'misc': 0, 'netapi': 0, 'network': 0, 'ole': 0,
                             'process': 0, 'registry': 0, 'resource': 0, 'services': 0, 'synchronisation': 0,
                             'system': 0, 'ui': 0, }
        api_dict = {'__process__' : 0, '__anomaly__' : 0, '__exception__' : 0, '__missing__' : 0, 'CertControlStore' : 0, 'CertCreateCertificateContext' : 0, 'CertOpenStore' : 0, 'CertOpenSystemStoreA' : 0, 'CertOpenSystemStoreW' : 0, 'CryptAcquireContextA' : 0, 'CryptAcquireContextW' : 0, 'CryptCreateHash' : 0, 'CryptDecrypt' : 0, 'CryptEncrypt' : 0, 'CryptExportKey' : 0, 'CryptGenKey' : 0, 'CryptHashData' : 0, 'CryptDecodeMessage' : 0, 'CryptDecodeObjectEx' : 0, 'CryptDecryptMessage' : 0, 'CryptEncryptMessage' : 0, 'CryptHashMessage' : 0, 'CryptProtectData' : 0, 'CryptProtectMemory' : 0, 'CryptUnprotectData' : 0, 'CryptUnprotectMemory' : 0, 'PRF' : 0, 'Ssl3GenerateKeyMaterial' : 0, 'SetUnhandledExceptionFilter' : 0, 'RtlAddVectoredContinueHandler' : 0, 'RtlAddVectoredExceptionHandler' : 0, 'RtlDispatchException' : 0, 'RtlRemoveVectoredContinueHandler' : 0, 'RtlRemoveVectoredExceptionHandler' : 0, 'CopyFileA' : 0, 'CopyFileExW' : 0, 'CopyFileW' : 0, 'CreateDirectoryExW' : 0, 'CreateDirectoryW' : 0, 'DeleteFileW' : 0, 'DeviceIoControl' : 0, 'FindFirstFileExA' : 0, 'FindFirstFileExW' : 0, 'GetFileAttributesExW' : 0, 'GetFileAttributesW' : 0, 'GetFileInformationByHandle' : 0, 'GetFileInformationByHandleEx' : 0, 'GetFileSize' : 0, 'GetFileSizeEx' : 0, 'GetFileType' : 0, 'GetShortPathNameW' : 0, 'GetSystemDirectoryA' : 0, 'GetSystemDirectoryW' : 0, 'GetSystemWindowsDirectoryA' : 0, 'GetSystemWindowsDirectoryW' : 0, 'GetTempPathW' : 0, 'GetVolumeNameForVolumeMountPointW' : 0, 'GetVolumePathNameW' : 0, 'GetVolumePathNamesForVolumeNameW' : 0, 'MoveFileWithProgressW' : 0, 'RemoveDirectoryA' : 0, 'RemoveDirectoryW' : 0, 'SearchPathW' : 0, 'SetEndOfFile' : 0, 'SetFileAttributesW' : 0, 'SetFileInformationByHandle' : 0, 'SetFilePointer' : 0, 'SetFilePointerEx' : 0, 'NtCreateDirectoryObject' : 0, 'NtCreateFile' : 0, 'NtDeleteFile' : 0, 'NtDeviceIoControlFile' : 0, 'NtOpenDirectoryObject' : 0, 'NtOpenFile' : 0, 'NtQueryAttributesFile' : 0, 'NtQueryDirectoryFile' : 0, 'NtQueryFullAttributesFile' : 0, 'NtQueryInformationFile' : 0, 'NtReadFile' : 0, 'NtSetInformationFile' : 0, 'NtWriteFile' : 0, 'COleScript_Compile' : 0, 'CDocument_write' : 0, 'CElement_put_innerHTML' : 0, 'CHyperlink_SetUrlComponent' : 0, 'CIFrameElement_CreateElement' : 0, 'CScriptElement_put_src' : 0, 'CWindow_AddTimeoutCode' : 0, 'GetUserNameA' : 0, 'GetUserNameW' : 0, 'LookupAccountSidW' : 0, 'GetComputerNameA' : 0, 'GetComputerNameW' : 0, 'GetDiskFreeSpaceExW' : 0, 'GetDiskFreeSpaceW' : 0, 'GetTimeZoneInformation' : 0, 'WriteConsoleA' : 0, 'WriteConsoleW' : 0, 'CoInitializeSecurity' : 0, 'UuidCreate' : 0, 'GetUserNameExA' : 0, 'GetUserNameExW' : 0, 'ReadCabinetState' : 0, 'SHGetFolderPathW' : 0, 'SHGetSpecialFolderLocation' : 0, 'EnumWindows' : 0, 'GetCursorPos' : 0, 'GetSystemMetrics' : 0, 'NetGetJoinInformation' : 0, 'NetShareEnum' : 0, 'NetUserGetInfo' : 0, 'NetUserGetLocalGroups' : 0, 'NetUserGetLocalGroups' : 0, 'NetShareEnum' : 0, 'DnsQuery_A' : 0, 'DnsQuery_UTF8' : 0, 'DnsQuery_W' : 0, 'GetAdaptersAddresses' : 0, 'GetAdaptersInfo' : 0, 'GetBestInterfaceEx' : 0, 'GetInterfaceInfo' : 0, 'ObtainUserAgentString' : 0, 'URLDownloadToFileW' : 0, 'DeleteUrlCacheEntryA' : 0, 'DeleteUrlCacheEntryW' : 0, 'HttpOpenRequestA' : 0, 'HttpOpenRequestW' : 0, 'HttpQueryInfoA' : 0, 'HttpSendRequestA' : 0, 'HttpSendRequestW' : 0, 'InternetCloseHandle' : 0, 'InternetConnectA' : 0, 'InternetConnectW' : 0, 'InternetCrackUrlA' : 0, 'InternetCrackUrlW' : 0, 'InternetGetConnectedState' : 0, 'InternetGetConnectedStateExA' : 0, 'InternetGetConnectedStateExW' : 0, 'InternetOpenA' : 0, 'InternetOpenUrlA' : 0, 'InternetOpenUrlW' : 0, 'InternetOpenW' : 0, 'InternetQueryOptionA' : 0, 'InternetReadFile' : 0, 'InternetSetOptionA' : 0, 'InternetSetStatusCallback' : 0, 'InternetWriteFile' : 0, 'ConnectEx' : 0, 'GetAddrInfoW' : 0, 'TransmitFile' : 0, 'WSAAccept' : 0, 'WSAConnect' : 0, 'WSARecv' : 0, 'WSARecvFrom' : 0, 'WSASend' : 0, 'WSASendTo' : 0, 'WSASocketA' : 0, 'WSASocketW' : 0, 'WSAStartup' : 0, 'accept' : 0, 'bind' : 0, 'closesocket' : 0, 'connect' : 0, 'getaddrinfo' : 0, 'gethostbyname' : 0, 'getsockname' : 0, 'ioctlsocket' : 0, 'listen' : 0, 'recv' : 0, 'recvfrom' : 0, 'select' : 0, 'send' : 0, 'sendto' : 0, 'setsockopt' : 0, 'shutdown' : 0, 'socket' : 0, 'CoCreateInstance' : 0, 'CoInitializeEx' : 0, 'OleInitialize' : 0, 'CreateProcessInternalW' : 0, 'CreateRemoteThread' : 0, 'CreateThread' : 0, 'CreateToolhelp32Snapshot' : 0, 'Module32FirstW' : 0, 'Module32NextW' : 0, 'Process32FirstW' : 0, 'Process32NextW' : 0, 'ReadProcessMemory' : 0, 'Thread32First' : 0, 'Thread32Next' : 0, 'WriteProcessMemory' : 0, 'system' : 0, 'NtAllocateVirtualMemory' : 0, 'NtCreateProcess' : 0, 'NtCreateProcessEx' : 0, 'NtCreateSection' : 0, 'NtCreateThread' : 0, 'NtCreateThreadEx' : 0, 'NtCreateUserProcess' : 0, 'NtFreeVirtualMemory' : 0, 'NtGetContextThread' : 0, 'NtMakePermanentObject' : 0, 'NtMakeTemporaryObject' : 0, 'NtMapViewOfSection' : 0, 'NtOpenProcess' : 0, 'NtOpenSection' : 0, 'NtOpenThread' : 0, 'NtProtectVirtualMemory' : 0, 'NtQueueApcThread' : 0, 'NtReadVirtualMemory' : 0, 'NtResumeThread' : 0, 'NtSetContextThread' : 0, 'NtSuspendThread' : 0, 'NtTerminateProcess' : 0, 'NtTerminateThread' : 0, 'NtUnmapViewOfSection' : 0, 'NtWriteVirtualMemory' : 0, 'RtlCreateUserProcess' : 0, 'RtlCreateUserThread' : 0, 'ShellExecuteExW' : 0, 'RegCloseKey' : 0, 'RegCreateKeyExA' : 0, 'RegCreateKeyExW' : 0, 'RegDeleteKeyA' : 0, 'RegDeleteKeyW' : 0, 'RegDeleteValueA' : 0, 'RegDeleteValueW' : 0, 'RegEnumKeyExA' : 0, 'RegEnumKeyExW' : 0, 'RegEnumKeyW' : 0, 'RegEnumValueA' : 0, 'RegEnumValueW' : 0, 'RegOpenKeyExA' : 0, 'RegOpenKeyExW' : 0, 'RegQueryInfoKeyA' : 0, 'RegQueryInfoKeyW' : 0, 'RegQueryValueExA' : 0, 'RegQueryValueExW' : 0, 'RegSetValueExA' : 0, 'RegSetValueExW' : 0, 'NtCreateKey' : 0, 'NtDeleteKey' : 0, 'NtDeleteValueKey' : 0, 'NtEnumerateKey' : 0, 'NtEnumerateValueKey' : 0, 'NtLoadKey' : 0, 'NtLoadKey2' : 0, 'NtLoadKeyEx' : 0, 'NtOpenKey' : 0, 'NtOpenKeyEx' : 0, 'NtQueryKey' : 0, 'NtQueryMultipleValueKey' : 0, 'NtQueryValueKey' : 0, 'NtRenameKey' : 0, 'NtReplaceKey' : 0, 'NtSaveKey' : 0, 'NtSaveKeyEx' : 0, 'NtSetValueKey' : 0, 'FindResourceA' : 0, 'FindResourceExA' : 0, 'FindResourceExW' : 0, 'FindResourceW' : 0, 'LoadResource' : 0, 'SizeofResource' : 0, 'ControlService' : 0, 'CreateServiceA' : 0, 'CreateServiceW' : 0, 'DeleteService' : 0, 'EnumServicesStatusA' : 0, 'EnumServicesStatusW' : 0, 'OpenSCManagerA' : 0, 'OpenSCManagerW' : 0, 'OpenServiceA' : 0, 'OpenServiceW' : 0, 'StartServiceA' : 0, 'StartServiceW' : 0, 'GetLocalTime' : 0, 'GetSystemTime' : 0, 'GetSystemTimeAsFileTime' : 0, 'GetTickCount' : 0, 'NtCreateMutant' : 0, 'NtDelayExecution' : 0, 'NtQuerySystemTime' : 0, 'timeGetTime' : 0, 'LookupPrivilegeValueW' : 0, 'GetNativeSystemInfo' : 0, 'GetSystemInfo' : 0, 'IsDebuggerPresent' : 0, 'OutputDebugStringA' : 0, 'SetErrorMode' : 0, 'LdrGetDllHandle' : 0, 'LdrGetProcedureAddress' : 0, 'LdrLoadDll' : 0, 'LdrUnloadDll' : 0, 'NtClose' : 0, 'NtDuplicateObject' : 0, 'NtLoadDriver' : 0, 'NtUnloadDriver' : 0, 'RtlCompressBuffer' : 0, 'RtlDecompressBuffer' : 0, 'RtlDecompressFragment' : 0, 'ExitWindowsEx' : 0, 'GetAsyncKeyState' : 0, 'GetKeyState' : 0, 'GetKeyboardState' : 0, 'SendNotifyMessageA' : 0, 'SendNotifyMessageW' : 0, 'SetWindowsHookExA' : 0, 'SetWindowsHookExW' : 0, 'UnhookWindowsHookEx' : 0, 'DrawTextExA' : 0, 'DrawTextExW' : 0, 'FindWindowA' : 0, 'FindWindowExA' : 0, 'FindWindowExW' : 0, 'FindWindowW' : 0, 'GetForegroundWindow' : 0, 'LoadStringA' : 0, 'LoadStringW' : 0, 'MessageBoxTimeoutA' : 0, 'MessageBoxTimeoutW' : 0}
        # 键是pid，值是pid对应的api数量
        tid_list={}
        ret_value_dict={}
        if json_data is None or 'behavior' not in json_data or 'processes' not in json_data['behavior']:
            return {'api_category_dict':api_category_dict,
                    'api_dict': api_dict,
                    'tid_list':tid_list,
                    'ret_value_dict': ret_value_dict}

        data = json_data['behavior']['processes']


        for i,each_thread in enumerate(data):
            tid_list[each_thread['tid']] = len(each_thread['calls'])
            for each_call in each_thread['calls']:
                if (each_call['category'] in api_category_dict):
                    api_category_dict[each_call['category']] += 1
                # else:
                #     api_category_dict[each_call['category']] = 1

                if (each_call['api'] in api_dict):
                    api_dict[each_call['api']] += 1
                # else:
                #     api_dict[each_call['api']] = 1

                if each_call['return_value'] in ret_value_dict:
                    ret_value_dict[each_call['return_value']] += 1
                else:
                    ret_value_dict[each_call['return_value']] = 1
        # print(api_category_dict)
        return {'api_category_dict':api_category_dict,
                'api_dict': api_dict,
                    'tid_list':tid_list,
                    'ret_value_dict': ret_value_dict}


    def process_raw_features(self, raw_obj):
        # TODO 忽略掉第一个lsass.exe进程
        return np.hstack([
            # 线程统计特征
            len(raw_obj['tid_list']),
            np.mean(list(raw_obj['tid_list'].keys())),
            max(list(raw_obj['tid_list'].keys())),
            min(list(raw_obj['tid_list'].keys())),
            # 返回值是0的总数和比例，有多少不同的返回值
            raw_obj['ret_value_dict'].get(0,0),#取返回值是0的个数，
            raw_obj['ret_value_dict'].get(0,0)/sum(raw_obj['ret_value_dict'].values()),
            len(raw_obj['ret_value_dict']),
            # 线程调用的 最多/最少/平均 api数目
            np.mean(list(raw_obj['tid_list'].values())),
            max(list(raw_obj['tid_list'].values())),
            min(list(raw_obj['tid_list'].values())),

            list(raw_obj['api_category_dict'].values()),
            list(raw_obj['api_dict'].values()),
        ]).astype(np.float32)

# 这个需要对整个数据集一并操作
class ApiTfidfStatsInfo(CuckooFeatureType):
    # tfidf
    name = 'ApiTfidf'
    dim = 0

    def __init__(self):
        super(CuckooFeatureType, self).__init__()
    def raw_features(self, json_data):
        call_list=[]
        if json_data is None or 'behavior' not in json_data or 'processes' not in json_data['behavior']:
            return []
        count=0
        data = json_data['behavior']['processes']
        for i,each_thread in enumerate(data):
            for each_call in each_thread['calls']:
                if count<100:
                    call_list.append(each_call['api'])
                    count+=1
                else:
                    break
        if count<100:
            call_list+=[NULL]*(100-count)

        # return " ".join(w for w in call_list)
        return call_list


    def process_raw_features(self, raw_obj):
        # vectorizer = TfidfVectorizer(ngram_range=(1, 5), min_df=3, max_df=0.9)
        # tfidf_features = vectorizer.fit_transform(raw_obj)
        return raw_obj


class CuckooExtractor(object):
    ''' Extract useful features from a PE file, and return as a vector of fixed size. '''

    features = [
        BasicInfo(),NetworkInfo(),BehaviorGenericInfo(),
        ApiStatsGenericInfo(),ApiCallStatsInfo(),
    ]  # ImportsInfo(), StringExtractor()
    dim = sum([fe.dim for fe in features])

    def raw_features(self, json_data):

        # features = {"sha256": hashlib.sha256(bytez).hexdigest()}
        features={}
        features.update({fe.name: fe.raw_features(json_data) for fe in self.features})
        return features

    def process_raw_features(self, raw_obj):
        feature_vectors = [fe.process_raw_features(raw_obj[fe.name]) for fe in self.features]
        return np.hstack(feature_vectors).astype(np.float32)

    def feature_vector(self, json_data):
        return self.process_raw_features(self.raw_features(json_data))

if __name__ == "__main__":
    with open(
            r"H:\viruses-sample-2010-05-18\Backdoor.Win32.Amitis.11.g.json") as data_file:
        data = json.load(data_file)
    t = CuckooExtractor()
    print(len(t.feature_vector(data)))
    #print(t.feature_vector(data))
    '''
    H:\viruses-sample-2010-05-18\Backdoor.Win32.Amitis.11.g H:\viruses-sample-2010-05-18\Backdoor.Win32.BackAttack.19
    '''






# # 获取file_api_nunique, file_api_cnt
# # 每个样本调用不同api的类别数和调用api的总次数
# def get_api_cnt(json_data):
#     # print(len(data['behavior']['apistats']))
#     # if 'behavior' not in json_data:
#     #     return 0,0
#     if 'behavior' not in json_data or 'apistats' not in json_data['behavior']:
#         return 0, 0
#     data = json_data['behavior']['apistats']
#     api_dict = {}
#     api_cnt = 0
#     for pid, proc_api_dict in data.items():  # 遍历外层，有可能有多个进程
#         # print(len(proc_api_dict),pid,proc_api_dict)
#         for each_api, cnt in proc_api_dict.items():
#             api_cnt += cnt
#             if (each_api in api_dict):
#                 api_dict[each_api] += cnt
#             else:
#                 api_dict[each_api] = cnt
#     # print(api_dict)
#     # print(len(api_dict),api_cnt)
#     return len(api_dict), api_cnt


# # 获取每个样本的tid_mean, tid_min, tid_std,
# # 好像没有区分度
# def get_tid_static_feature(json_data):
#     tid_list = []
#     pid_list = []
#     # min_tid_list,min_pid_list=0,0
#     if 'behavior' not in json_data or 'processes' not in json_data['behavior']:
#         return 0, 0, 0, 0, 0, 0, 0, 0,
#     data = json_data['behavior']['processes']
#     if len(data) > 1:
#         for each_proc in data[1:]:
#             pid_list.append(each_proc['pid'])
#             tid_list.append(each_proc['tid'])
#     # print(pid_list,tid_list)
#
#     return len(tid_list), np.mean(tid_list) if tid_list else 0, min(tid_list) if tid_list else 0, np.std(
#         tid_list) if tid_list else 0, len(pid_list), np.mean(pid_list) if pid_list else 0, min(
#         pid_list) if pid_list else 0, np.std(pid_list) if pid_list else 0


# def get_dropped_cnt(json_data):
#     if 'dropped' in json_data:
#         return len(json_data['dropped'])
#     return 0


# def get_duration(json_data):
#     return json_data['info']['duration']


# def get_source_and_mem_url_cnt(json_data):
#     src_url_cnt = len(json_data['target']['file']['urls'])
#     mem_url_cnt = 0
#     if 'procmemory' in json_data:
#         data = json_data['procmemory']
#         for proc in data:
#             mem_url_cnt += len(proc['urls'])
#     # print(url_cnt)
#     return src_url_cnt, mem_url_cnt


# def get_signature_severity(json_data):
#     # 没有考虑markcount
#     serverity_sum = 0
#     for sig in json_data['signatures']:
#         serverity_sum += sig['severity']
#     # print(serverity_sum)
#     return serverity_sum


# 返回一个api调用类别的字典
# {'synchronisation': 27, 'system': 696, 'file': 3270, 'exception': 2, 'process': 253, 'registry': 66, 'misc': 10, 'ui': 1, 'crypto': 1, 'network': 1}
# def get_api_category_dict(json_data):
#     api_category_dict = {'__notification__': 0, 'certificate': 0, 'crypto': 0, 'exception': 0, 'file': 0,
#                          'iexplore': 0, 'misc': 0, 'netapi': 0, 'network': 0, 'ole': 0,
#                          'process': 0, 'registry': 0, 'resource': 0, 'services': 0, 'synchronisation': 0,
#                          'system': 0, 'ui': 0, }
#     if 'behavior' not in json_data or 'processes' not in json_data['behavior']:
#         return api_category_dict
#     data = json_data['behavior']['processes']
#     if len(data) > 1:
#         for each_proc in data[1:]:
#             for each_call in each_proc['calls']:
#                 if (each_call['category'] in api_category_dict):
#                     api_category_dict[each_call['category']] += 1
#                 else:
#                     api_category_dict[each_call['category']] = 1
#     # print(api_category_dict)
#     return api_category_dict


# def get_network_features(json_data):
#     net_data = json_data['network']
#     domains = len(net_data['domains'])
#     tls = len(net_data['tls'])
#     udp = len(net_data['udp'])
#     dns_servers = len(net_data['dns_servers'])
#     dns_caps = len(net_data['dns'])
#
#     return domains, tls, udp, dns_servers, dns_caps
