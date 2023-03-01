from asyncio.windows_events import NULL
from hashlib import sha1
from extract_cuckoo_report import ApiTfidfStatsInfo
import os
import json


apilists = [
    "__process__",
    "__anomaly__",
    "__exception__",
    "__missing__",
    "__action__",
    "__guardrw__",
    "IWbemServices_ExecMethod",
    "IWbemServices_ExecMethodAsync",
    "IWbemServices_ExecQuery",
    "IWbemServices_ExecQueryAsync",
    "ControlService",
    "CreateServiceA",
    "CreateServiceW",
    "CryptAcquireContextA",
    "CryptAcquireContextW",
    "CryptCreateHash",
    "CryptDecrypt",
    "CryptEncrypt",
    "CryptExportKey",
    "CryptGenKey",
    "CryptHashData",
    "DeleteService",
    "EnumServicesStatusA",
    "EnumServicesStatusW",
    "GetUserNameA",
    "GetUserNameW",
    "LookupAccountSidW",
    "LookupPrivilegeValueW",
    "NotifyBootConfigStatus",
    "OpenSCManagerA",
    "OpenSCManagerW",
    "OpenServiceA",
    "OpenServiceW",
    "RegCloseKey",
    "RegCreateKeyExA",
    "RegCreateKeyExW",
    "RegDeleteKeyA",
    "RegDeleteKeyW",
    "RegDeleteValueA",
    "RegDeleteValueW",
    "RegEnumKeyExA",
    "RegEnumKeyExW",
    "RegEnumKeyW",
    "RegEnumValueA",
    "RegEnumValueW",
    "RegOpenKeyExA",
    "RegOpenKeyExW",
    "RegQueryInfoKeyA",
    "RegQueryInfoKeyW",
    "RegQueryValueExA",
    "RegQueryValueExW",
    "RegSetValueExA",
    "RegSetValueExW",
    "StartServiceA",
    "StartServiceCtrlDispatcherW",
    "StartServiceW",
    "TaskDialog",
    "CertControlStore",
    "CertCreateCertificateContext",
    "CertOpenStore",
    "CertOpenSystemStoreA",
    "CertOpenSystemStoreW",
    "CryptDecodeMessage",
    "CryptDecodeObjectEx",
    "CryptDecryptMessage",
    "CryptEncryptMessage",
    "CryptHashMessage",
    "CryptProtectData",
    "CryptProtectMemory",
    "CryptUnprotectData",
    "CryptUnprotectMemory",
    "DnsQuery_A",
    "DnsQuery_UTF8",
    "DnsQuery_W",
    "GetAdaptersAddresses",
    "GetAdaptersInfo",
    "GetBestInterfaceEx",
    "GetInterfaceInfo",
    "AssignProcessToJobObject",
    "CopyFileA",
    "CopyFileExW",
    "CopyFileW",
    "CreateActCtxW",
    "CreateDirectoryExW",
    "CreateDirectoryW",
    "CreateJobObjectW",
    "CreateProcessInternalW",
    "CreateRemoteThread",
    "CreateRemoteThreadEx",
    "CreateThread",
    "CreateToolhelp32Snapshot",
    "DeleteFileW",
    "DeviceIoControl",
    "FindFirstFileExA",
    "FindFirstFileExW",
    "FindResourceA",
    "FindResourceExA",
    "FindResourceExW",
    "FindResourceW",
    "GetComputerNameA",
    "GetComputerNameW",
    "GetDiskFreeSpaceExW",
    "GetDiskFreeSpaceW",
    "GetFileAttributesExW",
    "GetFileAttributesW",
    "GetFileInformationByHandle",
    "GetFileInformationByHandleEx",
    "GetFileSize",
    "GetFileSizeEx",
    "GetFileType",
    "GetLocalTime",
    "GetNativeSystemInfo",
    "GetShortPathNameW",
    "GetSystemDirectoryA",
    "GetSystemDirectoryW",
    "GetSystemInfo",
    "GetSystemTime",
    "GetSystemTimeAsFileTime",
    "GetSystemWindowsDirectoryA",
    "GetSystemWindowsDirectoryW",
    "GetTempPathW",
    "GetTickCount",
    "GetTimeZoneInformation",
    "GetVolumeNameForVolumeMountPointW",
    "GetVolumePathNameW",
    "GetVolumePathNamesForVolumeNameW",
    "GlobalMemoryStatus",
    "GlobalMemoryStatusEx",
    "IsDebuggerPresent",
    "LoadResource",
    "Module32FirstW",
    "Module32NextW",
    "MoveFileWithProgressW",
    "OutputDebugStringA",
    "Process32FirstW",
    "Process32NextW",
    "ReadProcessMemory",
    "RemoveDirectoryA",
    "RemoveDirectoryW",
    "SearchPathW",
    "SetEndOfFile",
    "SetErrorMode",
    "SetFileAttributesW",
    "SetFileInformationByHandle",
    "SetFilePointer",
    "SetFilePointerEx",
    "SetFileTime",
    "SetInformationJobObject",
    "SetStdHandle",
    "SetUnhandledExceptionFilter",
    "SizeofResource",
    "Thread32First",
    "Thread32Next",
    "WriteConsoleA",
    "WriteConsoleW",
    "WriteProcessMemory",
    "WNetGetProviderNameW",
    "CDocument_write",
    "CElement_put_innerHTML",
    "CHyperlink_SetUrlComponent",
    "CIFrameElement_CreateElement",
    "CImgElement_put_src",
    "CScriptElement_put_src",
    "CWindow_AddTimeoutCode",
    "system",
    "PRF",
    "Ssl3GenerateKeyMaterial",
    "NetGetJoinInformation",
    "NetShareEnum",
    "NetUserGetInfo",
    "NetUserGetLocalGroups",
    "LdrGetDllHandle",
    "LdrGetProcedureAddress",
    "LdrLoadDll",
    "LdrUnloadDll",
    "NtAllocateVirtualMemory",
    "NtClose",
    "NtCreateDirectoryObject",
    "NtCreateFile",
    "NtCreateKey",
    "NtCreateMutant",
    "NtCreateProcess",
    "NtCreateProcessEx",
    "NtCreateSection",
    "NtCreateThread",
    "NtCreateThreadEx",
    "NtCreateUserProcess",
    "NtDelayExecution",
    "NtDeleteFile",
    "NtDeleteKey",
    "NtDeleteValueKey",
    "NtDeviceIoControlFile",
    "NtDuplicateObject",
    "NtEnumerateKey",
    "NtEnumerateValueKey",
    "NtFreeVirtualMemory",
    "NtGetContextThread",
    "NtLoadDriver",
    "NtLoadKey",
    "NtLoadKey2",
    "NtLoadKeyEx",
    "NtMakePermanentObject",
    "NtMakeTemporaryObject",
    "NtMapViewOfSection",
    "NtOpenDirectoryObject",
    "NtOpenFile",
    "NtOpenKey",
    "NtOpenKeyEx",
    "NtOpenMutant",
    "NtOpenProcess",
    "NtOpenSection",
    "NtOpenThread",
    "NtProtectVirtualMemory",
    "NtQueryAttributesFile",
    "NtQueryDirectoryFile",
    "NtQueryFullAttributesFile",
    "NtQueryInformationFile",
    "NtQueryKey",
    "NtQueryMultipleValueKey",
    "NtQuerySystemInformation",
    "NtQuerySystemTime",
    "NtQueryValueKey",
    "NtQueueApcThread",
    "NtReadFile",
    "NtReadVirtualMemory",
    "NtRenameKey",
    "NtReplaceKey",
    "NtResumeThread",
    "NtSaveKey",
    "NtSaveKeyEx",
    "NtSetContextThread",
    "NtSetInformationFile",
    "NtSetValueKey",
    "NtShutdownSystem",
    "NtSuspendThread",
    "NtTerminateProcess",
    "NtTerminateThread",
    "NtUnloadDriver",
    "NtUnmapViewOfSection",
    "NtWriteFile",
    "NtWriteVirtualMemory",
    "RtlAddVectoredContinueHandler",
    "RtlAddVectoredExceptionHandler",
    "RtlCompressBuffer",
    "RtlCreateUserProcess",
    "RtlCreateUserThread",
    "RtlDecompressBuffer",
    "RtlDecompressFragment",
    "RtlDispatchException",
    "RtlRemoveVectoredContinueHandler",
    "RtlRemoveVectoredExceptionHandler",
    "CoCreateInstance",
    "CoCreateInstanceEx",
    "CoGetClassObject",
    "CoInitializeEx",
    "CoInitializeSecurity",
    "CoUninitialize",
    "OleConvertOLESTREAMToIStorage",
    "OleInitialize",
    "UuidCreate",
    "DecryptMessage",
    "EncryptMessage",
    "GetUserNameExA",
    "GetUserNameExW",
    "ReadCabinetState",
    "SHGetFolderPathW",
    "SHGetSpecialFolderLocation",
    "ShellExecuteExW",
    "NetShareEnum",
    "ObtainUserAgentString",
    "URLDownloadToFileW",
    "DrawTextExA",
    "DrawTextExW",
    "EnumWindows",
    "ExitWindowsEx",
    "FindWindowA",
    "FindWindowExA",
    "FindWindowExW",
    "FindWindowW",
    "GetAsyncKeyState",
    "GetCursorPos",
    "GetForegroundWindow",
    "GetKeyState",
    "GetKeyboardState",
    "GetSystemMetrics",
    "LoadStringA",
    "LoadStringW",
    "MessageBoxTimeoutA",
    "MessageBoxTimeoutW",
    "RegisterHotKey",
    "SendNotifyMessageA",
    "SendNotifyMessageW",
    "SetWindowsHookExA",
    "SetWindowsHookExW",
    "UnhookWindowsHookEx",
    "vbe6_CallByName",
    "vbe6_Close",
    "vbe6_CreateObject",
    "vbe6_GetIDFromName",
    "vbe6_GetObject",
    "vbe6_Import",
    "vbe6_Invoke",
    "vbe6_Open",
    "vbe6_Print",
    "vbe6_Shell",
    "GetFileVersionInfoExW",
    "GetFileVersionInfoSizeExW",
    "GetFileVersionInfoSizeW",
    "GetFileVersionInfoW",
    "DeleteUrlCacheEntryA",
    "DeleteUrlCacheEntryW",
    "HttpOpenRequestA",
    "HttpOpenRequestW",
    "HttpQueryInfoA",
    "HttpSendRequestA",
    "HttpSendRequestW",
    "InternetCloseHandle",
    "InternetConnectA",
    "InternetConnectW",
    "InternetCrackUrlA",
    "InternetCrackUrlW",
    "InternetGetConnectedState",
    "InternetGetConnectedStateExA",
    "InternetGetConnectedStateExW",
    "InternetOpenA",
    "InternetOpenUrlA",
    "InternetOpenUrlW",
    "InternetOpenW",
    "InternetQueryOptionA",
    "InternetReadFile",
    "InternetSetOptionA",
    "InternetSetStatusCallback",
    "InternetWriteFile",
    "timeGetTime",
    "ConnectEx",
    "GetAddrInfoW",
    "TransmitFile",
    "WSAAccept",
    "WSAConnect",
    "WSARecv",
    "WSARecvFrom",
    "WSASend",
    "WSASendTo",
    "WSASocketA",
    "WSASocketW",
    "WSAStartup",
    "accept",
    "bind",
    "closesocket",
    "connect",
    "getaddrinfo",
    "gethostbyname",
    "getsockname",
    "ioctlsocket",
    "listen",
    "recv",
    "recvfrom",
    "select",
    "send",
    "sendto",
    "setsockopt",
    "shutdown",
    "socket",
    "COleScript_Compile",
    "ActiveXObjectFncObj_Construct",
    "COleScript_Compile",
    "pdf_unescape",
    "pdf_eval",
    "BaseExecMgr_setJit",
    "BaseExecMgr_setNative",
    "Loader__loadBytes",
    "JsGlobalObjectDefaultEvalHelper",
    "Math_random",
    NULL

]

api_order_dict=dict()
for i in range(len(apilists)):
    api_order_dict[apilists[i]]=i

sample_api_order=dict()

def getAPI(target_folder):
    for json_file in os.listdir(target_folder):
        full_path = os.path.join(target_folder, json_file)
        with open(full_path, "r") as f:
            json_dict = json.load(f)
            sha1 = json_dict["target"]["file"]["sha1"]
            api_list = ApiTfidfStatsInfo().raw_features(json_dict)
            order_list=[api_order_dict[x] for x in api_list]
            # print(sha1, api_list,order_list)
            # print(len(api_list))
            sample_api_order[sha1]=order_list



getAPI(target_folder=".\\cuckoo_reports")
import json
data=json.dumps(sample_api_order)
with open("api.json","w") as f:
    f.write(data)
# print(sample_api_order)
# t=api_order_dict["GetSystemTimeAsFileTime"]
# print(t)
print(len(apilists))


