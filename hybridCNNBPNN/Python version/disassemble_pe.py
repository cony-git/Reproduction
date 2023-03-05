#反汇编原始pe文件
from multiprocessing import Pool
from optparse import OptionParser
import subprocess as sub
import os
import sys
import pandas as pd
import time

IDA_PATH="E:\\BaiduNetdiskDownload\\IDA_Pro_v7.5_Portable\\idat64.exe"

import os
import signal
import subprocess


def run_cmd(cmd_string, timeout=600):
    """
    执行命令
    :param cmd_string:  string 字符串
    :param timeout:  int 超时设置
    :return:
    """
    p = subprocess.Popen(cmd_string, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True, close_fds=True,
                         start_new_session=True)
    return p
 
    # format = 'utf-8'
    # if platform.system() == "Windows":
    #     format = 'gbk'
 
#     try:
#         (msg, errs) = p.communicate(timeout=timeout)
#         ret_code = p.poll()
#         if ret_code:
#             code = 1
#             msg = "[Error]Called Error ： " + str(msg.decode(format))
#         else:
#             code = 0
#             msg = str(msg.decode(format))
#     except subprocess.TimeoutExpired:
#         # 注意：不能使用p.kill和p.terminate，无法杀干净所有的子进程，需要使用os.killpg
#         p.kill()
#         p.terminate()
#         # os.killpg(p.pid, signal.SIGUSR1)
#         # os.kill(p.pid, signal.CTRL_C_EVENT)
 
#         # 注意：如果开启下面这两行的话，会等到执行完成才报超时错误，但是可以输出执行结果
#         # (outs, errs) = p.communicate()
#         # print(outs.decode('utf-8'))
 
#         code = 1
#         msg = "[ERROR]Timeout Error : Command '" + str(cmd_string) + "' timed out after " + str(timeout) + " seconds"
#     except Exception as e:
#         code = 1
#         msg = "[ERROR]Unknown Error : " + str(e)
 
#     return code, msg

def disasemble_by_ida(file_name):

    if (os.path.isfile(file_name)):
        print(file_name+" disassemble finished")
        #fop = open(os.path.join(os.path.dirname(file_name),os.path.basename(file_name)+'.asm'), 'w')
        #sub.call([ IDA_PATH, "-b","-B", "-P+", file_name],stdout=fop)
        # sub.call([IDA_PATH, "-b", "-B", file_name])
        # p=sub.Popen([IDA_PATH, "-b", "-B", file_name],shell=True)
        p=run_cmd([IDA_PATH, "-b", "-B", file_name])
        print("Next Round!")
        if os.path.isfile(file_name+".i64"):
            os.remove(file_name+".i64")
        # if os.path.isfile(file_name+".idb"):
        #     os.remove(file_name+".idb")
        # now delete the binary, we do not need it anymore.
        # sub.call(["rm", file_path])
    
    # return file_name+'.asm'
    return p

def disasemble_batch_by_ida(file_dir):
    for filename in os.listdir(file_dir):
        if filename.split('.')[-1] not in ['json','asm']:
            disasemble_by_ida(os.path.join(file_dir,filename))

def disasemble_by_objdump(file_name):
    # command_line = "objdump -d {:s} > {:s}".format(file_path1, asm_file_name)
    # sub.call(["rasm2", "-d", "-a", "x86", "-s", "intel", "-f", file_path, "-O", asm_file_name])
    # sub.call(["./idaq69", "-B", file_path])
    # sub.call(["python", "vivisect", "-B", file_path])
    # sub.call(["objdump", "-g", "-x", "-D", "-s", "-t", "-T", "-M", "intel", file_path], stdout=fop)
    # sub.call(["ndisasm", "-a", "-p", "intel", file_path])
    fop=""
    sub.call(["objdump", "-g", "-x", "-D", "-s", "-t", "-T", "-M", "intel", file_name], stdout=fop)

def write_unpacked_file_list(packer_id_feature_file, unpacked_list_file_name):
    # Load the malware packer id features sets from the sample set.
    packer_id_features = pd.read_csv(packer_id_feature_file)
    unpacked_files = packer_id_features[packer_id_features['is_packed'] == 0]
    unpacked_pe_files = unpacked_files[unpacked_files['valid_pe'] == 1]

    fop = open(unpacked_list_file_name, 'w')
    counter = 0

    for idx, file_name in enumerate(unpacked_pe_files['file_name']):
        fdesc = unpacked_pe_files.iloc[idx, 1]
        if fdesc.startswith("PE32+"):  # IDA Pro free does not do 64 bit PE binary files,
            continue  # use objdump instead.
        full_name = "VirusShare_" + file_name + "\n"
        fop.write(full_name)
        counter += 1

    print("Wrote {:d} filenames.".format(counter))

    fop.close()

    return


def get_unpacked_file_list(packer_id_feature_file, file_id_feature_file, trid_id_feature_file):
    # Load the malware packer id features and file id features from the sample set.
    packer_id_features = pd.read_csv(packer_id_feature_file)
    file_id_features = pd.read_csv(file_id_feature_file)
    trid_id_features = pd.read_csv(trid_id_feature_file)

    # Get a list of unpacked PE files that are not .NET CIL format and not 64 bit.
    # IDA Pro cannot disassemble .NET files, have to use Ildisasm.exe in Visual Studio,
    # and free version will not disassemble 64 bit, use objdump instead.
    unpacked_files = packer_id_features[packer_id_features['is_packed'] == 0]
    unpacked_pe_files = unpacked_files[unpacked_files['valid_pe'] == 1]
    not_dot_net = []
    counter = 0
    dot_net_counter = 0
    amd64_bit_counter = 0

    # Get the trid and file rows that are for unpacked PE files.
    trids = trid_id_features[trid_id_features['file_name'].isin(unpacked_pe_files['file_name'])]
    fids = file_id_features[file_id_features['file_name'].isin(unpacked_pe_files['file_name'])]

    # Iterate over the unpacked PE file list and check if each is a .NET file.
    # If not a .NET or 64 bit file then add to file list.
    pe_names_list = unpacked_pe_files['file_name']

    for idx, file_name in enumerate(pe_names_list):
        trid_name = trids.iloc[idx, 1]
        fid_name = fids.iloc[idx, 1]
        trid_name = trid_name.lower()
        fid_name = fid_name.lower()

        if trid_name.find('.net') > -1 or fid_name.find('.net') > -1:
            print('Found: {:s} - {:s}'.format(trid_name, fid_name))
            dot_net_counter += 1
            continue

        if trid_name.find('win64') > -1 or fid_name.startswith('pe32+'):
            print('Found: {:s} - {:s}'.format(trid_name, fid_name))
            amd64_bit_counter += 1
            continue

        # print('Found: {:s} - {:s}'.format(trid_name, fid_name))
        not_dot_net.append(file_name)
        counter += 1

    file_list = []
    write_list = []
    counter = 0

    # Iterate over the file list and prepend the full file name.
    for file_name in not_dot_net:
        full_name = "VirusShare_" + file_name
        file_list.append(full_name)
        write_list.append(full_name + "\n")
        counter += 1

    if (len(file_list) > 0):
        fop = open('data/temp-unpacked-pe-non-dot-net.txt', 'w')
        fop.writelines(write_list)
        fop.close()

    print("Got {:d} unpacked PE files.".format(counter))
    print("Got {:d} .NET file and {:d} 64 Bit files.".format(dot_net_counter, amd64_bit_counter))

    return file_list


def get_64bit_pe_file_list(packer_id_feature_file, file_id_feature_file, trid_id_feature_file):
    # Load the malware packer id features and file id features from the sample set.
    packer_id_features = pd.read_csv(packer_id_feature_file)
    file_id_features = pd.read_csv(file_id_feature_file)
    trid_id_features = pd.read_csv(trid_id_feature_file)

    # Get a list of 64 bit unpacked PE files that are not .NET CIL format.
    unpacked_files = packer_id_features[packer_id_features['is_packed'] == 0]
    unpacked_pe_files = unpacked_files[unpacked_files['valid_pe'] == 1]
    dot_net_counter = 0
    amd64_bit_counter = 0
    amd64_bit_file_list = []

    # Get the trid and file rows that are for unpacked PE files.
    trids = trid_id_features[trid_id_features['file_name'].isin(unpacked_pe_files['file_name'])]
    fids = file_id_features[file_id_features['file_name'].isin(unpacked_pe_files['file_name'])]

    # Iterate over the unpacked PE file list and check if each is a .NET file.
    # If not a .NET file then add to file list.
    pe_names_list = unpacked_pe_files['file_name']

    for idx, file_name in enumerate(pe_names_list):
        trid_name = trids.iloc[idx, 1]
        fid_name = fids.iloc[idx, 1]
        trid_name = trid_name.lower()
        fid_name = fid_name.lower()

        if trid_name.find('.net') > -1 or fid_name.find('.net') > -1:
            print('Found: {:s} - {:s}'.format(trid_name, fid_name))
            dot_net_counter += 1
            continue

        if trid_name.find('win64') > -1 or fid_name.startswith('pe32+'):
            print('Found: {:s} - {:s}'.format(trid_name, fid_name))
            amd64_bit_counter += 1
            amd64_bit_file_list.append(file_name)
            continue

    file_list = []
    write_list = []
    counter = 0

    # Iterate over the file list and prepend the full file name.
    for file_name in amd64_bit_file_list:
        full_name = "VirusShare_" + file_name
        file_list.append(full_name)
        write_list.append(full_name + "\n")
        counter += 1

    if (len(file_list) > 0):
        fop = open('data/temp-unpacked-64bit-pe-file-list-.txt', 'w')
        fop.writelines(write_list)
        fop.close()

    print("Got {:d} unpacked PE files.".format(counter))
    print("Got {:d} .NET file and {:d} 64 Bit files.".format(dot_net_counter, amd64_bit_counter))

    return file_list


def disassemble_pe_mem_dumps(file_list):
    # Disassemble the unpacked memory segments dumped by the unpack tool.
    #
    # TODO: everything

    return


def disassemble_pe_binaries(file_list):
    # Can use the command "objdump -d file_name -o file_name.asm" to dump out all
    # sections of the PE binary and generate assembly code.
    # However the objdump output is not optimal for machine learning objectives,
    # as we need to translate call operand target addresses to function names,
    # A better alternative is to use IDA Pro in batch mode to generate the
    # assembly code.
    #
    # NOTE: IDA Pro Demo does not save any output, IDA Pro Free has a
    #       popup window on startup that prevents batch processing mode.
    #

    counter = 0
    disassed = 0
    error_count = 0
    pid = os.getpid()
    log_file = "data/" + str(pid) + '-pe-disass-log.txt'

    smsg = "{:d} Disassembling {:d} binary PE32 files.".format(pid, len(file_list))
    print(smsg)
    flog = open(log_file, 'w')
    flog.write(smsg + "\n")

    for file_name in file_list:
        file_path = file_name.rstrip()  # remove the newlines or else !!!
        asm_file_name = file_path + ".pe.asm"
        hdr_file_name = file_path + ".pe.txt"

        if (os.path.isfile(file_path)):

            # command_line = "objdump -d {:s} > {:s}".format(file_path1, asm_file_name)
            # sub.call(["rasm2", "-d", "-a", "x86", "-s", "intel", "-f", file_path, "-O", asm_file_name])
            # sub.call(["./idaq69", "-B", file_path])
            # sub.call(["python", "vivisect", "-B", file_path])
            # sub.call(["objdump", "-g", "-x", "-D", "-s", "-t", "-T", "-M", "intel", file_path], stdout=fop)
            # sub.call(["ndisasm", "-a", "-p", "intel", file_path])

            # Dump the assembly code listing.
            sub.call(["wine", '/opt/vs/ida/idag.exe', "-B", "-P+", file_path])

            # now delete the binary, we do not need it anymore.
            # sub.call(["rm", file_path])

            disassed += 1

        else:
            error_count += 1

        counter += 1

        if (counter % 10) == 0:  # print progress
            smsg = '{:d} Disassembled: {:d} - {:s}'.format(pid, counter, file_name)
            print(smsg)
            flog.write(smsg + "\n")

    smsg = "{:d} Disassembled {:d} binaries with {:d} file path errors.".format(pid, disassed, error_count)
    print(smsg)
    flog.write(smsg + "\n")
    flog.close()

    # sub.call(["mv", "*.asm", "/opt/vs/asm"])

    return


def disassemble_pe64_binaries(file_list):
    counter = 0
    disassed = 0
    error_count = 0
    pid = os.getpid()
    log_file = "data/" + str(pid) + '-pe-disass-log-64.txt'

    smsg = "{:d} Disassembling {:d} binary PE64 files.".format(pid, len(file_list))
    print(smsg)
    flog = open(log_file, 'w')
    flog.write(smsg + "\n")

    for file_name in file_list:
        file_path = file_name.rstrip()  # remove the newlines or else !!!
        asm_file_name = file_path + ".64.pe.asm"
        # hdr_file_name = file_path + ".64.pe.txt"

        if (os.path.isfile(file_path)):
            fop = open(asm_file_name, 'w')
            # Dump the assembly code listing.
            sub.call(["objdump", "-g", "-x", "-D", "-s", "-t", "-T", "-M", "intel", file_path], stdout=fop)
            disassed += 1
            fop.close
        else:
            error_count += 1

        counter += 1

        if (counter % 10) == 0:  # print progress
            smsg = '{:d} Disassembled: {:d} - {:s}'.format(pid, counter, file_name)
            print(smsg)
            flog.write(smsg + "\n")

    smsg = "{:d} Disassembled {:d} PE64 binaries with {:d} file path errors.".format(pid, disassed, error_count)
    print(smsg)
    flog.write(smsg + "\n")
    flog.close()

    return


def extract_pe_headers(file_list):
    # Use the command "objdump -g -x file_name" to dump out all
    # the header sections of the PE binary.
    # Separating this from the disassembly because of errors with
    # IDA Pro disassembly making everything too complicated.

    counter = 0
    disassed = 0
    error_count = 0

    print("Extracting headers from {:d} binary PE files.".format(len(file_list)))

    for file_name in file_list:
        file_path = file_name.rstrip()  # remove the newlines or else !!!
        # asm_file_name = file_path + ".pe.asm"
        hdr_file_name = file_path + ".pe.txt"

        if (os.path.isfile(file_path)):

            # Dump the PE section headers and import tables.
            fop = open(hdr_file_name, "w")
            sub.call(["objdump", "-g", "-x", file_path], stdout=fop)
            fop.close()

            disassed += 1

        else:
            # print("Error: file does not exist - {:s}".format(file_path))
            error_count += 1

        counter += 1
        if (counter % 1000) == 0:  # print progress
            print('Extracted: {:d} - {:s}'.format(counter, file_name))

    print("Extracted {:d} binaries with {:d} file path errors.".format(disassed, error_count))

    return


def rename_asm_files(ext_dir, new_dir, file_extension):
    # Rename all the PE ASM files and move them to a new directory
    # so it is easier to process them.
    # Example file name extensions:
    # filename.pe.asm - 32 bit
    # filename.64.pe.asm - 64 bit
    # filename.net.pe.asm - .NET CIL

    file_list = os.listdir(ext_dir)
    counter = 0

    for fname in file_list:
        if fname.endswith('.asm'):
            file_path = ext_dir + fname
            trunc_name = fname[0:fname.find('.asm')]
            new_path = new_dir + trunc_name + file_extension
            result = sub.check_call(['mv', file_path, new_path])
            counter += 1

            if (counter % 1000) == 0:
                print('Renamed {:d} ASM files.'.format(counter))

    print('Completed rename of {:d} ASM files.'.format(counter))

    return


def validate_disassembly(asm_path, hdr_path, file_ext):
    # Check disassembly results for the PE/COFF files in the malware set.

    t1asm = os.listdir(asm_path)
    t1hdr = os.listdir(hdr_path)
    asm_files = []
    hdr_files = []

    for fname in t1asm:
        if fname.endswith('.pe.asm'):
            asm_files.append(fname)

    for fname in t1hdr:
        if fname.endswith('.pe.txt'):
            hdr_files.append(fname)

    print("asm dir: {:d} asm files {:d} hdr dir {:d} hdr files {:d}".format(len(t1asm), len(asm_files), len(t1hdr),
                                                                            len(hdr_files)))

    counter = 0
    missing_hdr_list = []
    # 互相判断文件在另一个列表里是否存在？
    for fname in asm_files:
        hdr_name = fname.replace('.asm', '.txt')
        if hdr_name not in hdr_files:
            print("{:s} not in header file list.".format(hdr_name))
            counter += 1
            missing_hdr_list.append(hdr_name)

    print("{:d} missing header files.".format(counter))

    counter = 0
    missing_asm_list = []

    for fname in hdr_files:
        asm_name = fname.replace('.txt', '.asm')
        if asm_name not in asm_files:
            print("{:s} not in asm file list.".format(asm_name))
            counter += 1
            missing_asm_list.append(asm_name)

    print("{:d} missing assembly files.".format(counter))

    if len(missing_asm_list) > 0:
        counter = 0
        fop = open('data/temp-disass-missing-asm-files' + file_ext + '.txt', 'w')
        for fname in missing_asm_list:
            fop.write(fname + "\n")
            counter += 1

        fop.close()
        print("Wrote {:d} missing asm file names.".format(counter))

    if len(missing_hdr_list) > 0:
        counter = 0
        fop = open('data/temp-disass-missing-hdr-files' + file_ext + '.txt', 'w')
        for fname in missing_hdr_list:
            fop.write(fname + "\n")
            counter += 1

        fop.close()
        print("Wrote {:d} missing hdr file names.".format(counter))

    counter = 0
    bad_asm_list = []

    for fname in asm_files:
        fsize = os.path.getsize(asm_path + fname)
        if fsize < 1000:  # 仅根据文件size判断是否bad是否合理？
            print("{:s} bad output, filesize = {:d}.".format(fname, fsize))
            counter += 1
            bad_asm_list.append(fname)

    print("{:d} bad asm files.".format(counter))

    counter = 0
    bad_hdr_list = []

    for fname in hdr_files:
        fsize = os.path.getsize(hdr_path + fname)
        if fsize < 1000:
            print("{:s} bad output, filesize = {:d}.".format(fname, fsize))
            counter += 1
            bad_hdr_list.append(fname)

    print("{:d} bad header files.".format(counter))

    if len(bad_hdr_list) > 0:
        counter = 0
        fop = open('data/temp-disass-bad-hdr-files' + file_ext + '.txt', 'w')
        for fname in bad_hdr_list:
            fop.write(fname + "\n")
            counter += 1

        fop.close()

    print("Wrote {:d} bad hdr file names.".format(counter))

    if len(bad_asm_list) > 0:
        counter = 0
        fop = open('data/temp-disass-bad-asm-files' + file_ext + '.txt', 'w')
        for fname in bad_asm_list:
            fop.write(fname + "\n")
            counter += 1

        fop.close()

    print("Wrote {:d} bad asm file names.".format(counter))

    return


def run_disassembly_processes(tfiles):
    # Spawn worker processes.

    quart = len(tfiles) / 4
    train1 = tfiles[:quart]
    train2 = tfiles[quart:(2 * quart)]
    train3 = tfiles[(2 * quart):(3 * quart)]
    train4 = tfiles[(3 * quart):]

    print(
        "Files: {:d} - {:d} - {:d}".format(len(tfiles), quart, (len(train1) + len(train2) + len(train3) + len(train4))))

    trains = [train1, train2, train3, train4]
    p = Pool(4)
    p.map(disassemble_pe_binaries, trains)

    return


def run_header_extraction_processes(tfiles):
    # Spawn worker processes.

    quart = len(tfiles) / 4
    train1 = tfiles[:quart]
    train2 = tfiles[quart:(2 * quart)]
    train3 = tfiles[(2 * quart):(3 * quart)]
    train4 = tfiles[(3 * quart):]

    print(
        "Files: {:d} - {:d} - {:d}".format(len(tfiles), quart, (len(train1) + len(train2) + len(train3) + len(train4))))

    trains = [train1, train2, train3, train4]
    p = Pool(4)
    p.map(extract_pe_headers, trains)

    return




class Multi_Params(object):
    def __init__(self, outfile="", tokenfile="", fieldnames=[], filelist=[]):
        self.out_file = outfile
        self.token_file = tokenfile
        self.field_names = fieldnames
        self.file_list = filelist


# Start of Script
if __name__ == "__main__":
    # file_test="E:\\PycharmProjects\\malware\\sample\\jocker.exe"
    # disasemble_by_ida(file_test)
    FILE_DIR=r"E:\STD_PE"
    count=0
    for root, dirs, files in os.walk(FILE_DIR):
        for file in files:
            suffix=file.split('.')[-1]
            if suffix!='json' and \
                    suffix!='asm' and \
                    suffix != 'id0' and \
                    suffix != 'id1' and \
                    suffix != 'id2' and \
                    suffix != 'nam' and \
                    suffix != 'til' and \
                    file+'.asm' not in files:# 已经反汇编过的就不再反汇编，只反汇编无后缀的二进制文件
                tmp_lst=[]
                p=disasemble_by_ida(os.path.join(root, file))
                tmp_lst.append(p)
                if count%8==0:
                    for item in tmp_lst:
                        item.wait()
                    tmp_lst=[]
                count+=1
                print(count)


        # disasemble_batch_by_ida(FILE_DIR)
