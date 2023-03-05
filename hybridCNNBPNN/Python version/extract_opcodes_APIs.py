from distutils.ccompiler import new_compiler
from html.entities import name2codepoint
from importlib.resources import path
import opcode
import pickle
from collections import Counter
import os
from pyexpat import features
import numpy as np
#from sklearn.decomposition import PCA  # 加载PCA算法包
#from PIL import Image
import re
# import pymongo
import hashlib
import json

import numpy
import sklearn
import warnings
warnings.filterwarnings('ignore')

# from copulas.datasets import sample_trivariate_xyz
# from copulas.multivariate import GaussianMultivariate
# from copulas.visualization import compare_3d

# from copulas.multivariate import VineCopula
import pandas
from sklearn import tree


# 统计反汇编文件中 关键词、节名、寄存器名、操作码的次数
# 还未统计 ngram 特征
KEYWORDS = ['Virtual', 'Offset', 'loc', 'Import', 'Imports', 'var', 'Forwarder', 'UINT', 'LONG', 'BOOL', 'WORD', 'BYTES', 'large', 'short', 'dd', 'db', 'dw', 'XREF', 'ptr', 'DATA', 'FUNCTION', 'extrn', 'byte', 'word', 'dword', 'char', 'DWORD', 'stdcall', 'arg', 'locret', 'asc', 'align', 'WinMain', 'unk', 'cookie', 'off', 'nullsub', 'DllEntryPoint', 'System32', 'dll', 'CHUNK', 'BASS', 'HMENU', 'DLL', 'LPWSTR', 'void', 'HRESULT', 'HDC', 'LRESULT', 'HANDLE', 'HWND', 'LPSTR', 'int', 'HLOCAL', 'FARPROC', 'ATOM', 'HMODULE', 'WPARAM', 'HGLOBAL', 'entry', 'rva', 'COLLAPSED', 'config', 'exe', 'Software', 'CurrentVersion', '__imp_', 'INT_PTR', 'UINT_PTR', '---Seperator', 'PCCTL_CONTEXT', '__IMPORT_', 'INTERNET_STATUS_CALLBACK', '.rdata:', '.data:', '.text:', 'case', 'installdir', 'market', 'microsoft', 'policies', 'proc', 'scrollwindow', 'search', 'trap', 'visualc', '___security_cookie', 'assume', 'callvirtualalloc', 'exportedentry', 'hardware', 'hkey_current_user', 'hkey_local_machine', 'sp-analysisfailed', 'unableto']
KNOWN_SECTIONS = ['.text', '.data', '.bss', '.rdata', '.edata', '.idata', '.rsrc', '.tls', '.reloc']
REGISTERS = ['edx', 'esi', 'es', 'fs', 'ds', 'ss', 'gs', 'cs', 'ah', 'al',
                 'ax','bh','bl','bx','ch','cl','cx','dh','dl','dx',
                 'eax','ebp','ebx','ecx','edi','esp']

OPCODES = ["mov", "push", "call", "retn", "lea", "add", "cmp", "pop", "test", "jmp", "sub", "xor", "movzx", "and", "or", "shl", "inc", "sar", "imul", "shr", "movdqa", "fld", "fstp", "movsd", "jle", "movss", "nop", "jb", "movaps", "movq", "jl", "dec", "movsx", "jg", "jbe", "ja", "movsxd", "jge", "js", "mulss", "fxch", "paddd", "paddw", "neg", "movups", "fmul", "movdqu", "movd", "addss", "vmovups", "sbb", "punpcklbw", "mulsd", "cdq", "leave", "pmaddwd", "jns", "adc", "addsd", "movapd", "pxor", "faddp", "rol", "punpcklwd", "psubw", "xorps", "not", "fadd", "psadbw", "fild", "subss", "punpckhwd", "psrad", "subsd", "mul", "por", "vmovdqa", "pshufd", "packssdw", "xchg", "pand", "fsub", "ror", "bswap", "punpckldq", "vpaddd", "fldcw", "vmovdqu", "packuswb", "cmovg", "fmulp", "psraw", "fst", "idiv", "fucomi", "ret", "cmova", "comiss", "psubusw", "fldz", "pmullw", "comisd", "shrd", "jp", "div", "cdqe", "pmuludq", "pmaddubsw", "cvttsd2si", "psrlq", "cvtdq2ps", "fdiv", "vmulpd", "punpckhbw", "psrlw", "vaddpd", "fnstsw", "fsubp", "psubd", "cvtdq2pd", "mulpd", "cmovl", "kmovw", "ucomisd", "movhlps", "addpd", "cmovle", "vsubpd", "vmovapd", "cmovs", "vpaddw", "fistp", "cvtsi2sd", "pmulhrsw", "divsd", "bt", "paddsw", "psrldq", "pshuflw", "palignr", "vmovaps", "shld", "cvtps2pd", "xorpd", "fucomip", "psllw", "psubusb", "addps", "psrld", "vpmaddwd", "vpsadbw", "subpd", "divss", "punpcklqdq", "fsubr", "cmovbe", "cmovb", "fld1", "mulps"]

with open('API_1500.pkl', 'rb') as file:
    MODIFIED_APIS = pickle.load(file)
    MODIFIED_APIS=list(set(MODIFIED_APIS))
# MODIFIED_APIS=MODIFIED_APIS[0:100000]


# 数据中心化
def Z_centered(dataMat):
    rows, cols = dataMat.shape
    meanVal = np.mean(dataMat, axis=0)  # 按列求均值，即求各个特征的均值
    meanVal = np.tile(meanVal, (rows, 1))
    newdata = dataMat - meanVal
    return newdata, meanVal

# 得到最大的k个特征值和特征向量
def EigDV(covMat, k):
    D, V = np.linalg.eig(covMat)  # 得到特征值和特征向量
    # k = Percentage2n(D, p)  # 确定k值
    print("降维后的特征个数：" + str(k) + "\n")
    eigenvalue = np.argsort(D)
    K_eigenValue = eigenvalue[-1:-(k + 1):-1]
    K_eigenVector = V[:, K_eigenValue]
    return K_eigenValue, K_eigenVector


# 得到降维后的数据
def getlowDataMat(DataMat, K_eigenVector):
    return DataMat * K_eigenVector


# 重构数据
def Reconstruction(lowDataMat, K_eigenVector, meanVal):
    reconDataMat = lowDataMat * K_eigenVector.T + meanVal
    return reconDataMat


# PCA算法
def PCA(data, k):
    dataMat = np.float32(np.mat(data))
    # 数据中心化
    dataMat, meanVal = Z_centered(dataMat)
    # 计算协方差矩阵
    # covMat = Cov(dataMat)
    covMat = np.cov(dataMat, rowvar=0)
    # 得到最大的k个特征值和特征向量
    D, V = EigDV(covMat, k)
    # 得到降维后的数据
    lowDataMat = getlowDataMat(dataMat, V)
    # 重构数据
    reconDataMat = Reconstruction(lowDataMat, V, meanVal)
    return reconDataMat

def get_opcodes_sequence(asm_code):
    opcodes_sequence = []
    for row in asm_code:
        parts = row.split()#操作码每行只有一个，并且在每行的开头，前后肯定有空格，直接split就行
        
        if parts and parts[0] in OPCODES:
            # print(parts)
            opcodes_sequence.append(parts[0]+" ")
    return opcodes_sequence


# def get_APIs_sequence(asm_code, apis):
#     apis_sequence = []
#     for row in asm_code:
#         for i in range(len(apis)):
#             if apis[i] in row:#字符串匹配
#                 apis_sequence.append(apis[i])
#                 break
#     return apis_sequence

def get_APIs_probability(asm_code, apis):
    apis_values = [0]*len(apis)
    sum = 0
    # api_seq=[]
    for line in asm_code:
        parts = line.split()
        for i, part in enumerate(parts):
            if part == 'call' and i + 1 < len(parts):
                target = parts[i + 1]
                if target.startswith("ds:") or target.startswith("cs:"):
                    target = target[3:]
                try:
                    index = apis.index(target)  # target为API名臣
                    # print("target",target)
                    # api_seq.append(target)
                    apis_values[index] += 1
                    sum += 1
                    break
                except ValueError:
                    break
    if sum!=0:
        # apis_probability = [x/sum for x in apis_values]
        apis_probability = apis_values
        return apis_probability
    else:
        apis_probability=[0]*len(apis)
        return apis_probability

    # for row in asm_code:
    #     for i in range(len(apis)):
    #         if apis[i] in row:#字符串匹配
    #             apis_values[i] += 1
    #             sum += 1
    #             break
    #apis_positive_values = [x for x in apis_values if x > 0]
    # if sum == 0:
    #     return [0]*len(apis),[0]*len(apis)
    # apis_probability = [x/sum for x in apis_values]
    # return apis_probability,apis_values

def generate_bigram(opcodes_sequence):

    # vec = CountVectorizer(min_df=1, ngram_range=(2, 2))
    # X = vec.fit_transform(" ".join(opcodes_sequence))
    # print(vec.get_feature_names(),X)
    list_comb_seq = []
    for i in range(len(opcodes_sequence) - 1):
        list_comb_seq.append(opcodes_sequence[i] + opcodes_sequence[i + 1])  # 构造2-gram
    data_count = Counter(list_comb_seq)
    # print(data_count)
    # matrix_len = len(index_map)
    matrix_len = len(OPCODES)
    matrix = np.zeros((matrix_len, matrix_len))

    # 2-gram共生矩阵
    for i in range(matrix_len):
        for j in range(matrix_len):
            matrix[i][j] = data_count[OPCODES[i] + OPCODES[j]]
            if(matrix[i][j] > 255):
                matrix[i][j] = 255
    # print(matrix.shape,matrix)

    # 标准化
    for i in range(matrix_len):
        for j in range(matrix_len):
            matrix[j][i] = matrix[j][i] / (255 * matrix_len * matrix_len)

    return matrix


    # # 2-gram共生矩阵——按行做标准化处理
    # norm_matrix = np.zeros((matrix_len, matrix_len))
    # for i in range(matrix_len):
    #     for j in range(matrix_len):
    #         if np.sum(matrix[i]) == 0:
    #             norm_matrix[i][j] = 0
    #         else:
    #             norm_matrix[i][j] = matrix[i][j] / np.sum(matrix[i])
    # print(norm_matrix.shape, norm_matrix)

def pre_PCA(API_probability_data):
    pca = PCA(n_components=2)  # 加载PCA算法，设置降维后主成分数目为2
    reduced_x = pca.fit_transform(API_probability_data)  # 对样本进行降维
    print(pca.components_)  # 输出主成分，即行数为降维后的维数，列数为原始特征向量转换为新特征的系数
    print(pca.explained_variance_)  # 新特征 每维所能解释的方差大小
    print(pca.explained_variance_ratio_)  # 新特征 每维所能解释的方差大小在全方差中所占比例



def extract_asm_features(file_path):
    asm_lines=''
    with open(file_path, 'r',errors='ignore') as fasm:
        asm_lines = fasm.readlines()
    opcodes_sequence = get_opcodes_sequence(asm_lines)
    APIs_probability = get_APIs_probability(asm_lines, DEFINED_APIS)
    # print(len(opcodes_sequence),opcodes_sequence)
    #print(len(APIs_probability),len(DEFINED_APIS),APIs_probability)
    return APIs_probability,opcodes_sequence

# 遍历文件夹中的asm文件转化为png（转化为jpg会压缩）
def save_to_img(sample_dir,output_dir):
    for file in os.listdir(sample_dir):
        if file.split('.')[-1]=='asm':
            with open(os.path.join(sample_dir, file), 'r', errors='ignore') as fasm:
                asm_lines = fasm.readlines()
                # APIs_probability, opcodes_sequence = extract_asm_features(filename)
                opcodes_sequence = get_opcodes_sequence(asm_lines)

                matrix = generate_bigram(opcodes_sequence)
                matrix = np.reshape(matrix,(matrix.shape[0],matrix.shape[1],1))
                matrix = np.concatenate([matrix,matrix,matrix],2)

                # print(matrix.shape, matrix)
                im = Image.fromarray(np.uint8(matrix))
                # im.show()
                # TODO 需要改变save的文件名
                im.save(os.path.join(output_dir, file)+'.png')
                # x = Image.open(os.path.join(sample_path, file)+'.png')  # 打开图片
                # data = np.asarray(x)
                # print(data.shape,data)

def save_opcodes_to_pickle(sample_dir, output_path ,out_name):
    ans_list = []
    label_list = []
    ret_lst=[]
    ret_lst1=[]
    ret_lst2=[]
    ret_lst3=[]
    ret_lst4=[]
    ret_lst5=[]
    opcodes_matrix_list=[]
    count=0
    # myclient = pymongo.MongoClient('mongodb://localhost:27017/')
    # mydb = myclient["pe"] # target set
    # mycol = mydb["store_opcodes"] # target collection
    for file in os.listdir(sample_dir):
        if file.split('.')[-1]=='asm':
            full_path=os.path.join(sample_dir, file)
            if count>-1:
            # if count<2500:
            # if count>=2500:
                with open(full_path, 'r', errors='ignore') as fasm:
                    # if count<5000:
                    #     pass
                    # else:

                    asm_lines = fasm.readlines()
                    # APIs_probability, opcodes_sequence = extract_asm_features(filename)
                    # print("opcodes处理" + file)
                    opcodes_sequence = get_opcodes_sequence(asm_lines) # seq
                    # ret_lst.append((full_path,opcodes_sequence))
                    list_comb_seq=[]
                    for i in range(len(opcodes_sequence) - 1):
                        list_comb_seq.append(opcodes_sequence[i].strip(" ") + opcodes_sequence[i + 1].strip(" "))  # 构造2-gram
                    data_count = Counter(list_comb_seq)
                    # print(data_count)
                    # print(data_count)
                    # matrix_len = len(index_map)
                    matrix_len = len(OPCODES)
                    matrix = np.zeros((matrix_len, matrix_len))

                    # 2-gram共生矩阵
                    for i in range(matrix_len):
                        for j in range(matrix_len):
                            matrix[i][j] = data_count[OPCODES[i] + OPCODES[j]]
                    matrix_1d=list(matrix.flatten())
                    opcodes_matrix_list.append(matrix_1d)

                    # raw_filename=full_path.split(os.sep)[-1].strip(".asm")
                    # raw_filepath=full_path.strip(".asm")
                    # fp=open(raw_filepath,'rb')
                    # data = fp.read()
                    # fp.close()
                    # file_md5= hashlib.md5(data).hexdigest()
                    # file_dict={
                    # "filename":raw_filename,
                    # "file_md5":file_md5,
                    # "opcodes":opcodes_sequence
                    #                 }
                    # mycol.insert_one(file_dict)


                    
                    # if count<5000:
                    #     # ret_lst1.append((full_path,opcodes_sequence))
                    #     # with open(os.path.join(output_path,out_name+"1"), 'wb') as file:
                    #     #     pickle.dump(ret_lst1, file)
                    #     pass
                    # # elif count>=5000 and count<10000:
                    # else:
                    #     with open(os.path.join(output_path,out_name+".txt"), 'a') as file:
                    #         file.writelines(full_path+" ")
                    #         file.writelines(opcodes_sequence)
                    #         file.writelines("\n")
                    #         file.close()
                    #     ret_lst2.append((full_path,opcodes_sequence))
                    #     ret_lst1.clear()
                    #     with open(os.path.join(output_path,out_name+"2"), 'wb') as file:
                    #         pickle.dump(ret_lst2, file)
                    # elif count>=10000 and count<15000:
                    #     ret_lst3.append((full_path,opcodes_sequence))
                    #     ret_lst2.clear()
                    #     with open(os.path.join(output_path,out_name+"3"), 'wb') as file:
                    #         pickle.dump(ret_lst3, file)
                    # elif count>=15000 and count<20000:
                    #     ret_lst4.append((full_path,opcodes_sequence))
                    #     ret_lst3.clear()
                    #     with open(os.path.join(output_path,out_name+"4"), 'wb') as file:
                    #         pickle.dump(ret_lst4, file)
                    # else:
                    #     ret_lst5.append((full_path,opcodes_sequence))
                    #     ret_lst4.clear()
                    #     with open(os.path.join(output_path,out_name+"5"), 'wb') as file:
                    #         pickle.dump(ret_lst5, file)
                # # print(opcodes_sequence)
                # matrix = generate_bigram(opcodes_sequence)
                # ans_list.append(matrix)
                # if is_black:
                #     label_list.append(1)
                # else:
                #     label_list.append(0)
                # # reconImage = PCA(matrix, 30)
                # # print(reconImage)
                fasm.close()
                
                if count%100==0:
                    print("OPCODES ",count)
            count+=1
    with open(os.path.join(output_path,out_name), 'wb') as file:
        pickle.dump( np.array(opcodes_matrix_list), file)
        file.close()
    

    # ans_array = np.array(ans_list)
    # label_array = np.array(label_list)
    # return ret_lst


    # print(ans_array.shape,ans_array)
    # if is_black:
    #     with open(os.path.join(output_path,"black_opcodes_feature.pkl"), 'wb') as file:
    #         pickle.dump(ans_array, file)
    #     with open(os.path.join(output_path,"black_opcodes_label.pkl"), 'wb') as file:
    #         pickle.dump(label_array, file)
    # else:
    #     with open(os.path.join(output_path,"white_opcodes_feature.pkl"), 'wb') as file:
    #         pickle.dump(ans_array, file)
    #     with open(os.path.join(output_path,"white_opcodes_label.pkl"), 'wb') as file:
    #         pickle.dump(label_array, file)

    # with open('array.pkl', 'rb') as file:
    #     array_1 = pickle.load(file)

def save_API_to_pickle(sample_dir, output_path ,out_name):
    ans_list = []
    count_list = []
    label_list = []
    ret_lst=[]
    api_prob_list=[]
    count=0
    for file in os.listdir(sample_dir):
        if file.split('.')[-1]=='asm':
            full_path=os.path.join(sample_dir, file)
            with open(full_path, 'r', errors='ignore') as fasm:
                asm_lines = fasm.readlines()
                # APIs_probability, opcodes_sequence = extract_asm_features(filename)
                # print("API处理" + file)
                apis_probability = get_APIs_probability(asm_lines,MODIFIED_APIS)
                api_prob_list.append(apis_probability)
                # ret_lst.append((full_path,api_seq))
                count+=1
                if count%100==0:
                    print("API",count)
                # ans_list.append(API_probs)
                # count_list.append(API_count)
                # if is_black:
                #     label_list.append(1)
                # else:
                #     label_list.append(0)
                # reconImage = PCA(matrix, 30)
                # print(reconImage)
    with open(os.path.join(output_path,out_name), 'wb') as file:
        pickle.dump(np.array(api_prob_list), file)

    # ans_array = np.array(ans_list)
    # count_array = np.array(count_list)
    # label_array = np.array(label_list)


    # print(ans_array.shape,ans_array)
    # if is_black:
    #     with open(os.path.join(output_path,"black_API_feature.pkl"), 'wb') as file:
    #         pickle.dump(ans_array, file)
    #     with open(os.path.join(output_path,"black_API_label.pkl"), 'wb') as file:
    #         pickle.dump(label_array, file)
    #     with open(os.path.join(output_path,"black_API_count.pkl"), 'wb') as file:
    #         pickle.dump(count_array, file)
    # else:
    #     with open(os.path.join(output_path,"white_API_feature.pkl"), 'wb') as file:
    #         pickle.dump(ans_array, file)
    #     with open(os.path.join(output_path,"white_API_label.pkl"), 'wb') as file:
    #         pickle.dump(label_array, file)
    #     with open(os.path.join(output_path,"white_API_count.pkl"), 'wb') as file:
    #         pickle.dump(count_array, file)

    # with open('array.pkl', 'rb') as file:
    #     array_1 = pickle.load(file)

def api_to_list(path):
    with open(path, 'rb') as file:
        sample_api_info=pickle.load(file)
        sample_dict=dict()
        sample_name_lst=[]

        for sample_info in sample_api_info:
            sample_path=sample_info[0]
            origin_api_lst=sample_info[1]
            sample_name_lst.append(sample_path)

            api_dict=dict()
            for item in MODIFIED_APIS:
                api_dict[item]=0
            api_set=set(origin_api_lst)
            for item in api_set:
                api_dict[item]=1
                # print(item)
            sample_dict[sample_path]=api_dict

        if ".pkl" in path :
            path=path.strip(".pkl")
        with open(path+".json","w") as f:
            json.dump(sample_dict,f)
            f.close()
        with open(path+"_paths.pkl","wb") as f:
            pickle.dump(sample_name_lst,f)
        return sample_dict

def opcodes_to_gram(path):
    with open(path, 'rb') as file:
        sample_opcode_info=pickle.load(file)


        sample_dict=dict()
        for sample_info in sample_opcode_info:
            sample_path=sample_info[0]
            origin_opcodes_lst=sample_info[1]

            opcode_pair_dict=dict()
            for item_pre in OPCODES:
                for item_aft in OPCODES:
                    opcode_pair_dict[item_pre+"_"+item_aft]=0

            op_set=set()
            opcodes_len=len(origin_opcodes_lst)
            for i in range(opcodes_len-1):
                pair_name=origin_opcodes_lst[i]+"_"+origin_opcodes_lst[i+1]
                op_set.add(pair_name)
            for item in op_set:
                opcode_pair_dict[item]=1
            sample_dict[sample_path]=opcode_pair_dict
        
        if ".pkl" in path :
            path=path.strip(".pkl")

        with open(path+".json","w") as f:
            json.dump(sample_dict,f)
            f.close()
        return sample_dict
    
def data_to_matrix():
    api_path=os.path.join(r"C:\Users\Cony\Desktop\sample_api_opcodes","win10_64_api")+".json"
    op_path=os.path.join(r"C:\Users\Cony\Desktop\sample_api_opcodes","win10_64_opcodes")+".json"
    name_path=os.path.join(r"C:\Users\Cony\Desktop\sample_api_opcodes","win10_64_api")+"_paths.pkl"

    f=open(name_path,"rb")
    name_lst=pickle.load(f)
    f.close()

    f=open(api_path,"rb")
    api_dict=json.load(f)
    f.close()

    f=open(op_path,"rb")
    op_dict=json.load(f)
    f.close()
    

    length=len(name_lst)
    api_feature_matrix=[]
    op_feature_matrix=[]
    for i in range(length):
        api_feature_lst=[]
        op_feature_lst=[]
        path_name=name_lst[i]
        api_set=api_dict[path_name]
        for value in api_set.values():
            api_feature_lst.append(value)
        op_set=op_dict[path_name]
        for value in op_set.values():
            op_feature_lst.append(value)
        api_feature_matrix.append(api_feature_lst)
        op_feature_matrix.append(op_feature_lst)
    
    save_path=os.path.join(r"C:\Users\Cony\Desktop\sample_api_opcodes","win10_64_api")+"_features.pkl"
    f=open(save_path,"wb")
    features_matrix=numpy.array(api_feature_matrix)
    pickle.dump(features_matrix,f)
    f.close()

    save_path=os.path.join(r"C:\Users\Cony\Desktop\sample_api_opcodes","win10_64_op")+"_features.pkl"
    f=open(save_path,"wb")
    features_matrix=numpy.array(op_feature_matrix)
    pickle.dump(features_matrix,f)
    f.close()
        
def feature_PCA():
    f=open(r"C:\Users\Cony\Desktop\sample_api_opcodes\win10_64_api_features.pkl","rb")
    api_features_matrix=pickle.load(f)
    f.close()
    # features_matrix=numpy.array(features)
    sample_num,api_feature_num=api_features_matrix.shape
    print(sample_num,api_feature_num)
    target_dim=int(api_feature_num/2)
    if target_dim>=sample_num:
        target_dim=sample_num
    pca = sklearn.decomposition.PCA(n_components=target_dim)
    api_new_features = pca.fit_transform(api_features_matrix)
    print(api_new_features.shape)
    save_path=os.path.join(r"C:\Users\Cony\Desktop\sample_api_opcodes","win10_64_api_pca")+"_features.pkl"
    f=open(save_path,"wb")
    features_matrix=numpy.array(api_new_features)
    pickle.dump(features_matrix,f)
    f.close()

    
    f=open(r"C:\Users\Cony\Desktop\sample_api_opcodes\win10_64_op_features.pkl","rb")
    op_features_matrix=pickle.load(f)
    f.close()
    # features_matrix=numpy.array(features)
    sample_num,op_feature_num=op_features_matrix.shape
    print(sample_num,op_feature_num)
    target_dim=int(op_feature_num/2)
    if target_dim>=sample_num:
        target_dim=sample_num
    pca = sklearn.decomposition.PCA(n_components=target_dim)
    op_new_features = pca.fit_transform(op_features_matrix)
    print(op_new_features.shape)
    save_path=os.path.join(r"C:\Users\Cony\Desktop\sample_api_opcodes","win10_64_op_pca")+"_features.pkl"
    f=open(save_path,"wb")
    features_matrix=numpy.array(op_new_features)
    pickle.dump(features_matrix,f)
    f.close()

def copulas_aug():
    f=open(r"C:\Users\Cony\Desktop\sample_api_opcodes\win10_64_api_pca_features.pkl","rb")
    api_features_matrix=pickle.load(f)
    f.close()

    # Fit a gaussian copula to the data
    copula = VineCopula('regular')
    copula.fit(pandas.DataFrame(api_features_matrix))

    # Sample synthetic data
    synthetic_data = copula.sample(len(api_features_matrix))

    save_path=os.path.join(r"C:\Users\Cony\Desktop\sample_api_opcodes","win10_64_api_syn")+"_features.pkl"
    f=open(save_path,"wb")
    pickle.dump(synthetic_data,f)
    f.close()
    print(synthetic_data.shape)


    # # Plot the real and the synthetic data to compare
    # plt=compare_3d(api_features_matrix, synthetic_data)
    # plt.show()

def dt():
    X = [[0, 0], [2, 2]]
    y = [0.5, 2.5]
    clf = tree.DecisionTreeRegressor()
    clf = clf.fit(X, y)
    clf.predict([[1, 1]])

if __name__ == "__main__":
    print("count of opcodes:",len(OPCODES))
    print("count of apis:",len(MODIFIED_APIS))

    black_sample_dir = r"E:\Reproduction\dataset_raw\black_asm"
    white_sample_dir = r"E:\Reproduction\dataset_raw\WIN10-7\7-10"
    output_path = r"E:\Reproduction\dataset_raw"
    # save_API_to_pickle(white_sample_dir,output_path,out_name="store_ext_cuckoo_api.pkl")
    # sample_dir=r"E:\test\sum_m"

    

    # data_to_matrix()
    # feature_PCA()
    # copulas_aug()

    # api_to_list(os.path.join(r"C:\Users\Cony\Desktop\sample_api_opcodes","win10_64_api"))
    # opcodes_to_gram(os.path.join(r"C:\Users\Cony\Desktop\sample_api_opcodes","win10_64_opcodes"))


    # print(res)
    # print(res)
    # save_API_to_pickle(sample_dir,output_path,out_name="store_api")
    # save_opcodes_to_pickle(sample_dir,output_path,out_name="store_opcodes")

    # black_opcodes_feature = ".\\IDA_dis\\black_opcodes_feature.pkl"
    # black_opcodes_label = ".\\IDA_dis\\black_opcodes_label.pkl"
    # white_opcodes_feature = ".\\IDA_dis\\white_opcodes_feature.pkl"
    # white_opcodes_label = ".\\IDA_dis\\white_opcodes_label.pkl"
    #
    # black_API_feature = ".\\IDA_dis\\black_API_feature.pkl"
    # black_API_label = ".\\IDA_dis\\black_API_label.pkl"
    # white_API_feature = ".\\IDA_dis\\white_API_feature.pkl"
    # white_API_label = ".\\IDA_dis\\white_API_label.pkl"

    save_API_to_pickle(white_sample_dir,output_path,out_name="win7-10_api_red.pkl")
    # save_API_to_pickle(white_sample_dir,output_path,out_name="black_cuckoo_api.pkl")
    # save_API_to_pickle(white_sample_dir,output_path,0)

    save_opcodes_to_pickle(white_sample_dir, output_path,out_name="win7-10_opcodes_red.pkl")
    # save_opcodes_to_pickle(white_sample_dir, output_path, 0)


