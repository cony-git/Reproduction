import json
import os
import pickle

b_corpus = []  # 词库list
m_corpus=[]


def strings(lst, target_dir):  # 提取string存储到txt
    # for _, _, files in os.walk(dir_file):
    for filename in lst:
        print(filename)
        cmd = "strings " + filename#+ dir_file + filename
        r = os.popen(cmd)
        info = r.readlines()
        if "/" in filename:
            filename=filename.split("/")[-1]
            target_file = target_dir +"/"+ filename + ".txt"
            with open(target_file, "w+") as f:
                for line in info:
                    words = line.split()
                    for word in words:
                        f.write(word + "\n")
        # for line in info:
        #     print(line, end="")


# def get_corpus(b_txt_file_dir,m_txt_file_dir, corpusfile):  # 生成语料库到txt
#     for _, _, files in os.walk(b_txt_file_dir):
#         for filename in files:
#             filepath = b_txt_file_dir + filename
#             print(filepath)
#             with open(filepath, "r") as f:  # 读取样本的string.txt
#                 word_freq = {}
#                 words = f.read().split("\n")
#                 for word in words:
#                     if len(word) >= 3:  # 词长大于3
#                         word_freq[word] = word_freq.get(word, 0) + 1
#                 if word_freq:
#                     sorted_word_freq = sorted(word_freq.items(),
#                                               key=lambda v: v[1],
#                                               reverse=True)  # 词频排序
#                     for item in sorted_word_freq[:1500]:  # 获取top 1500的单词
#                         # print(item[0], item[1])
#                         b_corpus.append(item[0])

#     for _, _, files in os.walk(m_txt_file_dir):
#         for filename in files:
#             filepath = m_txt_file_dir + filename
#             print(filepath)
#             with open(filepath, "r") as f:  # 读取样本的string.txt
#                 word_freq = {}
#                 words = f.read().split("\n")
#                 for word in words:
#                     if len(word) >= 3:  # 词长大于3
#                         word_freq[word] = word_freq.get(word, 0) + 1
#                 if word_freq:
#                     sorted_word_freq = sorted(word_freq.items(),
#                                               key=lambda v: v[1],
#                                               reverse=True)  # 词频排序
#                     for item in sorted_word_freq[:1500]:  # 获取top 1500的单词
#                         # print(item[0], item[1])
#                         m_corpus.append(item[0])


#     all_corpus=(set(b_corpus),set(m_corpus))
#     set_corpus = set(b_corpus+m_corpus)
#     with open ("corpus.pkl", 'wb') as f: #打开文件 
#         pickle.dump(all_corpus,f)
#         f.close()
#     with open(corpusfile, "w+") as f:
#         for word in set_corpus:
#             f.write(word + " ")
#     # pass


def get_corpus(txt_file_dir, corpusfile,flag):  # 生成语料库到txt
    word_freq = {}
    for _, _, files in os.walk(txt_file_dir):
        for filename in files:
            filepath = txt_file_dir + filename
            print(filepath)
            with open(filepath, "r") as f:  # 读取样本的string.txt
                words = f.read().split("\n")
                for word in words:
                    if len(word) >= 3:  # 词长大于3
                        word_freq[word] = word_freq.get(word, 0) + 1
    # print(word_freq)
    if word_freq:
        sorted_word_freq = sorted(word_freq.items(),
                                  key=lambda v: v[1],
                                  reverse=True)  # 词频排序
        for item in sorted_word_freq[:1500]:  # 获取top 1500的单词
            print(item[0], item[1])
            if flag==0:
                b_corpus.append(item[0])
            else:
                m_corpus.append(item[0])
    if flag==0:
        with open(corpusfile, "w+") as f:
            for word in b_corpus:
                f.write(word + " ")
    else:
        with open(corpusfile, "w+") as f:
            for word in m_corpus:
                f.write(word + " ")
    # pass

def get_files():
    b_path="/mnt/e/test/sum_m"
    b_lst=[]
    count=0
    for item in os.listdir(b_path):
        full_path=os.path.join(b_path,item)
        if count<16560:
            if "." in full_path:
                suffix=full_path.split(".")[-1]
                if suffix.upper()=="DLL" or suffix.upper()=="EXE":
                    b_lst.append(full_path)
                    count+=1
                    print(count)
        else:
            break
    with open ("good.pkl", 'wb') as f: #打开文件 
        pickle.dump(b_lst,f)
        f.close()

    m_path="/mnt/e/viruses-2010-05-18/viruses-2010-05-18"
    m_lst=[]
    count=0
    for item in os.listdir(m_path):
        full_path=os.path.join(m_path,item)
        if count<23131:
            # if "." in full_path:
            #     suffix=full_path.split(".")[-1]
            #     if suffix.upper()=="DLL" or suffix.upper()=="EXE":
            m_lst.append(full_path)
            count+=1
            print(count)
        else:
            break
    with open ("bad.pkl", 'wb') as f: #打开文件 
        pickle.dump(m_lst,f)
        f.close()

def process_all_strings():
    with open ("good.pkl", 'rb') as f:
        b_lst=pickle.load(f)
    with open ("bad.pkl", 'rb') as f:
        m_lst=pickle.load(f)
    strings(b_lst,"./b_string")
    strings(m_lst,"./m_string")

# get_files()
# process_all_strings()
# strings("/Users/shilin/Desktop/samples/malware/", "/Users/shilin/Desktop/samples/string_malware/")
# get_corpus("/mnt/e/NLP_STRING/b_string/", "/mnt/e/NLP_STRING/m_string/","/mnt/e/NLP_STRING/corpus.txt")
get_corpus("/mnt/e/NLP_STRING/b_string/","/mnt/e/NLP_STRING/b_corpus.txt",0)
get_corpus("/mnt/e/NLP_STRING/m_string/","/mnt/e/NLP_STRING/m_corpus.txt",1)



# print("_________")
# set_corpus = set(corpus)
# print(set_corpus)
