import math
import numpy as np
from shapelet_candidates import FastShapeletCandidates
from pyts.transformation import ShapeletTransform
from sklearn.model_selection import train_test_split
from sklearn import tree

def read_time_series(target_file_path):
    ts_in_file=[]
    with open(target_file_path,"rb") as f:
        
        while True:
            one_byte_lst=[]
            empty_flag=0
            for i in range(256):
                one_byte=f.read(1)
                if not one_byte:
                    empty_flag=1
                    one_byte_lst+=[0]*(256-i)
                    break
                else:
                    # print(one_byte)
                    one_int=int.from_bytes(one_byte, "big")
                    one_byte_lst.append(one_int)
                    # print(one_int)
            ts_in_file.append(one_byte_lst)
            if empty_flag==1:
                break
        
        f.close()
    return ts_in_file
    


def cal_entropy(ts_lst):
    index_lst=list(range(256))
    count_dict=dict()
    for item in index_lst:
        count_dict[item]=0
    for item in ts_lst:
        # print(item)
        count_dict[item]+=1
    # print(count_dict)
    entropy=0
    for key in count_dict:
        item=count_dict[key]
        item_prob=item/256
        if item==0:
            entropy+=0
        else:
            entropy+=item_prob*math.log(item_prob)
    entropy=-entropy
    return entropy
        
def generate_entropy_ts(ts_in_file):
    entropy_ts=[]
    for item in ts_in_file:
        current_entropy=cal_entropy(item)
        entropy_ts.append(current_entropy)
    return entropy_ts

def data_preprocessing(entropy_ts):
    train_data=[entropy_ts]*200
    train_data=np.array(train_data)
    # np.random.shuffle(train_data)
    train_label=np.random.randint(0,high=1,size=(200,1))
    print(train_data.shape)
    print(train_data.dtype)
    print(train_label.shape)
    print(train_label.dtype)
    print(train_data[0:6,0:6])
    return train_data,train_label

def shapeletML(train_data,train_label):
    # 训练
    
    x_train,x_test,y_train,y_test=train_test_split(train_data,train_label,test_size=0.1)
    st = ShapeletTransform(n_shapelets=100, window_sizes=[5],n_jobs=8)
    print(0)
    x_train_new = st.fit_transform(x_train,y_train)
    print(1)
    # 转换
    x_test_new = st.transform(x_test)
    print(2)

    
    
    clf = tree.DecisionTreeClassifier() #实例化模型对象
    clf = clf.fit(x_train_new,y_train) #用训练集数据训练模型
    result = clf.score(x_test_new,y_test) #对我们训练的模型精度进行打分
    print(result)

if __name__=="__main__":
    file_path=r"C:\Program Files\Windows Media Player\wmplayer.exe"
    ts_in_file=read_time_series(file_path)
    entropy_ts=generate_entropy_ts(ts_in_file)
    print(0)
    train_data,train_label=data_preprocessing(entropy_ts)
    print(1)
    shapeletML(train_data,train_label)
    # print(entropy_ts)
    # print(len(entropy_ts))

    # lst=[entropy_ts]*100



    # data=np.array(lst)
    # print(data,data.shape)
    # # data = np.load(open('fordA_sample.npy', 'rb'))
    # # print(data[0:5,0:5])
    # # print(data.shape)
    # # length_ts * 0.05 + 2 as the number of LFDPs. Empirically tested to
    # # be good value according to the paper
    # len_ts = data.shape[1]
    # print(len_ts)
    # # must be an integer
    # n_lfdp = int(len_ts * 0.05 + 2)
    # print(n_lfdp)
    # std_split = 2
    # shapelet_selection = FastShapeletCandidates(n_lfdp)
    # print(shapelet_selection)
    # shapelet_candidates = shapelet_selection.transform(data)
    # print(shapelet_candidates)
    # t=np.array(shapelet_candidates)
    # print(t.shape)

    # n = 10
    # fig, axes = pyplot.subplots(n, 1)
    # fig.suptitle('Shapelet Candidates (Note that the candidates have different lengths)')
    # fig.set_size_inches(23, n * 2)
    # for i, x in enumerate(random.sample(shapelet_candidates, 10)):
    #     axes[i].plot(x) if n > 1 else axes.plot(x)
    # pyplot.show()
