import tensorflow as tf
from sklearn import preprocessing
import scipy.stats as st
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import pickle
# from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from keras.utils import np_utils

import matplotlib.pyplot as plt
import keras
from keras.callbacks import ReduceLROnPlateau
import random
import time
import math
# from keras.backend import clear_session
from sklearn.model_selection import KFold

import time
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))

from keras.backend import set_session
from keras.backend import clear_session
from keras.backend import get_session
import tensorflow as tf

white_fea1=r"win10+7_all_fea.pkl"
white_fea2=r"win7-10_all_fea.pkl"
black_fea=r"black_all_fea.pkl"
test_path2=r"win10-7_all_fea.pkl"

# white_fea1=r"even_all_fea.pkl"
# # white_fea2=r"win7-10_all_fea.pkl"
# black_fea=r"black_all_fea.pkl"
# test_path2=r"win10_64_all_fea.pkl"

proj_label="7-10"
seed_label="_seed1"

# seed1,seed2,seed3,seed4=49692,43541,44669,6802

seed1,seed2,seed3,seed4=49692,43541,44669,37704
# seed1,seed2,seed3,seed4=11483, 49492, 9158, 864
# seed1,seed2,seed3,seed4=41347, 58762, 13374, 5752
# seed1,seed2,seed3,seed4=1352, 40429, 61975, 59042
# seed1,seed2,seed3,seed4=17939, 64633, 56958, 62206


def mean2(x):
    y = np.sum(x) / np.size(x)
    return y

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)
 
    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum())
    return r

def get_dist(all_features_data):

    # GET DIST INFO

    print(all_features_data.shape)
    sample_count=all_features_data.shape[0]
    features_count=all_features_data.shape[1]

    dis_lst=[]
    for i in range(features_count):
        column_data=all_features_data[:,[i]]
        # print(column_data)
        data=column_data

        preprocessing.scale(data)
        # print(pd.DataFrame(data).describe())
        mean_val=np.mean(data)
        std_val=np.std(data, ddof = 1 )
        max_val=data.max()
        min_val=data.min()
        # counts_val=np.bincount(data)
        value_lst=list(data)
        mode_val=st.mode(value_lst)[0]
        skew_val=st.skew(value_lst) # 计算偏度
        kurt_val=st.kurtosis(value_lst) # 计算峰度

        # quar_1=np.percentile(data,25)
        median_val=np.median(data)
        # quar_3=np.percentile(data,75)
        perc_1=np.percentile(data,10)
        perc_2=np.percentile(data,20)
        perc_3=np.percentile(data,30)
        perc_4=np.percentile(data,40)
        perc_6=np.percentile(data,60)
        perc_7=np.percentile(data,70)
        perc_8=np.percentile(data,80)
        perc_9=np.percentile(data,90)

        dis_lst.append((mean_val,max_val,min_val,mode_val,perc_1,perc_2,perc_3,perc_4,median_val,perc_6,perc_7,perc_8,perc_9,std_val,skew_val,kurt_val)) 

    dist_arr=np.array(dis_lst)
    print(dist_arr.shape)

    return dist_arr

# Reset Keras Session
def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()
 
    try:
        del classifier # this is from global space - change this as you need
    except:
        pass
 
    print(gc.collect()) # if it does something you should see a number as output
 
    # use the same config as you used to create the session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tf.compat.v1.Session(config=config))

# 绘图
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}
        self.accuracy = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('accuracy'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_accuracy'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('accuracy'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_accuracy'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        #创建一个图
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')#plt.plot(x,y)，这个将数据画成曲线
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)#设置网格形式
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')#给x，y轴加注释
        plt.legend(loc="upper right")#设置图例显示位置
        plt.show()

def give_p(d):
    # print("D", d)
    sum = np.sum(d)
    sum_85 = 0.95 * sum
    temp = 0
    p = 0
    while temp < sum_85:
        temp += d[p]
        p += 1
    return p

def reduce_dim(img):

    images = img
    mean_face = np.mean(images, 0)
    images_mean_subtracted = images - mean_face


    no_of_images = images.shape[0]
    mat_height = images.shape[1]
    mat_width = images.shape[2]
    g_t = np.zeros((mat_height, mat_height))
    h_t = np.zeros((mat_width, mat_width))

    for i in range(no_of_images):
        temp = np.dot(images_mean_subtracted[i].T, images_mean_subtracted[i])
        g_t += temp
        h_t += np.dot(images_mean_subtracted[i], images_mean_subtracted[i].T)

    g_t /= no_of_images
    h_t /= no_of_images

    #For G_T
    d_mat, p_mat = np.linalg.eig(g_t)
    p_1 = give_p(d_mat)
    new_bases_gt = p_mat[:, 0:p_1]

    #For H_T
    d_mat, p_mat = np.linalg.eig(h_t)
    p_2 = give_p(d_mat)
    new_bases_ht = p_mat[:, 0:p_2]


    new_coordinates_temp = np.dot(images, new_bases_gt)

    new_coordinates = np.zeros((no_of_images, p_2, p_1))

    for i in range(no_of_images):
        new_coordinates[i, :, :] = np.dot(new_bases_ht.T, new_coordinates_temp[i])

    return new_coordinates


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
    # print("降维后的特征个数：" + str(k) + "\n")
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

def PCA_2(datas, k):
# PCA算法
    # print(datas.shape)
    final_arr=[]
    for data in datas:
        # print(data.shape)
        dataMat = np.float64(np.mat(data))
        # dataMat = np.float64(np.mat(data[0]))
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
        final_arr.append(reconDataMat)
    
    return np.asarray(final_arr,dtype=np.float64)


def downscale(black_opcodes_feature,white_opcodes_feature,black_API_feature,white_API_feature,black_label, white_label):
    ## -----------------------OPCODES------------------------------------

    # merge black and white opcodes
    opcodes_data = np.concatenate((black_opcodes_feature, white_opcodes_feature), axis=0)
    # print(self.opcodes_data.shape)
    reshape_dim=int(math.sqrt(opcodes_data.shape[1]))
    # print(reshape_dim)
    opcodes_data=opcodes_data.reshape((opcodes_data.shape[0],reshape_dim,reshape_dim))

    # 2d pca
    tmp_pca=PCA_2(opcodes_data,100)

    print("2d_pca:",tmp_pca.shape)
    
    # adjust opcodes dimension to fit model input
    opcodes_data = np.expand_dims(tmp_pca,axis=-1)

    ## -----------------------API------------------------------------
    # merge black and white API
    API_data = np.concatenate((black_API_feature,white_API_feature), axis=0)
    
    print("pca:begin!")
    # PCA transformer
    pca = PCA(n_components=int(API_data.shape[1]/5))  # 加载PCA算法，设置降维后主成分数目为2 
    API_data=pca.fit_transform(API_data)
    print("pca effectiveness:",pca.explained_variance_ratio_.sum())

    # the api and opcodes label are the same
    label = np.concatenate((black_label, white_label), axis=0)

    return opcodes_data,API_data,label


def get_delta(info,w_matrix,b_matrix,proj_label,detail_str,seed_label):
    white_api_matrix=w_matrix[:,:1520]
    white_op_matrix=w_matrix[:,1520:]
    black_api_matrix=b_matrix[:,:1520]
    black_op_matrix=b_matrix[:,1520:]

    # print(white_api_matrix[:50,:50])

    white_api_matrix[white_api_matrix>0]=1
    white_op_matrix[white_op_matrix>0]=1
    black_api_matrix[black_api_matrix>0]=1
    black_op_matrix[black_op_matrix>0]=1

    # print(white_api_matrix[:50,:50])

    white_api_exi=np.sum(white_api_matrix,axis = 0)
    white_op_exi=np.sum(white_op_matrix,axis=0)
    black_api_exi=np.sum(black_api_matrix,axis=0)
    black_op_exi=np.sum(black_op_matrix,axis=0)

    # print(white_api_exi[:50])

    white_api_exi[white_api_exi>0]=1
    white_op_exi[white_op_exi>0]=1
    black_api_exi[black_api_exi>0]=1
    black_op_exi[black_op_exi>0]=1

    # print("white:",white_api_exi[:50])
    # print("black:",black_api_exi[:50])

    tmp_api=np.array([0]*white_api_exi.shape[0])
    tmp_op=np.array([0]*black_op_exi.shape[0])
    for i in range(tmp_api.shape[0]):
        if white_api_exi[i]==1 and black_api_exi[i]==1:
            tmp_api[i]=2
        elif white_api_exi[i]==1 and black_api_exi[i]==0:
            tmp_api[i]=1
        elif white_api_exi[i]==0 and black_api_exi[i]==1:
            tmp_api[i]=-1
        elif white_api_exi[i]==0 and black_api_exi[i]==0:
            tmp_api[i]=0
    for i in range(tmp_op.shape[0]):
        if white_op_exi[i]==1 and black_op_exi[i]==1:
            tmp_op[i]=2
        elif white_op_exi[i]==1 and black_op_exi[i]==0:
            tmp_op[i]=1
        elif white_op_exi[i]==0 and black_op_exi[i]==1:
            tmp_op[i]=-1
        elif white_op_exi[i]==0 and black_op_exi[i]==0:
            tmp_op[i]=0
    # delta_api=white_api_exi-black_api_exi
    # delta_op=white_op_exi-black_op_exi

    api_delta_path="./delta/api_"+proj_label+"_"+detail_str+seed_label+".pkl"
    op_delta_path="./delta/op_"+proj_label+"_"+detail_str+seed_label+".pkl"
    with open(api_delta_path,"wb") as f:
        pickle.dump(tmp_api,f)
    with open(op_delta_path,"wb") as f:
        pickle.dump(tmp_op,f)

    # import matplotlib.pyplot as plt
    # import seaborn as sns

    # plt.figure(figsize = (8,6))
    # cmap = sns.diverging_palette(200,20,sep=20,as_cmap=True)
    # sns.heatmap(delta_op.reshape(delta_op.shape[0],1), cmap=cmap)#热力图
    # plt.show()
    
    co1 = np.corrcoef(white_api_exi, black_api_exi)
    co2 = np.corrcoef(white_op_exi, black_op_exi)


    # api_nz_indexs=np.transpose(np.nonzero(delta_api))
    # op_nz_indexs=np.transpose(np.nonzero(delta_op))
    # print(info,api_nz_indexs.shape[0],co1[1,0],op_nz_indexs.shape[0],co2[1,0],api_nz_indexs.shape[0]+op_nz_indexs.shape[0],api_nz_indexs.shape[0]/1520+op_nz_indexs.shape[0]/(159*159),co1[1,0]+co2[1,0])
    # return api_nz_indexs.shape[0],op_nz_indexs.shape[0]
    return

#数据获取及预处理
class DataLoader():
    def __init__(self):
        '''
        文章参数
        Size of input vector 159 ×159
        Size of PCA vector 159 ×159
        '''

        # read white data from files        
        with open(white_fea1,"rb") as file:
            white_fea_all1=pickle.load(file)
            self.white_API_feature1=white_fea_all1[:,0:1520]
            self.white_opcodes_feature1=white_fea_all1[:,1520:]
        with open(white_fea2,"rb") as file:
            white_fea_all2=pickle.load(file)
            self.white_API_feature2=white_fea_all2[:,0:1520]
            self.white_opcodes_feature2=white_fea_all2[:,1520:]
        
        # read black data
        with open(black_fea,"rb") as file:
            black_fea_all=pickle.load(file)
            self.black_API_feature=black_fea_all[:,0:1520]
            self.black_opcodes_feature=black_fea_all[:,1520:]

        # merge white cross and delta data
        self.white_API_feature=np.concatenate((self.white_API_feature1,self.white_API_feature2),axis=0)
        self.white_opcodes_feature=np.concatenate((self.white_opcodes_feature1,self.white_opcodes_feature2),axis=0)
        self.white_len=self.white_API_feature.shape[0]
        self.white_label=np.array([0]*self.white_len)

        # self.white_API_feature=self.white_API_feature1[:3000,:]
        # self.white_opcodes_feature=self.white_opcodes_feature1[:3000,:]
        # self.white_len=self.white_API_feature.shape[0]
        # self.white_label=np.array([0]*self.white_len)[:3000]

        # sample 3k from black data
        black_all_len=self.black_opcodes_feature.shape[0]
        np.random.seed(seed1)
        black_all_shuffle_indexs = np.random.permutation(black_all_len)

        # 提取位于样本集中20%的那个索引值
        black_train_size=2400
        black_test_size = 600

        black_3k_index=black_all_shuffle_indexs[:(black_train_size+black_test_size)]
        black_api_3k=self.black_API_feature[black_3k_index]
        black_op_3k=self.black_opcodes_feature[black_3k_index]
        black_label_3k=np.array([1]*(black_train_size+black_test_size))

        # # 将随机打乱的前若干索引值赋值给测试索引
        # black_test_indexs = black_all_shuffle_indexs[:black_test_size]
        # # 将随机打乱的后若干索引值赋值给训练索引
        # black_train_indexs = black_all_shuffle_indexs[black_test_size:(black_train_size+black_test_size)
        
        self.data_op,self.data_api,self.data_label=downscale(black_op_3k,self.white_opcodes_feature,black_api_3k,self.white_API_feature,black_label_3k,self.white_label)
        print("data shape:",self.data_op.shape,self.data_api.shape,self.data_label.shape)

        # all_shuffle_indexs = np.random.permutation((black_train_size+black_test_size+self.white_len))
        # all_shuffle_indexs=list(range((black_train_size+black_test_size+self.white_len)))
        np.random.seed(seed2)
        black_shuffle_index=np.random.permutation((black_train_size+black_test_size))
        np.random.seed(seed3)
        white_shuffle_indexs=np.random.permutation(self.white_len)
        white_shuffle_indexs=[i+black_train_size+black_test_size for i in white_shuffle_indexs]
        all_shuffle_indexs=np.concatenate((black_shuffle_index,white_shuffle_indexs),axis=0)
        # all_shuffle_indexs=black_shuffle_index+white_shuffle_indexs

        print("white len1:",self.white_len)
        black_train_index=all_shuffle_indexs[:black_train_size]
        print(len(black_train_index))
        black_test_index=all_shuffle_indexs[black_train_size:(black_train_size+black_test_size)]
        print(len(black_test_index))
        white_train_index=all_shuffle_indexs[(black_train_size+black_test_size):(black_train_size+black_test_size+self.white_len-black_test_size)]
        
        print(len(white_train_index))
        white_test_index=all_shuffle_indexs[(black_train_size+black_test_size+self.white_len-black_test_size):]
        print(len(white_test_index))
        
        black_train_op,black_train_api,black_train_label=self.data_op[black_train_index],self.data_api[black_train_index],self.data_label[black_train_index]
        black_test_op,black_test_api,black_test_label=self.data_op[black_test_index],self.data_api[black_test_index],self.data_label[black_test_index]
        white_train_op,white_train_api,white_train_label=self.data_op[white_train_index],self.data_api[white_train_index],self.data_label[white_train_index]
        white_test_op,white_test_api,white_test_label=self.data_op[white_test_index],self.data_api[white_test_index],self.data_label[white_test_index]
        # print(white_test_label)

        self.train_data_op,self.train_data_api,self.train_label=np.concatenate((black_train_op,white_train_op),axis=0),np.concatenate((black_train_api,white_train_api),axis=0),np.concatenate((black_train_label,white_train_label),axis=0)
        self.test_data_op1,self.test_data_api1,self.test_label1=np.concatenate((black_test_op,white_test_op),axis=0),np.concatenate((black_test_api,white_test_api),axis=0),np.concatenate((black_test_label,white_test_label),axis=0)
        print("train data shape:",self.train_data_op.shape,self.train_data_api.shape,self.train_label.shape)
        print("Train data processed!")
        print("first test data shape:",self.test_data_op1.shape,self.test_data_api1.shape,self.test_label1.shape)

        # read second white test data
        with open(test_path2,"rb") as file:
            white_test_fea_all2=pickle.load(file)
            self.white_test_api2=white_test_fea_all2[:,0:1520]
            self.white_test_op2=white_test_fea_all2[:,1520:]
        self.white_test_length2=self.white_test_api2.shape[0]
        print("white_len 2:",self.white_test_length2)
        self.white_test_label2=np.array([0]*self.white_test_length2)

        np.random.seed(seed4)
        white_shuffle_indexs = np.random.permutation(self.white_test_length2)
        white_test_index2=white_shuffle_indexs[:black_test_size]

        self.white_test_api2=self.white_test_api2[white_test_index2]
        self.white_test_op2=self.white_test_op2[white_test_index2] 
        self.white_test_label2=self.white_test_label2[white_test_index2] 

        var_test_mat=np.concatenate((self.white_test_api2,self.white_test_op2),axis=1)
        white_test_index_s=[i-(black_train_size+black_test_size) for i in white_test_index]
        fix_model_mat=np.concatenate((self.white_API_feature[white_test_index_s],self.white_opcodes_feature[white_test_index_s]),axis=1)
        print("test1,2 shape:",var_test_mat.shape,fix_model_mat.shape)

        white_train_index_s=[i-(black_train_size+black_test_size) for i in white_train_index]
        train_mat=np.concatenate((self.white_API_feature[white_train_index_s],self.white_opcodes_feature[white_train_index_s]),axis=1)

        black_train_index_s=[i for i in black_train_index]
        black_mat=np.concatenate((self.black_API_feature[black_train_index_s],self.black_opcodes_feature[black_train_index_s]),axis=1)

        # get_delta("w test2 w test1:",var_test_mat,fix_model_mat,proj_label,"wte12",seed_label)
        # get_delta("w train w test2:",train_mat,var_test_mat,proj_label,"wtr-wte2",seed_label)
        # get_delta("w train w test1:",train_mat,fix_model_mat,proj_label,"wtr-wte1",seed_label)
        # get_delta("b train w test1:",black_mat,fix_model_mat,proj_label,"btr-wte1",seed_label)
        # get_delta("b train w test2:",black_mat,var_test_mat,proj_label,"btr-wte2",seed_label)
        get_delta("b train w train:",black_mat,train_mat,proj_label,"bw-tr",seed_label)
        # t1=get_dist(fix_model_mat)
        # t2=get_dist(var_test_mat)
        # self.c=corr2(t1,t2)
        # print("corr2-seed4:",self.c,seed4)
        self.c=0

        # seed_lst=[random.randint(0,65536) for _ in range(100) ]
        # for seed_t in seed_lst:
            
        #     np.random.seed(seed_t)
        #     white_shuffle_indexs = np.random.permutation(self.white_test_length2)
        #     white_test_index2=white_shuffle_indexs[:black_test_size]

        #     self.white_test_api_t=self.white_test_api2[white_test_index2]
        #     self.white_test_op_t=self.white_test_op2[white_test_index2] 
        #     self.white_test_label_t=self.white_test_label2[white_test_index2] 

        #     var_test_mat=np.concatenate((self.white_test_api_t,self.white_test_op_t),axis=1)
        #     t1=get_dist(fix_model_mat)
        #     t2=get_dist(var_test_mat)
        #     c=corr2(t1,t2)
        #     print("corr2:",seed_t,c)

        ## -----------------------OPCODES------------------------------------
        # print(self.opcodes_data.shape)
        reshape_dim=int(math.sqrt(self.white_test_op2.shape[1]))
        # print(reshape_dim)
        self.white_test_op2=self.white_test_op2.reshape((self.white_test_op2.shape[0],reshape_dim,reshape_dim))
        # 2d pca
        tmp_pca=PCA_2(self.white_test_op2,100)
        print("2d_pca:",tmp_pca.shape)
        # adjust opcodes dimension to fit model input
        self.white_test_op2 = np.expand_dims(tmp_pca,axis=-1)

        ## -----------------------API------------------------------------
        print("pca:begin!")
        # PCA transformer
        pca = PCA(n_components=int(self.white_test_api2.shape[1]/5))  # 加载PCA算法，设置降维后主成分数目为2 
        self.white_test_api2=pca.fit_transform(self.white_test_api2)
        print("pca effectiveness:",pca.explained_variance_ratio_.sum())

        # merge black test and white test2
        self.test_data_op2,self.test_data_api2,self.test_label2=np.concatenate((black_test_op,self.white_test_op2),axis=0),np.concatenate((black_test_api,self.white_test_api2),axis=0),np.concatenate((black_test_label,self.white_test_label2),axis=0)
    
        print("test data shape 2:",self.test_data_op2.shape,self.test_data_api2.shape,self.test_label2.shape)
        print("Test data processed!")


    def get_batch(self, batch_size):
        index = np.random.randint(0, np.shape(self.train_data_op)[0],batch_size)
        #print(index)
        return self.train_data_op[index, :],self.train_data_api[index, :], self.train_label[index]

#模型的构建   tf.keras.Model 和 tf.keras.layers 这里使用的是函数式编程
class HYDRA(tf.keras.Model):
    def __init__(self):
        #关联父类构造函数
        super(HYDRA, self).__init__()
        #卷积层
        '''
        文章中的参数
        Number of convolutional layers 2 
        Number of convolutional cores for each layer 5  这个参数体现在filters？
        Size of each convolutional core 5 ×5
        '''
        self.cnn_conv1 = tf.keras.layers.Conv2D(
            #filters=32,            #卷积层神经元个数？整数，输出空间的维度 （即卷积中滤波器的输出数量）。
            filters=5,  # 卷积层神经元个数？整数，输出空间的维度 （即卷积中滤波器的输出数量）。
            kernel_size=[5, 5],    #一个整数，或者 2 个整数表示的元组或列表， 指明 2D 卷积窗口的宽度和高度。 可以是一个整数，为所有空间维度指定相同的值。
            padding='same',        #是否进行边界填充，Same填充后输入输出维度一样
            activation=tf.nn.relu  #激活函数
        )
        #池化层
        '''
        文章中的参数
        Number of pooling layers 2 
        Number of pooling cores for each layer 5  这个参数体现在哪里？
        Size of each pooling core 2 ×2 
        '''
        self.cnn_pool1 = tf.keras.layers.MaxPooling2D(
            pool_size=[2,2],#最大池化的窗口大小。
            strides=2 #配置卷积核在做池化时移动步幅的大小，分为X、Y两个方向的步幅。数值是包含两个整型元素值的列表或者元组，默认与pool_size相同，即2。2会使得输入张量缩小一半。
        )
        self.cnn_conv2 = tf.keras.layers.Conv2D(
            #filters=64,
            filters=5,
            kernel_size=[5,5],
            padding='same',
            activation=tf.nn.relu
        )
        self.cnn_pool2 = tf.keras.layers.MaxPooling2D(pool_size=[2, 2], strides=2)
        # 拉成一维数据 文章中没说这一层，但需要满足Dense层的输入维度
        #self.flatten = tf.keras.layers.Reshape(target_shape=(7*7*64,))
        # self.cnn_flatten = tf.keras.layers.Reshape(target_shape=(7 * 7 * 5,))
        self.cnn_flatten = tf.keras.layers.Flatten()
        # 全连接层1
        self.cnn_dense1 = tf.keras.layers.Dense(
            #units=1024, #正整数，输出空间维度。与待分类类别数相关
            units=7220,  # 正整数，输出空间维度。与待分类类别数相关
            activation=tf.nn.softmax #激活函数。 若不指定，则不使用激活函数 (即，「线性」激活: a(x) = x)。
        )
        self.cnn_dense2 = tf.keras.layers.Dense(units=2,activation=tf.nn.softmax) #正整数，输出空间维度。与待分类类别数相关。TODO 确定全连接层是否需要激活函数？文章没有指明这一层的输出单元数

        '''
                文章中的参数
                Number of hidden layers 1
                Number of hidden units 50
                '''
        # 全连接层1
        self.bpnn_dense1 = tf.keras.layers.Dense(
            units=50,
            activation=tf.nn.sigmoid  # 激活函数。 若不指定，则不使用激活函数 (即，「线性」激活: a(x) = x)。
        )
        self.bpnn_dense2 = tf.keras.layers.Dense(
            units=2,  # 正整数，输出空间维度。与待分类类别数相关。如果是二分类则为2
            activation=tf.nn.sigmoid  # TODO 这一层也要激活函数吗？
        )

        self.hydra_dense = tf.keras.layers.Dense(units=2,
                                                activation=tf.nn.softmax)  # 正整数，输出空间维度。与待分类类别数相关。TODO 确定全连接层是否需要激活函数？文章没有指明这一层的输出单元数

    # call函数只能一个训练变量，但可以通过元组传递
    def call(self, inputs):
        opcode_inputs, API_inputs = inputs
        print("call",opcode_inputs.shape,API_inputs.shape)
        cnn_conv1 = self.cnn_conv1(opcode_inputs)
        cnn_pool1 = self.cnn_pool1(cnn_conv1)
        cnn_conv2 = self.cnn_conv2(cnn_pool1)
        cnn_pool2 = self.cnn_pool2(cnn_conv2)
        cnn_flatten = self.cnn_flatten(cnn_pool2)
        cnn_dense1 = self.cnn_dense1(cnn_flatten)

        bpnn_dense1 = self.bpnn_dense1(API_inputs)

        API_opcodes = tf.keras.layers.concatenate([cnn_dense1,bpnn_dense1])

        output = self.hydra_dense(API_opcodes)
        return output




#定义一些超参数
num_epochs = 50          #迭代次数
batch_size = 50          #每批数据的大小
learning_rate = 0.01    #学习率

data_loader = DataLoader()
x=(data_loader.train_data_op,data_loader.train_data_api)
y=np_utils.to_categorical(data_loader.train_label)

avg_accuracy = []
avg_loss = []
test2=[]

n_split=5
for train_index, test_index in KFold(n_split).split(x[0],x[1],y):
    # print("test index: ", test_index)
    x_train, x_test = (x[0][train_index],x[1][train_index]), (x[0][test_index],x[1][test_index])
    y_train, y_test = y[train_index], y[test_index]

    model = HYDRA()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=['accuracy'])

    #创建一个实例LossHistory
    # history = LossHistory()
    # 准确率不再上升时提前停止
    es = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,verbose=0,mode="min")
    #减小学习率
    Reduce=ReduceLROnPlateau(monitor='val_loss',
                            factor=0.1,
                            patience=10,
                            verbose=0,
                            mode='min',
                            min_delta=1e-3
                            )#min_lr=0.001

    model.fit(x,
    y,
    # validation_split=0.2,
    shuffle=True,
    validation_data = (x_test, y_test),
    verbose=1,
    batch_size=batch_size,
    epochs=num_epochs,
    callbacks=[Reduce,es]) # history,es,Reduce

    print('Model evaluation: ')
    t_acc=model.evaluate(x_test, y_test)[1]
    avg_accuracy.append(t_acc)
    t_loss = model.evaluate(x_test, y_test)[0]
    avg_loss.append(t_loss)
    print("k fold acc loss:",t_acc,t_loss)

    res1=model.evaluate((data_loader.test_data_op1,data_loader.test_data_api1), np_utils.to_categorical(data_loader.test_label1))
    res2=model.evaluate((data_loader.test_data_op2,data_loader.test_data_api2), np_utils.to_categorical(data_loader.test_label2))
    print(res1,res2)
    test2.append((res1,res2,data_loader.c))


print("K fold accuracy lst: {}".format(avg_accuracy))
print("Test2 accuracy lst: {}".format(test2))
print("K fold avg accuracy: {}".format(sum(avg_accuracy)/len(avg_accuracy)))
test_1=[i[0][1] for i in test2]
test_2=[i[1][1] for i in test2]
c = [i[2] for i in test2]
print("Test2 avg accuracy lst: {}".format(sum(test_1)/len(test_1)))
print("Test3 avg accuracy lst: {}".format(sum(test_2)/len(test_2)))
print("Corr2:",c)

# history.loss_plot('epoch')

# model.summary()# 打印神经网络结构参数
print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
