import pandas as pd
from gensim import corpora, models, similarities
from sklearn.metrics import classification_report
import pickle

if __name__ == '__main__':
    #加载数据
    print('加载数据')
    df = pd.read_csv('data.csv')
    Strings = [item.split(' ') for item in df['Strings'].tolist()]
    #向量化
    dictionary = corpora.Dictionary(Strings)
    corpus = [dictionary.doc2bow(doc) for doc in Strings]  # generate the corpus
    tf_idf = models.TfidfModel(corpus)  # the constructor
    corpus_tfidf = tf_idf[corpus]
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=100)
    print('建立LSA对应的文档主题矩阵')
    # 建立LSA对应的文档主题矩阵
    d_l = []
    labelss = df['Lables'].tolist()
    for x in range(len(Strings)):
        tmp = []
        a1 = dictionary.doc2bow(Strings[x])
        for xx in lsi[a1]:
            tmp.append(xx[1])
        if len(tmp) != lsi.num_topics:
            del labelss[x]
            continue
        d_l.append(tmp)
    print(d_l)
    # 利用SVM进行分类

    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split

    X_train, X_test, Y_train, Y_test = train_test_split(d_l, labelss, test_size=0.3)
    sv = SVC(C=1, kernel='linear')
    print('训练SVM')
    sv.fit(X_train, Y_train)
    y_pred = sv.predict(X_test)
    print('训练结果')
    print(classification_report(Y_test, y_pred))
    #保存模型
    with open('./models/dictionary.pickle', 'wb') as f:
        pickle.dump(dictionary, f)

    with open('./models/tf_idf.pickle', 'wb') as f:
        pickle.dump(tf_idf, f)

    with open('./models/lsi.pickle', 'wb') as f:
        pickle.dump(lsi, f)

    with open('./models/svm.pickle', 'wb') as f:
        pickle.dump(sv, f)