#!/usr/bin/env python
# _*_ coding:utf-8 _*_

"""
@Author       : 20180663 Zhang Qinghui
@Version      : V1.0
@E-Mail       : 1415984778@qq.com
@File         : test1.py
@CreateTime   : 2021/4/20
@Description  : none
@ModifyTime   : 2021/3/4
@company      : CSUFT
"""


import pandas as pd
import jieba

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB,BaseDiscreteNB,BaseNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn import metrics

# 对文本内容进行分词
def chinese_word_cut(mytext):
    return " ".join(jieba.cut(mytext))

# 将停用词作为列表的格式进行保存
def get_custom_stopwords(stop_words_file):
    with open(stop_words_file,'r',encoding='utf-8') as f:
        stopwords = f.read()
    stopwords_list = stopwords.split('\n')
    custom_stopwords_list = [i for i in stopwords_list]
    return custom_stopwords_list

data = pd.read_csv(open('train.csv','r',encoding='utf-8'))
test_data = pd.read_csv(open('titles.csv','r',encoding='utf-8'))
# data = pd.read_csv('train.csv',encoding='utf-8')
neg_pos = data[(data['label'] == 'Negative') | (data['label']== 'Positive')].copy()

neg_pos.loc[neg_pos['label'] == 'Negative','label'] = 0
neg_pos.loc[neg_pos['label'] == 'Positive','label'] = 1
neg_pos.loc[:,'cutted_review'] = neg_pos['review'].apply(chinese_word_cut)

test_data.loc[:,'cutted_review'] = test_data['title'].apply(chinese_word_cut)

# 训练数据
x_train = neg_pos[['cutted_review']]
y_train = neg_pos['label'].astype('int')
# 测试数据
x_test = test_data[['cutted_review']]


# x = neg_pos[['cutted_review']]
# y = neg_pos['label'].astype('int')
# # 默认模式下,train_test_split函数对训练集和测试集的划分比例为3:1
# x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.4,random_state=1)
# print(type(x_train))


# 停用词处理
stop_words_file = "stopwords.txt"
stopwords = get_custom_stopwords(stop_words_file)
stopwords = stopwords + ['ain', 'aren', 'couldn', 'daren', 'didn', 'doesn', 'don', 'hadn', 'hasn', 'haven', 'isn', 'll', 'mayn', 'mightn', 'mon', 'mustn', 'needn', 'oughtn', 'shan', 'shouldn', 've', 'wasn', 'weren', 'won', 'wouldn', '事情', '信息', '干嘛', '晚安', '正在', '看到', '肯定']

vectorizer = TfidfVectorizer(max_df=0.5,
                             min_df=1,
                             use_idf=True,
                             token_pattern=u'(?u)\\b[^\\d\\W]\\w+\\b',
                             stop_words=frozenset(stopwords))

# 使用生成的特征矩阵来训练模型
# 采用朴素贝叶斯分类模型
nb = MultinomialNB()

# 为了避免因修改参数和换用测试集而造成错误几率的增加,sklearn增加了pipline的功能
# pipline将顺序工作连接在一起,将隐藏其中的功能顺序关联,只需从外部调用一次就能顺序完成所有工作
# 这里将vect和nb串联起来
pipe = make_pipeline(vectorizer,nb)
# 查看管道中的步骤
# print(pipe.steps)
# 把管道当成一个整体的模型来进行调用
# 对未经特征向量化的训练集内容输入,做交叉验证,计算出模型分类准确率的均值
model_accuracy = cross_val_score(pipe,x_train.cutted_review,y_train,cv=6,scoring='roc_auc').mean()
print('交叉验证AUC均值:')
print(model_accuracy)

# 拟合模型
pipe.fit(x_train.cutted_review,y_train)

# 对情感分类标记进行预测
y_pred = pipe.predict(x_train.cutted_review)


# 预测结果
# print(y_pred)
'''
    TP: 本来是正向，预测也是正向的；
    FP: 本来是负向，预测却是正向的；
    FN: 本来是正向，预测却是负向的；
    TN: 本来是负向，预测也是负向的。
    准确率(accuracy) = 预测对的/所有 = (TP+TN)/(TP+FN+FP+TN)
    精确率(precision) = TP/(TP+FP)
    召回率(recall) = TP/(TP+FN)
'''
y_test = y_train
print('------测试结果------')
# 查看测试准确率
y_pred_accuracy = metrics.accuracy_score(y_test,y_pred)
print('准确率:',y_pred_accuracy)
# 查看精确率
y_pred_precision = metrics.precision_score(y_test,y_pred)
print('精确率',y_pred_precision)
# 查看测试召回率
y_pred_recall = metrics.recall_score(y_test,y_pred)
print('召回率:',y_pred_recall)

print('混淆矩阵:')
# 查看混淆矩阵
final_result = metrics.confusion_matrix(y_test,y_pred)
print(final_result)


# 输出预测结果文档
y_pred = pipe.predict_proba(x_test.cutted_review)[:,1]
# print(y_pred)
test_data.loc[:,'Pred'] = y_pred
test_data[['ID','Pred']].to_csv('pre.csv',index=False,encoding='utf-8')
