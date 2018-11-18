# 项目思路
## CountVectorizer
- 将处理后的文本数据转成词频矩阵，每段评论作为样本，设置5000个最高词频的单词作为特征，统计样本中词的出现次数
```
array([[1, 0, 2, ..., 0, 0, 0],  
       [0, 5, 0, ..., 7, 0, 0],  
       ...,  
       [0, 0, 3, ..., 4, 0, 0],  
       [4, 6, 0, ..., 0, 8, 0]]) 
```
- 用随机森林模型训练、预测
## TfidfVectorizer
- 将处理后的文本数据转成词频权重矩阵。TfidfTransformer用于统计vectorizer中每个词语的TFIDF值，TfidfVectorizer相当于CountVectorizer配合TfidfTransformer使用的效果。
```
array([[0.85, 0., 0.65, ..., 0., 0., 0.],
       [0., 0.74, 0., ..., 0.24, 0., 0.],
       ...,
       [0., 0., 0.53, ..., 0., 0.22, 0.],
       [0.89, 0., 0., ..., 0., 0., 0.36]])
```
- 用逻辑回归模型训练、预测
## Word2Vec
- 将处理后的文本数据转成词向量矩阵。每个单词为一个样本，设置最高词频的300个词作为特征。与该词对应的位置为1，其他位置全为0；若无对应则全为0
```
array([[1, 0, 0, ..., 0, 0, 0],  
       [0, 0, 0, ..., 0, 0, 0],  
       ...,  
       [0, 0, 1, ..., 0, 0, 0],  
       [0, 0, 0, ..., 0, 1, 0]]) 
```
- 从词向量到段落，处理方式一：向量平均  
将处理后的文本数据转成向量平均矩阵。每段评论作为样本，最高词频的300个词作为特征。累加评论中包含在词向量矩阵的单词向量最后取平均
```
array([[2, 0.5, 0, ..., 2.5, 0, 0],  
       [0, 1, 0, ..., 2, 0, 0.33],  
       ...,  
       [0, 0, 0.4, ..., 0, 0.8, 2],  
       [0, 0.2, 0, 10..., 0, 1, 0]]) 
```
- 从词向量到段落，处理方式二：聚类  
用Kmeans算法将词向量矩阵中相似词归类，创建质心矩阵，每段评论作为样本，所有类别作为特征。评论中的单词属于哪一类别则该位置加1
```
array([[2, 0, 7, ..., 0, 0, 0],  
       [0, 4, 0, ..., 0, 5, 0],  
       ...,  
       [0, 0, 1, ..., 0, 6, 0],  
       [0, 0, 3, ..., 0, 2, 0]]) 
```
- 用随机森林模型训练、预测
