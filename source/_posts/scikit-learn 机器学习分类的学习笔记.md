---
title: scikit-learn_机器学习分类的入门学习笔记
categories: scikit-learn
date: 2023-07-27 15:30:00
---

This note I will mainly write in English for its skeleton.

I learn it from a [Youtube video](https://www.youtube.com/watch?v=M9Itm95JzL0) made by Keith Galli. So This note is based on his instruction.

## Preliminary
### Data Class

Define the class of Data is the first step.

The object of the data is the comments in Amazon. So divide them into 3 categories according to the sentiments.

```py
import random

class Sentiment:
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"
```

Define the review and its affliated methods.

```py
class Review:
    def __init__(self, text, score):
        self.text = text
        self.score = score
        self.sentiment = self.get_sentiment()
        
    def get_sentiment(self):
        if self.score <= 2:
            return Sentiment.NEGATIVE
        elif self.score == 3:
            return Sentiment.NEUTRAL
        else: #Score of 4 or 5
            return Sentiment.POSITIVE
```

The container of review which facilitates the manipulate.

```py
class ReviewContainer:
    def __init__(self, reviews):
        self.reviews = reviews
        
    def get_text(self):
        return [x.text for x in self.reviews]
    
    def get_sentiment(self):
        return [x.sentiment for x in self.reviews]
        
    def evenly_distribute(self):
        negative = list(filter(lambda x: x.sentiment == Sentiment.NEGATIVE, self.reviews))
        positive = list(filter(lambda x: x.sentiment == Sentiment.POSITIVE, self.reviews))
        positive_shrunk = positive[:len(negative)]
        self.reviews = negative + positive_shrunk
        random.shuffle(self.reviews)
```

### Load Data

```py
import json

file_name = './data/sentiment/books_small_10000.json'

reviews = []
with open(file_name) as f:
    for line in f:
        review = json.loads(line)
        reviews.append(Review(review['reviewText'], review['overall']))
```

### Prep Data

`train_test_split`function will divide the data to 2 parts. Use container to store.

```py
from sklearn.model_selection import train_test_split

training, test = train_test_split(reviews, test_size=0.33, random_state=42)
train_container = ReviewContainer(training)
test_container = ReviewContainer(test)
```

`_x`store the text. `_y`store the sentiment.

```py
train_container.evenly_distribute()
train_x = train_container.get_text()
train_y = train_container.get_sentiment()

test_container.evenly_distribute()
test_x = test_container.get_text()
test_y = test_container.get_sentiment()

print(train_y.count(Sentiment.POSITIVE))
print(train_y.count(Sentiment.NEGATIVE))
```

#### Bag of words vectorization

`Tfidf` is the abbreviation of **Term Frequency-Inverse Document Frequency**。

Tfidf是一种文本特征提取方法，用于衡量一个词对于一个文档在整个语料库中的重要性。它结合了词频（Term Frequency）和逆文档频率（Inverse Document Frequency）两个因素来计算一个词的权重。

词频（TF）表示一个词在一个文档中出现的频率，它衡量了一个词在文档中的重要性。逆文档频率（IDF）衡量了一个词在整个语料库中的普遍程度，即一个词在多少个文档中出现。IDF的计算公式是语料库中文档总数除以包含该词的文档数的对数。

Tfidf的计算公式是TF乘以IDF，它可以帮助我们找到在一个文档中频繁出现但在整个语料库中不常见的词，从而确定它们在文档中的重要性。在文本挖掘和信息检索中，Tfidf常被用作特征选择和文本相似度计算的基础。

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

vectorizer = TfidfVectorizer()
train_x_vectors = vectorizer.fit_transform(train_x)
test_x_vectors = vectorizer.transform(test_x)

```

## Classification

**SVM** is the abbreviation of **Support Vector Machine**， a Supervised learning algorithm, commonly used in classification and regression problems.

### Linear SVM

Linear SVM是线性支持向量机（Support Vector Machine）的一种变体。线性SVM是一种二分类模型，通过寻找一个最优的超平面来将两个不同类别的样本点分开。它的目标是找到一个最大间隔的超平面，使得两个类别的样本点尽可能远离超平面。

在线性SVM中，样本点被表示为特征空间中的向量，每个特征代表一个维度。这些向量被映射到一个高维特征空间，通过计算特征向量之间的内积来进行分类。线性SVM使用线性核函数（也称为线性可分SVM）将样本点映射到特征空间，从而在特征空间中找到最优的超平面。

线性SVM的优点包括计算效率高、容易解释和泛化能力强。它在处理线性可分问题时表现良好，但对于非线性可分问题，需要使用核函数将样本点映射到更高维的特征空间，以实现非线性分类。

```python
from sklearn import svm

clf_svm = svm.SVC(kernel='linear') #kernel 核
clf_svm.fit(train_x_vectors, train_y)
test_x[0]
clf_svm.predict(test_x_vectors[0])
```

### Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier

clf_dec = DecisionTreeClassifier()
clf_dec.fit(train_x_vectors, train_y)

clf_dec.predict(test_x_vectors[0])
```

### Naive Bayes

朴素贝叶斯

GNB是Gaussian Naive Bayes的缩写，它是一种朴素贝叶斯分类器，假设特征之间的概率分布服从高斯分布（即正态分布）。

```python
from sklearn.naive_bayes import GaussianNB

clf_gnb = DecisionTreeClassifier()
clf_gnb.fit(train_x_vectors, train_y)

clf_gnb.predict(test_x_vectors[0])
```

### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression

clf_log = LogisticRegression()
clf_log.fit(train_x_vectors, train_y)

clf_log.predict(test_x_vectors[0])
```

### Evaluation

关于模型评价，参见“逻辑回归”。

#### Mean Accuracy

```python
print(clf_svm.score(test_x_vectors, test_y))
print(clf_dec.score(test_x_vectors, test_y))
print(clf_gnb.score(test_x_vectors, test_y))
print(clf_log.score(test_x_vectors, test_y))
```

#### f1_score

```python
from sklearn.metrics import f1_score

f1_score(test_y, clf_svm.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEGATIVE])
#f1_score(test_y, clf_log.predict(test_x_vectors), average=None, labels=[Sentiment.POSITIVE, Sentiment.NEUTRAL, Sentiment.NEGATIVE])
```

#### Test with Examples

```py
test_set = ['very fun', "bad book do not buy", 'horrible waste of time']
new_test = vectorizer.transform(test_set)

clf_svm.predict(new_test)
```

### Tuning model (with Grid Search)

CV指"Cross-Validation"，指交叉验证。交叉验证是一种模型评估的方法，将数据集划分为训练集和验证集，通过多次重复的训练和验证来评估模型的性能和泛化能力。

SVC的缩写是Support Vector Classifier，它是支持向量机（Support Vector Machine）的一种变体。SVC是一种用于分类问题的监督学习算法。

支持向量机通过寻找一个最优的超平面来将不同类别的样本点分开。SVC与线性SVM类似，但在处理线性不可分问题时使用了软间隔（soft margin）的概念。软间隔允许一些样本点位于超平面的错误一侧，以提高模型的泛化能力。

SVC可以使用不同的核函数来处理非线性分类问题，常用的核函数包括线性核、多项式核和高斯核等。通过引入核函数，SVC将样本点映射到更高维的特征空间，从而在特征空间中找到最优的超平面。

SVC的优点包括泛化能力强、对于高维数据和非线性问题有较好的表现，同时也可以通过调整参数来控制模型的复杂度。

```python
from sklearn.model_selection import GridSearchCV

parameters = {'kernel': ('linear', 'rbf'), 'C': (1,4,8,16,32)}

svc = svm.SVC()
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(train_x_vectors, train_y)
```

## Saving Model

pickle，腌菜。

```python
import pickle

with open('./models/sentiment_classifier.pkl', 'wb') as f:
    pickle.dump(clf, f) #save

with open('./models/entiment_classifier.pkl', 'rb') as f:
    loaded_clf = pickle.load(f) #load

#apply
print(test_x[0])
loaded_clf.predict(test_x_vectors[0])
```
