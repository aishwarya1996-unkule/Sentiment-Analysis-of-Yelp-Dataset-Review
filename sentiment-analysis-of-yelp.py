#!/usr/bin/env python
# coding: utf-8

# In[21]:


# %% [code]
# Read data
from string import punctuation
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud
from nltk.stem.porter import PorterStemmer
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
def loadJSON(path, data_dict,size=-1):
    """
        Function that loads data. Since the reviews is not JSON itself, but each row it is, i need to read it line by line.
        :param path: path to file
        :param data_dict: columns:list
        :param size: how many lines to load
        :rtype size: int
        :rtype path: str
        :rtype data_dict: dict
        :return: data with values read.

    """

    # check input
    if len(data_dict) == 0:
        raise RuntimeError("No columns")

    else:
        cnt = -1
        # data are to big to load in normal way, seoncondly, it seems no correct json format.
        with open(path, 'rb') as f:
            from json import loads
            if size !=-1:
                cnt = 0
            for line in f:
                cnt+=1
                line = loads(line)
                # not happy about nested loops, but for now it will do.
                for key in data_dict.keys():
                    # not using get method, since i need to raise error if key not exists
                    data_dict[key].append(line[key])
                del line
                if cnt > 0 and cnt == size  :
                    break
            # with contectx should do work but just in case
            f.close()
    return data_dict


def text_clean(review):
    """Process review into token
       remove following regex

    # remove hypertext links
    review = re.sub(r'https?:\/\/.*[\r\n]*', '', review)
    # extract hash tag
    review = re.sub(r'@', '', review)
    # extract @
    review = re.sub(r'#', '', review)
    # extract numbers
    review = re.sub('[0-9]*[+-:]*[0-9]+', '', review)
    # extract '
    review = re.sub("'s", "", review)

    strip empty spaces and lower case words.

       :param review: the review.
       :rtype review: string

       :return list_of_words: list with words cleaned fro mthe review.
    """

    import re

    # remove hypertext links
    review = re.sub(r'https?:\/\/.*[\r\n]*', '', review)
    # extract hash tag
    review = re.sub(r'@', '', review)
    # extract @
    review = re.sub(r'#', '', review)
    # extract numbers
    review = re.sub('[0-9]*[+-:]*[0-9]+', '', review)
    # extract '
    review = re.sub("'s", "", review)
    return review.strip().lower()


def remove_punctuations(string):
    """
    Remove puctuation.
    :param string:
    :return: string
    """
    return ''.join(c for c in string if c not in punctuation)



def remove_stopwords(string, stop_words):
    """
    Removing stop words.
    :param string:
    :param stop_words:
    :return: string
    """
    tokenized = word_tokenize(string)
    filtered_sentence = [word for word in tokenized if not word in stop_words]
    return ' '.join(c for c in filtered_sentence)


def sigmoid(z):
    '''
    Input:
        z: is the input (can be a scalar or an array)
    Output:
        h: the sigmoid of z
    '''

    # calculate the sigmoid of z
    h = 1 / (1 + np.exp(-z))
    return h

def print_significant_words(logreg_coef=None, class_=None, count=100, count_vector=None, graph=True):
    """

    :param logreg_coef: logistic regression coefficients
    :param class_:  0 is negative, 1 positive
    :param count: how many words to show, default to 100
    :param count_vector: counting vector
    :param graph: True ro print graph
    :rtype logreg_coef: numpy array
    :return:
    """


    if isinstance(logreg_coef, np.ndarray) and class_ in (0, 1) and isinstance(count_vector, CountVectorizer):
        pass
    else:
        raise TypeError("Parameters has wrong type, see help(print_significant_words)")

    # get id from model
    if class_ == 0:
        # since we sort ids, we choose range below or above 0
        # for negative sentiment, estimator should have + sing
        range_ = range(0, 1 + count, 1)
        sentiment = 'Negative'
    else:

        # since we sort ids, we choose range below or above 0
        # for positive sentiment, estimator should have + sing
        range_ = range(-1, -1 - count, -1)
        sentiment = 'Positive'
    ids = np.argsort(logreg_coef)

    words = [list(count_vector.vocabulary_.keys())[list(count_vector.vocabulary_.values()).index(id_)] for id_ in
             ids[range_]]

    # graph
    if graph == True:
        fig = plt.figure(figsize=(10, 6))
        ax = sns.barplot(words, logreg_coef[ids[range_]])
        plt.title("Top {} words for {} Sentiment".format(count, sentiment), fontsize=20)
        x_locs, x_labels = plt.xticks()
        plt.setp(x_labels, rotation=40)
        plt.ylabel('Feature Importance', fontsize=12)
        plt.xlabel('Word', fontsize=12);

    return words

def wc(words):
    """

    :param words: list of word to make fancy graph
    :type words: list
    :return:
    """

    wordcloud = WordCloud(background_color="white", max_words=len(' '.join(words)),                           max_font_size=40, relative_scaling=.5, colormap='summer').generate(' '.join(words))
    plt.figure(figsize=(13, 13))
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.show()


def stemming(str_input):
    """
    Stemmer. Removing ing , ed etc....
    :param str_input: Input sentence
    :return: string of stemmed words
    """

    words= word_tokenize(str_input)
    porter_stemmer = PorterStemmer()
    filtered_sentence = [porter_stemmer.stem(word) for word in words]
    return ' '.join(c for c in filtered_sentence)


if __name__ == '__main__':
    pass


# In[22]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns
import multiprocessing
import gc
# my imports
#import utils_yelp 
import sklearn.model_selection as model_selection
from tqdm.notebook import tqdm_notebook
tqdm_notebook.pandas()
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,mean_squared_error,plot_confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('E:\\DBDA\\Project\\yelp_academic_dataset_review'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:





# # Yelp analysis
# ## Introduction
# Yelp is company which is porviding the rating of hte restaurants. It is internet based rating forum, where people can write and rate restaurants aroud the globe. 
# ### Data
# We will use reviews for sentiment analysis. This is classification problem. The load data are to big to be loaded at one, therefore we read line by line. Secondly, the each row is JSON but file is not JSON itself.
# ### Analysis
# Since many of the Kaggle kernells trying fit model to some subset of data around 1 millinon. Therefore Kernell run quite long time. I will try Logistic regression, XGBoost, RandomForest to perform classification.
# ### Note
# Helepr function can be found in utils_yelp.py, and test_utils_yelp.py for basic parametric test.

# # 1. Load data
# 

# In[23]:


help(utils_yelp.loadJSON)


# In[15]:


data = pd.DataFrame(utils_yelp.loadJSON('/kaggle/input/yelp-dataset/yelp_academic_dataset_review.json',{'stars': [], 'text':[]},size=1000000))

# helping the RAM
gc.collect()
data.head(5)


# # 2.EDA

# The basic EDA is performed to find if there are extreme values and missing data.

# ## 2.1 Missing value

# In[4]:


display(data.shape)
display(data.isnull().sum(axis=0))
display(data.describe())
sns.heatmap(data.isnull(), cbar=False)


# As i seen from basic descritptive statistics and heatmat, there are no missing values. No treatment is required.

# ## 2.2 Descriptive statistics

# In[5]:


display(data.describe())
display(data.info())


# Basic descriptive statics showing, no outlayers since stars are between values 1 to 5.

# ## 2.3b Distribution of the stars
# 

# In[ ]:





# In[6]:



labels = data['stars'].value_counts().index
sizes = data['stars'].value_counts().values   
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True)
ax1.axis('equal')
plt.show()


# It seems that there quite a lot possitive ( > 3 ) values over 65 %. We use Logistic regression.The **Naive Bayes** does not seem reasonable, because it is required to have proportion of negative and possitive data to be similar, that is arou 50 % each. 

# In[7]:


sns.distplot(data['stars'], kde=False)


# # 3. Sentiment Analysis

# ## 3.1 Create labels

# To have some balance in possitive and negative sentiment, values over 3 would be treated as possitive.

# In[8]:


data['labels']  = np.where(data['stars']>3,1,0)
data.head(10)


# # 4 Preprocess data

# ## 4.1 Cleaning regex etc...

# In[9]:


help(utils_yelp.text_clean)


# In[10]:



data['text']=data['text'].progress_apply(lambda x: utils_yelp.text_clean(x))
data.head(5)


# ## 4.2 Cleaning punctuation

# In[11]:


help(utils_yelp.remove_punctuations)


# In[12]:




    
data['text']=data['text'].progress_apply(lambda x: utils_yelp.remove_punctuations(x))
data.head(5) 


# ## 4.3 Cleaning stopwords
# This takes very long since all sentences are split t the word a then reconstruct back to sentence.

# In[13]:


help(utils_yelp.remove_stopwords)


# In[14]:


stop_words = stopwords.words('english')
# It takes to long for whole data set, therefore logic is shift to the CountVectoriser
#data['text']=data['text'].progress_apply(lambda x: utils_yelp.remove_stopwords(x,stop_words))
data.head(5)    


# In[15]:


help(utils_yelp.stemming)


# In[16]:


# takes really long need to imporve if i have time
data['text']=data['text'].progress_apply(lambda x: utils_yelp.stemming(x))
data.head(5) 


# # 5. Splitting data

# Split data to train and test with size 0.7.****

# In[17]:


X_train_corp, X_test_corp, y_train, y_test = model_selection.train_test_split(data['text'], data['labels'],test_size=0.3, random_state=42)
display(X_train_corp.shape)
display(X_train_corp.shape)
display(y_train.shape)
display(y_test.shape)


# # 6. Modelling

# In[18]:


# Prepare vector with frequencies
vector_count = CountVectorizer(min_df=100, ngram_range=(1, 1),stop_words=stop_words)
X_train = vector_count.fit(X_train_corp).transform(X_train_corp) 
X_test = vector_count.transform(X_test_corp)
print(X_train.shape) 


# # 6.1 Logistic Regrssion

# In[19]:


logreg = LogisticRegression(max_iter=500,solver='liblinear').fit(X_train,y_train)
#print ("Accuracy: ", accuracy_score(y_test, logreg.predict(X_test)))
   
print(f"LG accuracy trainnig set: {logreg.score(X_train, y_train)}")
print(f"LG accuracy test set: {logreg.score(X_test, y_test)}")
print ("MSE: ", mean_squared_error(y_test, logreg.predict(X_test)))


# ## Model summary

# ### Confusion Matrix

# In[20]:


cm = plot_confusion_matrix(logreg, X_test, y_test,
                                 display_labels=['Class 0', 'Class 1'],
                                 cmap=plt.cm.Blues,
                                 normalize='true')
cm.ax_.set_title('Logistic regression Confusion Matrix');


# In[21]:


help(utils_yelp.print_significant_words)


# In[22]:


help(utils_yelp.wc)


# ##  Most positive words

# In[23]:




words = utils_yelp.print_significant_words(logreg.coef_[0],1,15,vector_count,True)
utils_yelp.wc(words)



# ##  Most negative words

# 

# In[24]:


words = utils_yelp.print_significant_words(logreg.coef_[0],0,15,vector_count,True)
utils_yelp.wc(words)


# # 

# # 6.2 XGBoost

# In[25]:



XGB_model = XGBClassifier(random_state=1)
XGB_model.fit(X_train, y_train)

print(f"XGBoost score - trainnig set: {XGB_model.score(X_train, y_train)}")
print(f"XGBoost score - test set: {XGB_model.score(X_test, y_test)}")
print ("MSE: ", mean_squared_error(y_test, XGB_model.predict(X_test)))


# In[26]:


cm = plot_confusion_matrix(XGB_model ,X_test, y_test,
                                 display_labels=['Class 0', 'Class 1'],
                                 cmap=plt.cm.Blues,
                                 normalize='true')
cm.ax_.set_title('XGBoost Confusion Matrix');


# # 6.3 Random Forest

# In[27]:


score_tr = []
score_t = []
estim_depth = []
mse_tr = []
mse_t = []
max_d = 5
for i in range(1,max_d+1):
    print(i)
    random_forest_model = RandomForestClassifier(n_estimators=i,random_state=1)
    random_forest_model.fit(X_train, y_train)
    score_tr.append(random_forest_model.score(X_train, y_train))
    score_t.append(random_forest_model.score(X_test, y_test))
    mse_tr.append(mean_squared_error(y_train, random_forest_model.predict(X_train)))
    mse_t.append(mean_squared_error(y_test, random_forest_model.predict(X_test)))
    estim_depth.append(i)
    


# In[28]:


plt.figure()
plt.title('The test and train accuracy for different number of Trees')
plt.plot(range(1,max_d+1),score_tr,color='red',label='Train Accuracy')
plt.plot(range(1,max_d+1),score_t,label=' Test Accuracy')
plt.legend()
plt.show()
print(estim_depth[np.argmax(score_t)])


# In[29]:


plt.figure()
plt.title('The test and train MSE for different number of Trees')
plt.plot(range(1,max_d+1),mse_tr,color='red',label='Train MSE')
plt.plot(range(1,max_d+1),mse_t,label=' Test MSE')
plt.legend()
plt.show()
print(estim_depth[np.argmin(mse_t)])


# In[30]:


print(f"Random forest score - trainnig set: {score_tr[np.argmax(score_t)]}")
print(f"Random forest score - test set: {score_t[np.argmax(score_t)]}")
print ("MSE: ", mse_t[np.argmax(score_t)] )


# In[31]:


cm = plot_confusion_matrix(random_forest_model ,X_test, y_test,
                                 display_labels=['Class 0', 'Class 1'],
                                 cmap=plt.cm.Blues,
                                 normalize='true')
cm.ax_.set_title('Random Forest Confusion Matrix');


# ## 6.4 Pipeline LG

# In[32]:


tfidf_vector = TfidfVectorizer(min_df=100,ngram_range=(1, 1),stop_words=stop_words
                       )

param_grid = [{
               'logreg_reg__penalty': ['l1', 'l2'], 
                
               'logreg_reg__max_iter':[500],
               'logreg_reg__C': [1.0, 10.0]},

             
                 ]


logreg_tfidf = Pipeline([
    ('vc_norm', tfidf_vector),
    ('logreg_reg', LogisticRegression(random_state=1, solver='saga'))

]
                       )

rf_tfidf = Pipeline([('vc_norm', tfidf_vector),
                     ('rf', RandomForestClassifier(random_state=1))])

gs_lr_tfidf = GridSearchCV(logreg_tfidf, param_grid,
                           scoring='accuracy',
                           cv=3,
                           verbose=1,
                           n_jobs=-1)

# Take long
gs_lr_tfidf.fit(X_train_corp, y_train)


print('Best LG parameter set: ' + str(gs_lr_tfidf.best_params_))
print('Best LG accuracy: %.3f' % gs_lr_tfidf.best_score_)





# In[ ]:





# In[33]:



import sklearn.metrics as metrics
fpr, tpr, threshold = metrics.roc_curve(y_test, random_forest_model.predict(X_test))
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic Forests')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[34]:


fpr, tpr, threshold = metrics.roc_curve(y_test, XGB_model.predict(X_test))
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating XGBoost')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[35]:


fpr, tpr, threshold = metrics.roc_curve(y_test, logreg.predict(X_test))
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating LG')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[ ]:




