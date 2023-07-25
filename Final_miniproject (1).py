#!/usr/bin/env python
# coding: utf-8

# In[51]:


#importimg dependancies

import pandas as pd 
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


# In[54]:


df = pd.read_csv('tweets_sentiment2.csv')


# In[55]:


df.head()


# In[56]:


df.info()


# In[57]:


df.isnull().sum()


# In[58]:


df.columns


# In[59]:


text_df = df.drop(['sentiment'
       ],
       axis=1)
text_df.head()


# In[60]:


print(text_df['tweet'].iloc[0],"\n")
print(text_df['tweet'].iloc[1],"\n")
print(text_df['tweet'].iloc[2],"\n")
print(text_df['tweet'].iloc[3],"\n")
print(text_df['tweet'].iloc[4],"\n")


# In[61]:


text_df.info()


# In[62]:


def data_processing(tweet):
    tweet = tweet.lower()
    tweet= re.sub(r"https\S+|www\S+https\S+", '',tweet, flags=re.MULTILINE)
    tweet= re.sub(r'\@w+|\#','',tweet)
    tweet= re.sub(r'[^\w\s]','',tweet)
    tweet_tokens = word_tokenize(tweet)
    filtered_text = [w for w in tweet_tokens if not w in stop_words]
    return " ".join(filtered_text)


# In[63]:


text_df.tweet = text_df['tweet'].apply(data_processing)


# In[64]:


text_df = text_df.drop_duplicates('tweet')


# In[65]:


stemmer = PorterStemmer()
def stemming(data):
    text = [stemmer.stem(word) for word in data]
    return data


# In[66]:


text_df['tweet'] = text_df['tweet'].apply(lambda x: stemming(x))


# In[67]:


text_df.head()


# In[68]:


print(text_df['tweet'].iloc[0],"\n")
print(text_df['tweet'].iloc[1],"\n")
print(text_df['tweet'].iloc[2],"\n")
print(text_df['tweet'].iloc[3],"\n")
print(text_df['tweet'].iloc[4],"\n")


# In[69]:


text_df.info()


# In[70]:


def polarity(tweet):
    return TextBlob(tweet).sentiment.polarity


# In[71]:


text_df['polarity'] = text_df['tweet'].apply(polarity)


# In[72]:


text_df.head(10)


# In[73]:


def sentiment(label):
    if label <0:
        return "Negative"
    elif label ==0:
        return "Neutral"
    elif label>0:
        return "Positive"


# In[74]:


text_df['sentiment'] = text_df['polarity'].apply(sentiment)


# In[75]:


text_df.head()


# In[76]:


fig = plt.figure(figsize=(5,5))
graph=sns.countplot(x='sentiment', data = text_df)


# In[77]:


pos_tweets = text_df[text_df.sentiment == 'Positive']
pos_tweets = pos_tweets.sort_values(['polarity'], ascending= False)
pos_tweets.head()


# In[78]:


text = ' '.join([word for word in pos_tweets['tweet']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in positive tweets', fontsize=19)
plt.show()


# In[79]:


neg_tweets = text_df[text_df.sentiment == 'Negative']
neg_tweets = neg_tweets.sort_values(['polarity'], ascending= False)
neg_tweets.head()


# In[80]:


text = ' '.join([word for word in neg_tweets['tweet']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in negative tweets', fontsize=19)
plt.show()


# In[81]:


neutral_tweets = text_df[text_df.sentiment == 'Neutral']
neutral_tweets = neutral_tweets.sort_values(['polarity'], ascending= False)
neutral_tweets.head()


# In[82]:


text = ' '.join([word for word in neutral_tweets['tweet']])
plt.figure(figsize=(20,15), facecolor='None')
wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title('Most frequent words in neutral tweets', fontsize=19)
plt.show()


# In[83]:


vect = CountVectorizer(ngram_range=(1,2)).fit(text_df['tweet'])


# In[84]:


feature_names = vect.get_feature_names()
print("Number of features: {}\n".format(len(feature_names)))
print("First 20 features:\n {}".format(feature_names[:20]))


# In[85]:


X = text_df['tweet']
Y = text_df['sentiment']
X = vect.transform(X)


# In[86]:


a


# In[87]:


print("Size of x_train:", (x_train.shape))
print("Size of y_train:", (y_train.shape))
print("Size of x_test:", (x_test.shape))
print("Size of y_test:", (y_test.shape))


# In[88]:


import warnings
warnings.filterwarnings('ignore')


# In[89]:


from sklearn.svm import LinearSVC


# In[90]:


SVCmodel = LinearSVC()
SVCmodel.fit(x_train, y_train)


# In[91]:


svc_pred = SVCmodel.predict(x_test)
svc_acc = accuracy_score(svc_pred, y_test)
print("test accuracy: {:.2f}%".format(svc_acc*100))


# In[92]:


print(confusion_matrix(y_test, svc_pred))
print("\n")
print(classification_report(y_test, svc_pred))


# In[93]:


from sklearn.pipeline import Pipeline

pipeline=Pipeline([('vect',vect),('SVCmodel',SVCmodel)])


# In[94]:


#saving machine learning model
import joblib 
joblib.dump(pipeline,'SentiAnlsModel4.pkl')


# In[95]:


load_model3=joblib.load('SentiAnlsModel4.pkl')


# In[96]:


pred=pipeline.predict(["is rishi sunak going to be britains next prime minister ?"])
print(pred)


# In[97]:


pred=pipeline.predict(["MP HM Narottam Mishra says, Pathan will be banned in Madhya Pradesh if objectionable scenes aren't removed from the film. Deepika's costume colour was chosen with a dirty mindset of Bollywood filmmakers."])
print(pred)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




