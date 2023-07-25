import streamlit as st
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import altair as alt
from PIL import Image
from streamlit_option_menu import option_menu
from annotated_text import annotated_text
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import pandas as pd
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import style
style.use('ggplot')
stop_words = set(stopwords.words('english'))
# from wordcloud import WordCloud


load_model = joblib.load('SentiAnlsModel4.pkl')
st.set_page_config(page_title='Twitter Sentiment Analysis', page_icon=None)


# Fxn


def convert_to_df(sentiment):
    sentiment_dict = {'polarity': sentiment.polarity,
                      'subjectivity': sentiment.subjectivity}
    sentiment_df = pd.DataFrame(
        sentiment_dict.items(), columns=['metric', 'value'])
    return sentiment_df


def analyze_token_sentiment(docx):
    analyzer = SentimentIntensityAnalyzer()
    pos_list = []
    neg_list = []
    neu_list = []
    for i in docx.split():
        res = analyzer.polarity_scores(i)['compound']
        if res > 0.1:
            pos_list.append(i)
            pos_list.append(res)

        elif res <= -0.1:
            neg_list.append(i)
            neg_list.append(res)
        else:
            neu_list.append(i)

    result = {'positives': pos_list,
              'negatives': neg_list, 'neutral': neu_list}
    return result


def main():
    # st.title("Twitter Sentiment Analysis ")
    # header(“Twitter”)
    st.markdown(
        f'<h1 style="color:#142664;font-size:46px;">{"Twitter Sentiment Analysis"}</h1>', unsafe_allow_html=True)

    col3, col4, col5 = st.columns(3)
    # with col4:
    # 	image = Image.open('Timage.png')
    # 	st.image(image)

    # navbar
    choice = option_menu(None, ["Home", "About"],
                         icons=['house', 'file-earmark-person'],
                         menu_icon="cast", default_index=0, orientation="horizontal")

    menu = ["Home","About"]
    orientation = "horizontal"
    choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Home":
        st.subheader("Home")
        #
        choice1 = option_menu(None, ["Tweet", "Dataset of Tweets"],
                              icons=['twitter', 'database'],
                              menu_icon="cast", default_index=0, orientation="horizontal")
        #
        if choice1 == "Tweet":
            st.subheader("Tweet")
            with st.form(key='nlpForm'):
                raw_text = st.text_area("Enter Tweet Here")
                submit_button = st.form_submit_button(label='Analyze')

                # layout
                col1, col2 = st.columns(2)
                if submit_button:

                    with col1:
                        st.info("Results")
                        sentiment = TextBlob(raw_text).sentiment
                        st.write(sentiment)

                        # Emoji
                    if sentiment.polarity > 0:
                        st.markdown("Sentiment:: Positive :smiley: ")
                    elif sentiment.polarity < 0:
                        st.markdown("Sentiment:: Negative :angry: ")
                    else:
                        st.markdown("Sentiment:: Neutral 😐 ")

                    # Dataframe
                    result_df = convert_to_df(sentiment)
                    st.dataframe(result_df)

                    # Visualization
                    c = alt.Chart(result_df).mark_bar().encode(
                        x='metric',
                        y='value',
                        color='metric')
                    st.altair_chart(c, use_container_width=True)

                    with col2:
                        st.info("Token Sentiment")

                        token_sentiments = analyze_token_sentiment(raw_text)
                        st.write(token_sentiments)
                else:
                    st.subheader("About")
        if choice1 == "Dataset of Tweets":
            st.subheader("Dataset of Tweets")
            # dataset
            with st.expander('Analyze CSV'):
                upl = st.file_uploader('upl')
                if upl:
                    # importimg dependancies
                    text_df = pd.read_csv(upl, usecols=['Text'])
                    load_model = joblib.load('SentiAnlsModel4.pkl')

                    def sentiment(Text):
                        return load_model.predict(Text)
                    
                    text_df['sentiment'] = text_df.apply(sentiment)
                    
 
                    
                    #fig = plt.figure(figsize=(5,5))
                    # Create a section for matplotlib figure
                    st.header('Plot of Data')

                    fig, ax = plt.subplots(1,1)
                    ax.graph=sns.countplot(x='sentiment', data = text_df)

                    st.pyplot(fig)
                    #st.altair_chart(graph, use_container_width=True)

                    # graph

                    # pos_tweets = text_df[text_df.sentiment == 'Positive']
                    # pos_tweets = pos_tweets.sort_values(['sentiment'], ascending= False)
                    # pos_tweets.head()

                    # text = ' '.join([word for word in pos_tweets['Text']]) #wordcloud
                    # st.pyplot.figure(figsize=(20,15), facecolor='None')
                    # wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
                    # st.pyplot.imshow(wordcloud, interpolation='bilinear')
                    # st.pyplot.axis("off")
                    # st.pyplot.title('Most frequent words in positive tweets', fontsize=19)
                    # st.pyplot.show()

                    # neg_tweets = text_df[text_df.sentiment == 'Negative']
                    # neg_tweets = neg_tweets.sort_values(['sentiment'], ascending= False)
                    # neg_tweets.head()
                    # text = ' '.join([word for word in neg_tweets['Text']])
                    # st.pyplot.figure(figsize=(20,15), facecolor='None')
                    # wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
                    # st.pyplot.imshow(wordcloud, interpolation='bilinear')
                    # st.pyplot.axis("off")
                    # st.pyplot.title('Most frequent words in negative tweets', fontsize=19)
                    # st.pyplot.show()
                    # neutral_tweets = text_df[text_df.sentiment == 'Neutral']
                    # neutral_tweets = neutral_tweets.sort_values(['sentiment'], ascending= False)
                    # neutral_tweets.head()
                    # text = ' '.join([word for word in neutral_tweets['Text']])
                    # st.pyplot.figure(figsize=(20,15), facecolor='None')
                    # wordcloud = WordCloud(max_words=500, width=1600, height=800).generate(text)
                    # st.pyplot.imshow(wordcloud, interpolation='bilinear')
                    # st.pyplot.axis("off")
                    # st.pyplot.title('Most frequent words in neutral tweets', fontsize=19)
                    # st.pyplot.show()

                    #fig = plt.figure(figsize=(7,7))

                    st.header('Pie chart of data')

                    fig2, ax = plt.subplots(1,1)

                    colors = ("yellowgreen", "gold", "red")
                    wp = {'linewidth':2, 'edgecolor':"black"}
                    tags = text_df['sentiment'].value_counts()
                    explode = (.1,.1,.1)
                    ax.fig2=tags.plot(kind='pie', autopct='%1.1f%%', shadow=False, colors = colors,startangle=90, wedgeprops = wp, explode = explode, label='')

                    st.pyplot(fig2)
                    #plt.title('Distribution of sentiments')
                    # tags.plot(kind='pie', autopct='%1.1f%%', shadow=False, colors=colors,
                    #           startangle=90, wedgeprops=wp, explode=explode, label='')
                    # tags.st.pyplot(kind='pie', autopct='%1.1f%%', shadow=False, colors=colors,
                    #           startangle=90, wedgeprops=wp, explode=explode, label='')
                    # #st.pyplot.title('Distribution of sentiments')
                    # st.pyplot.title('Distribution of sentiments')


                    
                    

                    
                    # extract the hashtag

                    def hashtag_extract(tweets):
                        hashtags = []
                        # loop words in the tweet
                        for Text in tweets:
                            ht = re.findall(r"#(\w+)", Text)
                            hashtags.append(ht)
                        return hashtags
                    ht = hashtag_extract(text_df['Text'])

                    # unnest list
                    ht = sum(ht, [])

                    import nltk
                    freq = nltk.FreqDist(ht)
                    d = pd.DataFrame({'Hashtag': list(freq.keys()),'Count': list(freq.values())})
                    d.head()
                    # select top 10 hashtags

                    st.header('top 10 hashtags')

                    fig3, ax = plt.subplots(1,1)

                    d = d.nlargest(columns='Count', n=10)
                    #plt.figure(figsize=(15,9))

                    ax.fig3=sns.barplot(data=d, x='Hashtag', y='Count')
                
                    st.pyplot(fig3)                    



if __name__ == '__main__':
    main()
