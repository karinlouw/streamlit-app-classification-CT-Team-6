"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Plase follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd
import numpy as np
import seaborn as sns
pd.set_option('display.max_colwidth', 400)


# Preprocessing
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter

# NLP Packages
from textblob import TextBlob 
import spacy
from spacy import displacy
from spacy_streamlit import visualize_parser
nlp = spacy.load('en')
from gensim.summarization import summarize

# Wordcloud
from wordcloud import WordCloud, ImageColorGenerator

# Images
from PIL import Image

# Plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 

# Sumy Summary Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

# Vectorizer
tweet_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(tweet_vectorizer) # loading your vectorizer from the pkl file

# Load csvs
raw = "resources/train.csv"
clean = "resources/clean_tweet_df.csv"
metadata = "resources/df_with_metadata.csv"
top_words = "resources/top_words_per_cat.csv"
top_hashtags="resources/top_hashtag_df.csv"
top_mentions="resources/top_mentions_df.csv"

# To Improve speed and cache data
@st.cache(allow_output_mutation=True)
def explore_data(dataset):
	df = pd.read_csv(os.path.join(dataset))
	return df 

# Dataframes
raw_df = pd.DataFrame(explore_data(raw))
clean_df = pd.DataFrame(explore_data(clean))
df_with_metadata = pd.DataFrame(explore_data(metadata))
top_words_df=pd.DataFrame(explore_data(top_words))
top_hashtags_df=pd.DataFrame(explore_data(top_hashtags))
top_mentions_df=pd.DataFrame(explore_data(top_mentions))



# Sentiment Dictionary
@st.cache
def get_keys(val,my_dict):
	for key, value in my_dict.items():
		if value == value:
			return key

# Function to Analyse Tokens and Lemma
@st.cache
def text_analyzer(my_text):
	nlp = spacy.load('en')
	docx = nlp(my_text)
	tokens = [ token.text for token in docx]
	allData = [('Token:{},\n"Lemma":{}'.format(token.text,token.lemma_))for token in docx ]
	return allData

# Function For Extracting Entities
@st.cache
def entity_analyzer(my_text):
	nlp = spacy.load('en')
	docx = nlp(my_text)
	tokens = [ token.text for token in docx]
	entities = [(entity.text,entity.label_)for entity in docx.ents]
	allData = ['"Token":{},\n"Entities":{}'.format(tokens,entities)]
	return allData

#Function to calculate work frequency
@st.cache
def word_freq(clean_text_list, top_n):
	"""
	Word Frequency
	"""
	flat = [item for sublist in clean_text_list for item in sublist]
	with_counts = Counter(flat)
	top = with_counts.most_common(top_n)
	word = [each[0] for each in top]
	num = [each[1] for each in top]
	return pd.DataFrame([word, num]).T


# The main function where we will build the actual app
def main():
	'''Creates a main title and subheader on your page -
	these are static across all pages'''
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Natural Language Processing Tool", "Exploratory Data Analysis"]
	selection = st.sidebar.selectbox("Choose Option", options)

	##### Building out the Prediction page ####
	if selection == "Prediction":
		st.markdown("# Machine Learning Model Predictions")
		st.markdown('Sentiment analysis is the classification of text in emotional categories such as positive, neutral, negative and news. The following machine learning models were built and trained to predict the emotional drive of tweets related to climate change. Please enter your text below and select a machine learning model to predict the sentiment of your text.')
		raw_text = st.text_area("Enter Text","Type Here")		
		

		# Model Prediction

		#Select model
		all_ml_modles= ["LR","LR3", "LR3"]
		model_choice = st.selectbox("Select base ML model",all_ml_modles)
		
		st.markdown("#### Select 'Classify' to view the result of the model prediction")
		st.markdown("")
		prediction_labels = {'anti':-1,'news':0,'pro':1,}
		if st.button("Classify"):
			#st.text("Original Text:\n{}".format(raw_text))
			vect_text = tweet_cv.transform([raw_text]).toarray()

			if model_choice == 'LR':
				predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
				# st.write(prediction)
			elif model_choice == 'LR2':
				predictor = joblib.load(open(os.path.join("resources/Logistic_regression2.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
				# st.write(prediction)
			elif model_choice == 'LR3':
				predictor = joblib.load(open(os.path.join("resources/Logistic_regression3.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
				# st.write(prediction)

			final_result = get_keys(prediction,prediction_labels)
			st.success("Tweet categorized as : {} using the {} model".format(final_result, model_choice))

	##### Building out the NLP page ####
	if selection == "Natural Language Processing Tool":
		st.markdown('# Natural Language Processing Tool')
		st.markdown('Natural language processing, commonly known as NLP, is a field of artificial intellegence about the interaction between computers and humans using natural language. The objective of NLP is for the computer to read, understand and derive meaning from human languages.')
		st.markdown('The following text processing tools can be viewed on your input text below:\n'
					'- **Tokenization** - Listing each word and punctuation \n'
					'- **Lemmatization** - Returns single base form of a word \n'
					'- **Named-entity recognition (NER)** - Locate and classify entities in categories such as person names and organisations\n'
					'- **Parts of Speech tags (POS)** - The identification of words as nouns, verbs, adjectives, etc.')
		st.markdown('Enter your text below to see how text are processed using the Spacy library.')


		nlp_text = st.text_area("","Type Here")
		nlp_task = ["Tokenization","Lemmatization","NER","POS Tags"]
		task_choice = st.selectbox("Choose NLP Task",nlp_task)
		
		docx = nlp(nlp_text)
		lemma = [word.lemma_ for word in docx]
		token = [word.text for word in docx]
		tag = [word.tag_ for word in docx]
		depend = [word.dep_ for word in docx]
		pos = [token.pos_ for token in docx ]
		
		if st.button("Analyze"):

			if task_choice == 'Tokenization':
				token_df =pd.DataFrame(token, columns = ['Tokens'])
				st.dataframe(token_df)
			elif task_choice == 'Lemmatization':
				lemma_df = pd.DataFrame(zip(token, lemma), columns=['Tokens', 'Lemma'])
				st.dataframe(lemma_df)
			elif task_choice == 'NER':
				html = displacy.render(docx,style="ent")
				html = html.replace("\n\n","\n")
				st.write(HTML_WRAPPER.format(html),unsafe_allow_html=True)
			elif task_choice == 'POS Tags':
				pos_df=pd.DataFrame(zip(token, tag, depend), columns=['Tokens', 'Tag', 'Dependency'])
				st.dataframe(pos_df)

		#NLP table	
		st.markdown('## View table of NLP results')
		st.markdown("Select 'View Table' to view a table of the tokens, lemma and POS tags of your text.")
		if st.button("View Table"):
			docx = nlp(nlp_text)
			table_df = pd.DataFrame(zip(token,lemma,pos),columns=['Tokens','Lemma','POS'])
			st.dataframe(table_df)

		#Word cloud
		st.markdown('## Generate text Word Cloud')
		st.markdown("Select 'Generate Word Cloud' to view a word cloud of the most common words in your text")
		if st.button("Generate Word Cloud"):
			wordcloud =  WordCloud().generate(nlp_text)
			plt.imshow(wordcloud)
			plt.axis("off")
			st.pyplot()

	##### Building out the EDA page #####
	
	if selection == "Exploratory Data Analysis":
		# You can read a markdown file from supporting resources folder
		st.markdown("# Exploratory Data Analysis")
		st.markdown('This page discusses the Exploratory Data Analysis done on the Twitter data received to analyse and to build predictive machine learning models. Here you will find some of the insights from exploring the data as well as visualisations to describe some of our findings.')		

		#Sentiment Description
		st.markdown("## Sentiment Description")
		st.markdown("The table displays the description of each sentiment category.")
		# Image
		st.image(Image.open(os.path.join("resources/sentiment_description.png")))
		
		# Show dataset
		st.markdown("## Raw Twitter data and labels")
		st.markdown("Select the checkbox to view the original data")
		if st.checkbox('Show raw dataset'): # data is hidden if box is unchecked
			st.dataframe(raw_df) # will write the df to the page
		
		# Dimensions
		st.markdown("## Dataframe Dimensions")
		st.markdown("Select the buttons below to view the number of rows and columns for the raw dataset")
		data_dim = st.radio('Select dimension',('All','Rows','Columns'))
		if data_dim == 'All':
			st.text("Showing Shape of Entire Dataframe")
			st.info(raw_df.shape)
		if data_dim == 'Rows':
			st.text("Showing Length of Rows")
			st.info(raw_df.shape[0])
		if data_dim == 'Columns':
			st.text("Showing Length of Columns")
			st.info(raw_df.shape[1])

		# Count of labels
		st.markdown("## Sentiment labels")
		st.markdown("Below is a table displaying the count of each sentiment in the dataset. Majority of the tweets are positive(1) towards climate change. The leaset amount of tweets are negative(-1). This means that we have an unbalanced dataset that might have an affect on our prediction models. Select 'Show Bar Graph' to view this information visually.")
		bar_info = pd.DataFrame(raw_df['sentiment'].value_counts(sort=False))
		bar_info.reset_index(level=0, inplace=True)
		bar_info.columns = ['Sentiment','Count']
		bar_info['Percentage'] = [(i/len(raw_df['sentiment'])*100) for i in bar_info['Count']]
		st.dataframe(bar_info[['Sentiment','Count']])

		# Bar Graph
		if st.button("Show Bar Graph"):
			sns.set(font_scale=.6)
			sns.set_style('white')
			plot = sns.catplot(x="sentiment", kind="count", edgecolor=".6",palette="pastel",data=df_with_metadata,label='small')
			plot.fig.set_figheight(2.5)
			plt.xlabel("Sentiment")
			plt.ylabel("Count")
			plt.title("Sentiment counts")
			st.pyplot(bbox_inches="tight")


		#Clean dataset
		st.markdown("# Processed dataset")

		# Clean tweets
		st.markdown("Select the checkbox to view the processed data with additional information extracted from the text.")
		if st.checkbox('Show processed dataset'): # data is hidden if box is uncheckedz
			st.dataframe(df_with_metadata)	

		# Retweets
		st.markdown("## Retweets")
		st.markdown("The first thing we look at are the retweets. We find that just over 60% of the tweets are retweets. There is a possibility that some of these retweets are duplicates. We also look at the top 5 most retweeted tweets and how many times they were retweeted.")

		valuecounts = df_with_metadata['retweet'].value_counts()
		st.write('No: ', round(valuecounts[1]/len(df_with_metadata['retweet'])*100,2),'%')
		st.write('Yes: ', round(valuecounts[0]/len(df_with_metadata['retweet'])*100,2),'%')
		#Bar graph of number of rewteets
		sns.set(font_scale=.6)
		sns.set_style('white')
		plot = sns.catplot(x="retweet", kind="count", edgecolor=".6",palette="pastel",data=df_with_metadata);
		plt.xlabel("Retweet")
		plt.ylabel("Count")
		plt.title("Retweet count")
		plot.fig.set_figheight(2.5)
		st.pyplot(bbox_inches="tight")	
		
		#View the top 10 retweeted tweets
		tdf = pd.DataFrame(df_with_metadata['message'].astype(str).value_counts())
		st.dataframe(tdf[:6])

		# Word Cloud - Static wordcloud
		st.markdown('## Hashtags and Mentions')
		st.markdown('We can tell a lot from the sentiment of tweets by looking at the hashtags or mentions that are used. Select an option from the dropdown menu to view a Word Cloud of the most common mentions and hashtags. You can also view the top mentions and hashtags per category.')
		wc_options = ["Top Hashtags", "Top Mentions", "Top Hashtags by Sentiment","Top Mentions by Sentiment"]
		wc_selection = st.selectbox("Select Word Cloud OPtion", wc_options)
		
		if wc_selection=="Top Hashtags":
			newsimg = Image.open('resources/imgs/TopHashWC.png')
			st.image(newsimg)
		elif wc_selection=="Top Mentions":
			newsimg = Image.open('resources/imgs/TopMentionWC.png')
			st.image(newsimg)
		elif wc_selection=="Top Hashtags by Sentiment":
			newsimg = Image.open('resources/imgs/HashtagCatWC.png')
			st.image(newsimg,  width=700)
		elif wc_selection=="Top Mentions by Sentiment":
			newsimg = Image.open('resources/imgs/MentionsCatWC.png')
			st.image(newsimg, width=700)
		
		st.markdown('---')
		st.markdown('Select a checkbox below to view a table of the top hashtags or mentions for each category and how often they appear:')
		if st.checkbox('View top hashtags table'):
			st.dataframe(top_hashtags_df)
		if st.checkbox('View top mentions table'):
			st.dataframe(top_mentions_df)
		st.markdown('---')

		st.markdown('After looking at the top mentions and hashtags from the wordcloud above and doing some research, we can make a couple of assumptions: \n\n'
					'- This data seems to be taken from Americans around the time of the 2016 US presidential elections.\n\n'
					'- **@realDonaldTrump** is the top mentioned account. \n\n'
					'- **#Climatechange**, **#climate**, and **#Trump#** are the three most used hashtags')


		# Most Common Words
		st.markdown("## Most Common Words")
		st.markdown('If we look at the most common words used, we see the following:\n\n'
		"- For all the words : **climate**, **change**, **rt**, **global**,and **warming** all are at the top of the word counts. These are top   occurences throughout all categories.\n\n"
		"- For negative words : **science**, **cause**, **real**, and **scam** stand out as top words that are distinct to negative.\n\n"
		"- For news words : **fight**, **epa**, **pruit**, **scientist**,and **new** stand out as top words that are distinct to news.")
		st.dataframe(top_words_df)		

		# Conclusion
		st.markdown("## Conclusion")
	
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
