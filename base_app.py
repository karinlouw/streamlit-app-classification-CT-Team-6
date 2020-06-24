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

# Preprocessing
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from collections import Counter

# NLP Packages
from textblob import TextBlob 
import spacy
from spacy_streamlit import visualize_parser
nlp = spacy.load('en')
from gensim.summarization import summarize

# Wordcloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# Images
from PIL import Image

# Plotting
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")

# Sumy Summary Pkg
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer


# Function for Sumy Summarization
def sumy_summarizer(docx):
	parser = PlaintextParser.from_string(docx,Tokenizer("english"))
	lex_summarizer = LexRankSummarizer()
	summary = lex_summarizer(parser.document,3)
	summary_list = [str(sentence) for sentence in summary]
	result = ' '.join(summary_list)
	return result

# Vectorizer
tweet_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(tweet_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = "resources/train.csv"
clean = "resources/clean_tweet_df.csv"

# To Improve speed and cache data
@st.cache(allow_output_mutation=True)
def explore_data(dataset):
	df = pd.read_csv(os.path.join(dataset))
	return df 

dataframe=pd.DataFrame(explore_data(raw))
pd.set_option('display.max_colwidth', None)

raw_df = pd.DataFrame(explore_data(raw))
clean_df = pd.DataFrame(explore_data(clean))

# Sentiment Dictionary
@st.cache
def get_keys(val,my_dict):
	for key, value in my_dict.items():
		if value == value:
			return key

#Contraction dictionary
contraction_dict = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "i'd": "I would",
  "i'd've": "I would have",
  "i'll": "I will",
  "i'll've": "I will have",
  "i'm": "I am",
  "i've": "I have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have"
}

#Contraction function
@st.cache
def lookup_dict(text, dictionary):
    for word in text.split():
        if word.lower() in dictionary:
            if word.lower() in text.split():
                text = text.replace(word, dictionary[word.lower()])
    return text

# Function to clean the tweets
@st.cache(allow_output_mutation=True)
def clean_text(text):
	text = re.sub('@[A-Za-z0–9]+', '', text) #Removing @mentions
	text = re.sub('#', '', text) # Removing # hash tag
	text = re.sub('RT[\s]+', '', text) # Removing RT
	text = re.sub(':', '', text) # Removing ':'
	text = re.sub('https?:\/\/\S+', '', text) # Removing hyperlink
	text = text.lower() #Change to 
	text = word_tokenize(text)
	return text

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

#Function to remove stopwords:
@st.cache
def remove_stopwords(text):
	return [word for word in text if word not in stopwords.words('english')]

#Function to generate wordcloud:
@st.cache
def gen_wordcloud(df):
	"""
	Word Cloud
	"""
	allWords = ' '.join([twts for twts in df['clean_tweet']])
	wordCloud = WordCloud(width=700, height=500, random_state=21, max_font_size=130).generate(allWords)
	plt.imshow(wordCloud, interpolation="bilinear")
	plt.axis('off')
	plt.savefig('resources/imgs/WC.jpg')
	img= Image.open("resources/imgs/WC.jpg") 
	return img

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
	options = ["Prediction", "NLP", "Information"]
	selection = st.sidebar.selectbox("Choose Option", options)

	##### Building out the Prediction page ####
	if selection == "Prediction":
		st.info("Prediction with Machine Learning Models")
		raw_text = st.text_area("Enter Text","Type Here")		
		

		# Model Prediction

		#Select model
		all_ml_modles= ["LR","LR3", "LR3"]
		model_choice = st.selectbox("Select base ML model",all_ml_modles)
		
		prediction_labels = {'anti':-1,'news':0,'pro':1,}
		
		if st.button("Classify"):
			st.text("Original Text:\n{}".format(raw_text))
			vect_text = tweet_cv.transform([raw_text]).toarray()

			if model_choice == 'LR':
				predictor = predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
				# st.write(prediction)
			elif model_choice == 'LR2':
				predictor = predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl2"),"rb"))
				prediction = predictor.predict(vect_text)
				# st.write(prediction)
			elif model_choice == 'LR3':
				predictor = predictor = joblib.load(open(os.path.join("resources/Logistic_regression3.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
				# st.write(prediction)

			final_result = get_keys(prediction,prediction_labels)
			st.success("Tweet categorized as : {} using the {} model".format(final_result, model_choice))

	##### Building out the NLP page ####
	if selection == "NLP":
		st.markdown('# Natural Language Processing Tool')

		nlp_text = st.text_area("Enter Text To Analyze","Type Here")
		nlp_task = ["Tokenization","Lemmatization","NER","POS Tags"]
		task_choice = st.selectbox("Choose NLP Task",nlp_task)
		if st.button("Analyze"):
			st.info("Original Text::\n{}".format(nlp_text))

			docx = nlp(nlp_text)
			if task_choice == 'Tokenization':
				result = [token.text for token in docx ]
			elif task_choice == 'Lemmatization':
				result = ["'Token':{},'Lemma':{}".format(token.text,token.lemma_) for token in docx]
			elif task_choice == 'NER':
				result = [(entity.text,entity.label_)for entity in docx.ents]
			elif task_choice == 'POS Tags':
				result = ["'Token':{},'POS':{},'Dependency':{}".format(word.text,word.tag_,word.dep_) for word in docx]

			st.json(result)

		#NLP table	
		st.markdown('# View NLP table')
		if st.button("View Table"):
			docx = nlp(nlp_text)
			c_tokens = [token.text for token in docx ]
			c_lemma = [token.lemma_ for token in docx ]
			c_pos = [token.pos_ for token in docx ]

			new_df = pd.DataFrame(zip(c_tokens,c_lemma,c_pos),columns=['Tokens','Lemma','POS'])
			st.dataframe(new_df)


	##### Building out the "Information" page #####
	
	if selection == "Information":
		# You can read a markdown file from supporting resources folder
		st.markdown("# Exploratory Data Analysis")
		
		#Sentiment Description
		st.markdown("### Sentiment Description")
		# Image
		st.image(Image.open(os.path.join("resources/sentiment_description.png")))
		
		# Show dataset
		st.markdown("# Raw Twitter data and labels")
		
		if st.checkbox('Show raw dataset'): # data is hidden if box is unchecked
			st.dataframe(raw_df) # will write the df to the page
		
		# Dimensions
		st.markdown("# Dataframe Dimensions")
		data_dim = st.radio('Choose dimensions to display',('All','Rows','Columns'))
		if data_dim == 'All':
			st.text("Showing Shape of Entire Dataframe")
			st.info(dataframe.shape)
		if data_dim == 'Rows':
			st.text("Showing Length of Rows")
			st.info(dataframe.shape[0])
		if data_dim == 'Columns':
			st.text("Showing Length of Columns")
			st.info(dataframe.shape[1])

		# Piechart - percentage of labels
		st.markdown("# Sentiment labels")
		bar_info = pd.DataFrame(dataframe['sentiment'].value_counts(sort=False))
		bar_info.reset_index(level=0, inplace=True)
		bar_info.columns = ['Sentiment','Count']
		st.dataframe(bar_info)

		if st.button("Show Chart"):
			bar_plot=bar_info.plot(kind = 'bar',figsize=(5,2),legend=False,fontsize=6)
			st.pyplot()

		#Clean dataset
		st.markdown("# Clean dataset")

		# Clean tweets
		if st.checkbox('Show clean dataset'): # data is hidden if box is unchecked
			st.dataframe(clean_df[['sentiment','clean_tweet']])
		
		# Word Cloud - Static wordcloud
		st.markdown('# Word Clouds')
		st.markdown('#### General Word Cloud')
		newsimg = Image.open('resources/imgs/WC.png')
		st.image(newsimg)

		st.markdown('### News, Pro and Anti Word Clouds')
		wc_options = ["News", "Pro", "Anti"]
		wc_selection = st.selectbox("Select Word Cloud Sentiment Option", wc_options)

		if wc_selection=="News":
			newsimg = Image.open('resources/imgs/newsWC.png')
			st.image(newsimg)
		elif wc_selection=="Pro":
			newsimg = Image.open('resources/imgs/proWC.png')
			st.image(newsimg)
		else:
			newsimg = Image.open('resources/imgs/antiWC.png')
			st.image(newsimg)

		# Most common words
		st.markdown("# Word Frequency")
			
		counter_text_list = [i.split() for i in clean_df['clean_tweet']]

		wf = word_freq(counter_text_list, 21)
		wf.columns = ['Word','Frequency']

		st.dataframe(wf)

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
