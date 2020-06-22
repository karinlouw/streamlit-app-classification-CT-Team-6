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

import re
from nltk.tokenize import word_tokenize
from collections import Counter

# NLP Packages
import spacy
nlp = spacy.load('en')

# Wordcloud
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# Images
from PIL import Image

# Plotting
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.use("Agg")

# Vectorizer
tweet_vectorizer = open("resources/tfidfvect.pkl","rb")
tweet_cv = joblib.load(tweet_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = "resources/train.csv"
clean = "resources/clean_tweet_df.csv"

# To Improve speed and cache data
@st.cache(persist=True, allow_output_mutation=True)
def explore_data(dataset):
	df = pd.read_csv(os.path.join(dataset))
	return df 

dataframe=pd.DataFrame(explore_data(raw))
pd.set_option('display.max_colwidth', None)

clean_df = pd.DataFrame(explore_data(clean))

# Sentiment Dictionary
def get_keys(val,my_dict):
	for key, value in my_dict.items():
		if value == value:
			return key

# Clean Tweets



# The main function where we will build the actual app
def main():
	'''Creates a main title and subheader on your page -
	these are static across all pages'''
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information"]
	selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the Prediction page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		
		tweet_text = st.text_area("Enter Text","Type Here")
		all_ml_modles= ["LR","LR2"]
		model_choice = st.selectbox("Choose ML model",all_ml_modles)
		prediction_labels = {'anti':-1,'news':0,'pro':1,}
		
		if st.button("Classify"):
			st.text("Original Text:\n{}".format(tweet_text))
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			if model_choice == 'LR':
				predictor = predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
				# st.write(prediction)
			elif model_choice == 'LR2':
				predictor = predictor = joblib.load(open(os.path.join("resources/Logistic_regression.pkl"),"rb"))
				prediction = predictor.predict(vect_text)
				# st.write(prediction)

			final_result = get_keys(prediction,prediction_labels)
			st.success("Tweet categorized as : {}".format(final_result))


	# Building out the "Information" page
	
	if selection == "Information":
		#st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("## Exploratory Data Analysis")
		
		#Sentiment Description
		st.subheader("Sentiment Description")
		# Image
		st.image(Image.open(os.path.join("resources/sentiment_description.png")))
		
		# Show dataset
		st.subheader("Raw Twitter data and label")
		
		if st.checkbox('Show raw dataset'): # data is hidden if box is unchecked
			st.dataframe(dataframe[['sentiment','message']]) # will write the df to the page
		
		# if st.checkbox("Preview DataFrame"):
		# 	data = dataframe[['sentiment','message']]
		# 	if st.button("Head"):
		# 		st.write(data.head())
		# 	if st.button("Tail"):
		# 		st.write(data.tail())

		# Dimensions
		st.subheader("Dataframe Dimensions")
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

		# Histogram - number of labels
		st.subheader("Number of labels")
		bar_info = pd.DataFrame(dataframe['sentiment'].value_counts(sort=False))
		st.write(bar_info)
		if st.button("Show Histogram"):
			bar_plot=bar_info.plot(kind = 'bar',figsize=(5,2),legend=False,fontsize=6)
			st.pyplot()

		#Clean dataset
		st.subheader("Clean dataset")
		# Create a function to clean the tweets
		def cleanTxt(text):
			text = re.sub('@[A-Za-z0–9]+', '', text) #Removing @mentions
			text = re.sub('#', '', text) # Removing '#' hash tag
			text = re.sub('RT[\s]+', '', text) # Removing RT
			text = re.sub(':', '', text) # Removing ':'
			text = re.sub('https?:\/\/\S+', '', text) # Removing hyperlink
			text = text.lower()

			
			return text

		# Clean the tweets
		dataframe['clean_tweets'] = dataframe['message'].apply(cleanTxt)

		if st.checkbox('Show clean dataset'): # data is hidden if box is unchecked
			st.dataframe(dataframe[['sentiment','clean_tweets']])

		if st.button("Generate Wordcloud"):
		
		# Word Cloud
			def gen_wordcloud():
				# Create a dataframe with a column called Tweets
				#df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])
				# word cloud visualization
				allWords = ' '.join([twts for twts in clean_df['clean_tweet']])
				wordCloud = WordCloud(width=700, height=500, random_state=21, max_font_size=110).generate(allWords)
				plt.imshow(wordCloud, interpolation="bilinear")
				plt.axis('off')
				plt.savefig('WC.jpg')
				img= Image.open("WC.jpg") 
				return img

			img=gen_wordcloud()

			st.image(img)

		# Most common words
		st.subheader("Word Frequency")
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
			
		counter_text_list = [i.split() for i in clean_df['clean_tweet']]

		wf = word_freq(counter_text_list, 20)
		st.dataframe(wf)

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
