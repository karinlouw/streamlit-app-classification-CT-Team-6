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

# To Improve speed and cache data
@st.cache(persist=True)
def explore_data(dataset):
	df = pd.read_csv(os.path.join(dataset))
	return df 

dataframe=pd.DataFrame(explore_data(raw))
pd.set_option('display.max_colwidth', -1)

# Sentiment Dictionary
def get_keys(val,my_dict):
	for key, value in my_dict.items():
		if value == value:
			return key

# The main function where we will build the actual app
def main():
	'''Creates a main title and subheader on your page -
	these are static across all pages'''
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	options = ["Prediction", "Information", "NLP"]
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
		data = explore_data(raw)
		
		if st.checkbox('Show raw dataset'): # data is hidden if box is unchecked
			st.dataframe(dataframe) # will write the df to the page
		if st.checkbox("Preview DataFrame"):
			data = dataframe[['sentiment','message']]
			if st.button("Head"):
				st.write(data.head())
			if st.button("Tail"):
				st.write(data.tail())

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
		bar_info = pd.DataFrame(data['sentiment'].value_counts(sort=False))
		st.write(bar_info)
		if st.button("Show Histogram"):
			bar_plot=bar_info.plot(kind = 'bar',figsize=(6,3),legend=False)
			st.pyplot()



	# Building out the "NLP" page
	if selection == "NLP":
		st.info("Natural Language Processing")
		tweet_text = st.text_area("Enter Text","Type Here")
		nlp_task = ["Tokenization", "NER","Lemmatization","POS Tags"]
		task_choice = st.selectbox("Chooose NLP Task",nlp_task)
		if st.button("Analyze"):
			st.info("Orinal Text".format(tweet_text))

			docx = nlp(tweet_text)
			if task_choice == 'Tokenization':
				result = [token.text for token in docx ]
			elif task_choice == 'Lemmatization':
				result = ["'Token':{},'Lemma':{}".format(token.text,token.lemma_) for token in docx]
			elif task_choice == 'NER':
				result = [(entity.text,entity.label_)for entity in docx.ents]
			elif task_choice == 'POS Tags':
				result = ["'Token':{},'POS':{},'Dependency':{}".format(word.text,word.tag_,word.dep_) for word in docx]

			st.json(result)

		if st.button("Tabulize"):
			docx = nlp(tweet_text)
			c_tokens = [token.text for token in docx ]
			c_lemma = [token.lemma_ for token in docx ]
			c_pos = [token.pos_ for token in docx ]

			new_df = pd.DataFrame(zip(c_tokens,c_lemma,c_pos),columns=['Tokens','Lemma','POS'])
			st.dataframe(new_df)

		if st.checkbox("WordCloud"):
			c_text = tweet_text
			wordcloud = WordCloud().generate(c_text)
			plt.imshow(wordcloud,interpolation='bilinear')
			plt.axis("off")
			st.pyplot()		
# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
