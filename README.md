# Climate Change Sentiment Analysis
#### by CT6 @ Explore Data Science Academy 

## 1) Overview

This Repo contains the Streamlit app related to a climate change twitter sentiment analysis project located here ----> https://github.com/jonnybegreat/Team_6_CPT 

#### 2) What is Streamlit?

In its own words:
> Streamlit ... is the easiest way for data scientists and machine learning engineers to create beautiful, performant apps in only a few hours!  All in pure Python. All for free.

> It’s a simple and powerful app model that lets you build rich UIs incredibly quickly.

Streamlit takes away much of the background work needed in order to get a platform which can deploy your models to clients and end users. Meaning that you get to focus on the important stuff (related to the data), and can largely ignore the rest. This will allow you to become a lot more productive.  

##### Description of files

For this repository, we are only concerned with a single file:

| File Name              | Description                       |
| :--------------------- | :--------------------             |
| `base_app.py`          | Streamlit application definition. |

## 2) Usage Instructions

#### 2.2) Running the Streamlit web app on your local machine

To do this, follow the steps below by running the given commands within a Git bash (Windows), or terminal (Mac/Linux):

 1. Ensure that you have the prerequisite Python libraries installed on your local machine:

 ```bash
 pip install -U streamlit numpy pandas scikit-learn
 pip install textblob
 pip install spacy
 pip install gensim
 pip install sumy
 ```
 2. Fork and clone this repo
 3. Navigate to the base of the cloned repo, and start the Streamlit app.

 ```bash
 cd classification-predict-streamlit-template/
 streamlit run base_app.py
 ```

 If the web server was able to initialise successfully, the following message should be displayed within your bash/terminal session:

```
  You can now view your Streamlit app in your browser.

    Local URL: http://localhost:8501
    Network URL: http://192.168.43.41:8501
```

Once the app has opened you can play around with it! Type in your own tweets and use our model to predict the sentiment, have a look at some interesting insights we found while looking through the twitter data, learn what stemming, lemmatisatin, part of speech tagging and named entity recognition is all about!

