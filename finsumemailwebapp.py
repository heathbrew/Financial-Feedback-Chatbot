import streamlit as st
import csv
import numpy as np
import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import openai
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.util import ngrams

file = 'FF_dataset - Sheet1.csv' # Target CSV file path
with open(file, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    data = list(reader)
datalist = list(data)
index = []
notes =[]
summary =[]
country =[]
datas = []
for i in range(0,len(datalist)):
    index.append(i)
    notes.append(datalist[i][0])
    summary.append(datalist[i][1])
    country.append(datalist[i][2])
    datas.append([notes,summary,country])
# fetcher = {}
# for i in range(len(index)):
#     fetcher[index[i]] = datas[i]
# print(fetcher)


# Create a new directory called "extract"
if not os.path.exists("email"):
    os.makedirs("email")
    


def preprocess(text):
    # Remove special characters, numbers and punctuations
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    
    # Convert text to lowercase
    text = text.lower()
    
    # Tokenize the text
    words = word_tokenize(text)
    
    # Remove stop words
    stop_words = set(stopwords.words("english"))
    words = [word for word in words if word not in stop_words]
    
    return " ".join(words)

def extractive_summarize_text(text, n):
    sentences = text.split(".")
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    scores = np.array(similarity_matrix.sum(axis=0)).flatten()
    top_sentence_indices = scores.argsort()[-n:]
    top_sentences = [sentences[i] for i in sorted(top_sentence_indices)]
    summarized_text = " ".join(top_sentences)
    return summarized_text

# Create a function that generates a response to a user's query
def generate_response(prompt):
    openai.api_key = api_key
    response = openai.Completion.create(
        engine="text-davinci-003",
        # engine="davinci", 	
        prompt= "<|endoftext|>"+ prompt +"\n--\nLabel:",
        # max_tokens=687,
        max_tokens= 256,
        temperature=0.5,
        top_p=1
        #0.22,0.83 , 63 stop
        # 0.22,0.91, 70 - stop
    )
    return response["choices"][0]["text"]

api_key = os.environ["OPENAI_API_KEY"] = "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"
if api_key is None:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")
    
def generate_email(sentiments,notes):
    positive_text = [sentence for sentiment, sentence in sentiments if sentiment == "Positive"]
    negative_text = [sentence for sentiment, sentence in sentiments if sentiment == "Negative"]
    neutral_text = [sentence for sentiment, sentence in sentiments if sentiment == "Neutral/Recommendation"]

    positive_text = "\n".join(positive_text)
    negative_text = "\n".join(negative_text)
    neutral_text = "\n".join(neutral_text)

    email_template = "Subject: {}\n\nDear [Name],\n\nThank you for providing me with the details of your financial situation. I appreciate the opportunity to offer my expert insights.\n\nBased on the information you've shared, I would like to summarize the key points of your financial situation:\n\n{}\n\n{}\n\n{}\n\n I hope that you find this summary helpful. If you have any further questions or would like to discuss your situation in more detail, please don't hesitate to reach out.\n\nBest regards, [Name]"
    email = email_template.format(extractive_summarize_text(notes,1),positive_text, negative_text, neutral_text)

    return email

def get_sentiments(text):
    nltk.download("vader_lexicon")
    sia = SentimentIntensityAnalyzer()
    sentiments = []
    for sentence in nltk.sent_tokenize(text):
        pol_score = sia.polarity_scores(sentence)
        if pol_score['compound'] > 0.05:
            sentiments.append(('Positive', sentence))
        elif pol_score['compound'] < -0.05:
            sentiments.append(('Negative', sentence))
        else:
            sentiments.append(('Neutral/Recommendation', sentence))
    return sentiments



# Prompt = """give comments with respect to 
# 1.investment - should be 40% of total income 
# 2.insurance - should be 10% of total income 
# 3.side hustle/entrepeneurial venture - should be 20% of total income 
# 4. tax - should be 10% of total income 
# 5.risk/crisis - should be 10% of total income 
# 6.purchases - should be 10% of total income 
# being used as headers for """   

Prompt = """give Financial planning advice with respect to 
investment
insurance
side hustle/entrepeneurial venture 
tax 
risk/crisis 
purchases 
"""    
    
from rouge_score import rouge_scorer
def rouge(text1, text2):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(text1,text2)
    return scores



notes = st.text_area("Enter notes", "", key='notes', height=200)
summary = st.text_area("Enter summary", "", key='summary', height=200)
country = st.text_input("Enter your country", "", key='country')

text = notes + summary + country
final = ""



if st.button("Generate Email"):
        try:  
            davinchi = generate_response("give suggestions as a financial advisor"+text + Prompt)
        except Exception as e:
            davinchi = " "
        
        summarized_text = extractive_summarize_text(text + davinchi,10)
        sentiments = get_sentiments(summarized_text)
                    
        email=generate_email(sentiments,notes)
        
        try:
            davinchi = generate_response("Format "+email+"as per"+ summary)
        except Exception as e:
            davinchi = email           
        
        ascii_text = email.encode('ascii', 'ignore').decode()
                    
        rouge_n_score = str(rouge(summarized_text, text))

        # global message
        message = ascii_text+"\n\n"+rouge_n_score
        final = message
    
    # st.write(message)
st.write(str(final))

import smtplib, ssl

port = 587  # For starttls
smtp_server = "smtp.gmail.com"

sender_email = st.text_input("Enter your mail id:")
password = st.text_input("Enter your password:")
#enter your app password - pqdngcgjltavgspx
receiver_email = st.text_input("Enter receiver's mail id:")

# message = st.text_area("Enter your message:", "", key='query', height=200)
# message = final

context = ssl.create_default_context()

with smtplib.SMTP(smtp_server, port) as server:
    server.ehlo()  # Can be omitted
    server.starttls(context=context)
    server.ehlo()  # Can be omitted
    
    if st.button("Send") :
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email,final)   
