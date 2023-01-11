import firebase_admin
from firebase_admin import credentials,firestore
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
import re
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import svm
import string

df = pd.read_csv("D:/Important files/Aditya/files/major/hate speech/data/train_E6oV3lV.csv.zip")
hate_sp = df[df.label == 1]
normal_sp = df[df.label == 0]

def process_tweet(tweet):
    return " ".join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])", " ",tweet.lower()).split())    
df['processed_tweets'] = df['tweet'].apply(process_tweet)
cnt_non_fraud = df[df['label'] == 0]['processed_tweets'].count()
df_class_fraud = df[df['label'] == 1]
df_class_nonfraud = df[df['label'] == 0]
df_class_fraud_oversample = df_class_fraud.sample(cnt_non_fraud, replace=True)
df_oversampled = pd.concat([df_class_nonfraud, df_class_fraud_oversample], axis=0)

from sklearn.model_selection import train_test_split
X = df_oversampled['processed_tweets']
y = df_oversampled['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify = None)


count_vect = CountVectorizer(stop_words='english')
transformer = TfidfTransformer(norm='l2',sublinear_tf=True)
x_train_counts = count_vect.fit_transform(X_train)
x_train_tfidf = transformer.fit_transform(x_train_counts)

#SVM Model

model = svm.LinearSVC()
model.fit(x_train_tfidf,y_train)
x_test_counts = count_vect.transform(X_test)
x_test_tfidf = transformer.transform(x_test_counts)
predict_svm = model.predict(x_test_tfidf)
def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    text = re.sub('[0-9]+', '', text)
    return text


cred = credentials.Certificate("firebase\serviceAccount.json")
firebase_admin.initialize_app(cred)

db=firestore.client()

docs=db.collection('questions').document("aDiXA2BBZEDU115cV3FL").collection('answer').get()
db.collection('person').delete()
for doc in docs:
    res=doc.to_dict()
    X_new = []
    answer=res["answer"]
    user_review = [ answer]
    for review in user_review:
        X_new.append(process_tweet(review))
    X_new = count_vect.transform(X_new)
    X_new = transformer.transform(X_new)
    X_new = X_new.toarray()
    predict_svm = model.predict(X_new)
    for analysis in predict_svm:
        speech = analysis
    speech = np.where(speech, 'Hate', 'Normal')
    if speech == 'Hate':
        db.collection('questions').document("aDiXA2BBZEDU115cV3FL").collection('answer').document("doc.id").delete()
        print("Deleted")
        print(res["user"]["email"])
