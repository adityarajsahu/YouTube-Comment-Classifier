#!usr/bin/env python

from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pickle
import os

comment = [input("Enter the YouTube comment to be verified : ")]

os.chdir("SavedModel/")
loaded_vocab = pickle.load(open('vectorizer1.pk', 'rb'))
loaded_vectorizer = CountVectorizer(vocabulary=loaded_vocab)
vectorized_comment = loaded_vectorizer.fit_transform(np.array(comment))

loaded_model = pickle.load(open('training1_model.sav', 'rb'))
prediction_score = loaded_model.predict(vectorized_comment)

if prediction_score < 0.5:
    print("It is not a spam comment.")
else:
    print("It's a spam comment. Please delete it.")
