#!usr/bin/env python

from sklearn.feature_extraction.text import CountVectorizer
import os
import pickle


def processor(df):
    content = []
    label = []
    
    df_content = df['CONTENT']
    df_label = df['CLASS']
    
    vectorizer = CountVectorizer(ngram_range=(0, 2), lowercase=True, decode_error="replace")
    df_token = vectorizer.transform(df_content)
    os.chdir("SavedModel/")
    filename = 'vectorizer1.pk'
    pickle.dump(vectorizer, open(filename, 'wb'))
    
    df_vector_train = df_token[:1700]
    df_vector_val = df_token[1700:]
    
    df_label_train = df_label[:1700]
    df_label_val = df_label[1700:]
    
    content = [df_vector_train, df_vector_val]
    label = [df_label_train, df_label_val]
    return content, label