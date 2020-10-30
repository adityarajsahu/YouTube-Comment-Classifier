#!usr/bin/env python

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle


def processor(df):
    content = []
    label = []
    
    df_content = df['CONTENT']
    df_label = df['CLASS']
    
    vectorizer = CountVectorizer(ngram_range=(0, 2), lowercase=True)
    df_vectorized = vectorizer.fit_transform(df_content)
    
    filename_vec = 'SavedModel/vectorizer1.pkl'
    pickle.dump(vectorizer, open(filename_vec, 'wb'))
    
    transformer = TfidfTransformer()
    df_transformed = transformer.fit_transform(df_vectorized)
    filename_trans = 'SavedModel/transformer1.pkl'
    pickle.dump(transformer, open(filename_trans, 'wb'))
    
    dfContent_train = df_transformed[:1700]
    dfContent_val = df_transformed[1700:1900]
    dfContent_test = df_transformed[1900:]
    
    dfLabel_train = df_label[:1700]
    dfLabel_val = df_label[1700:1900]
    dfLabel_test = df_label[1900:]
    
    content = [dfContent_train, dfContent_val, dfContent_test]
    label = [dfLabel_train, dfLabel_val, dfLabel_test]
    return content, label
