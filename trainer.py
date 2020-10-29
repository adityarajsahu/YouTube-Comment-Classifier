#!usr/bin/env python

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
from dataProcessor import processor

df_psy = pd.read_csv("Dataset/Youtube01-Psy.csv")
df_katyperry = pd.read_csv("Dataset/Youtube02-KatyPerry.csv")
df_lmfao = pd.read_csv("Dataset/Youtube03-LMFAO.csv")
df_eminem = pd.read_csv("Dataset/Youtube04-Eminem.csv")
df_shakira = pd.read_csv("Dataset/Youtube05-Shakira.csv")

df = pd.concat([df_psy, df_katyperry, df_lmfao, df_eminem, df_shakira])
df = df.sample(frac=1)

content, label = processor(df)

train_content = content[0]
val_content = content[1]

train_label = label[0]
val_label = label[1]

classifier = RandomForestClassifier(n_estimators=100, max_depth=2)
classifier.fit(train_content, train_label)
val_score = classifier.score(val_content, val_label)
print("Accuracy on validation dataset : {} %".format(val_score * 100))

filename = 'training1_model.sav'
pickle.dump(classifier, open(filename, 'wb'))