#!usr/bin/env python

import pickle
import pandas as pd

testdf = pd.read_csv('Dataset/testData.csv')
testContent = testdf['CONTENT']
testLabel = testdf['CLASS']

loaded_vectorizer = pickle.load(open('SavedModel/vectorizer1.pkl', 'rb'))
vectorized_content = loaded_vectorizer.transform(testContent)

loaded_transformer = pickle.load(open('SavedModel/transformer1.pkl', 'rb'))
transformed_content = loaded_transformer.transform(vectorized_content)

loaded_model = pickle.load(open('SavedModel/training1_model.pkl', 'rb'))
predictions = loaded_model.predict(transformed_content)

correct_predictions = 0
for i in range(len(testLabel)):
    if predictions[i] == testLabel[i]:
        correct_predictions = correct_predictions + 1
    else:
        pass

testAccuracy = correct_predictions/len(predictions)
print("Accuracy on Test Dataset : {:.3f}%".format(testAccuracy * 100))