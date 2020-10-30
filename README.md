# YouTube-Comment-Classifier

This aim of this project is to train a Random Forest classifier to classify YouTube comments as spam or non-spam.

## About Random Forest Algorithm

Random forest or random decision forest is a method that operates by constructing multiple decision trees during training phase. The decision of the majority of the trees is chosen by random forest as the final decision.

![Random Forest Image](img/random_forest_image.png)

## Dataset

The dataset for this classifier has been taken from **YouTube Spam Collection Data Set**, which belongs to UCI Machine Learning Repository.

```
curl -O https://archive.ics.uci.edu/ml/machine-learning-databases/00380/YouTube-Spam-Collection-v1.zip

unzip YouTube-Spam-Collection-v1.zip
```

There are 5 different comma separated value files containing a total of 1956 comments and class labels. There are 5 columns each csv file :-
- Comment ID
- Author
- Date
- Content
- Class

## Dependencies

Some packages and libraries have to be installed for running the codes of this repository. To install these dependencies, run this command in terminal :-

```
pip install -r requirements.txt
```

## Content of this repository

- **Dataset** - _contains the csv files._
- **SavedModel** - _contains the saved parameters of the classifier._
- **__pycache__/** - _contains the bytecode._
- **img** - _contains the image for README.md file._
- **LICENSE** - _License information._
- **README.md** - _Documentation of the repository._
- **classifier.py** - _Code to test our classifier._
- **dataProcessor.py** - _Code to process the data before training the classifier._
- **requirements.txt** - _Contains the libraries to be installed._
- **trainer.py** - _Code to train our classifier._

## Accuracy of classifier

Accuracy on validation set - 85.5%

Accuracy on test set - 42.857%
