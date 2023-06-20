# Natural Lenguage Processing Model for Suicide Prevention

## What is this? 

This is a very useful tool to predict which users have depression or have attempted suicide. In this case, Text Analytics (TA) techniques will be used with Machine Learning (AI) to be able to analyze user comments on Reddit that meet these characteristics in order to train a model that predicts whether a person, depending on their comments on this social network, suffers from any of these diseases.

---

## What we have used to built this app?
- Python Jupyter Notebooks
- NLTK
- FastAPI
- CountVectorizer, TfidfVectorizer
- Numpy
- Pandas
- Pyplot

## Algorithms used:
- Random Forest
- K-Nearest Neighbors (KNN)
- Support Vector Machines (SVM)

## Data preparation and understanding:

The data, given the theme we chose, is made up of a 3-column structure, which are: Unnamed 0, which represents the ID of the record; text, which represents the text to parse; class, which represents the category assigned to the text that can be between (Suicide, Non-suicide).

The text and category is in the English language
 
Steps for data preparation:
1. Stopwords: We define stopwords (words that do not have a meaning) for the English language, since this is the language that, in this case, is used by the texts within the dataframe.

To carry out this procedure, we must previously import the nltk library. This library provides us with functions for working with human language data.

Afterwards, we download the punk package and the stopwords to later save the stopwords in the English language in a variable.

2. Tokenizer: For tokenization, we define a function called tokenizer and we use word_tokenize, which is a class offered by NLTK to divide words into tokens that make up a text.
3. Reading data: We use the pandas library to read the data that is in csv format.
4. Understanding the data: We use the data.shape function to know the number of rows and columns that the dataset has. We also use data.columns to know the names of the columns and their data type.
5. Eliminate columns: We decided to eliminate the first column (Unnamed: 0) since it does not generate any value to the model that we are going to carry out.
6. Identify the classification of the "class" column: Here we want to see the percentage in the dataset of the two categories of the 'class' column, that is, non-suicide and suicide.
7. Selection of the model to vectorize: We select the TfidfVectorizer model, since in this case it is cheaper and faster, compared to the BoW model. However, both were implemented and tested, which were later removed because the results of the trained models were slightly lower than those used with TF-IDF.
Note: Remembering that the vectorizer is a structure that allows us to have a more organized index of the tokens that were found in the phrases and their occurrence in the dataset and the different records. Allowing the model to do a training to find patterns and to be able to classify a text.
8. Separate into datasets: We divide the data into Train and Test dataset.
9. Using the TfidfVectorizer: We apply the Train dataset to the TfidfVectorizer.
10. Vocabulary length check: We get the vocabulary length of the dataset using the max_df parameter in the TfidfVectorizer class, since it reduces the number of times a word appears in the dataset by not using it as a token.

## Results of the models:

### Random Forest:
#### Scores:

*Train*
Precision: 0.9953204413040784
Recall: 0.9387093002864324
F1: 0.9661863361511387

*Test*
Precision: 0.8547596400087802
Recall: 0.6828783538902203
F1: 0.759212322090076

-- 

### SVM:
#### Scores:

*Train*
Precision: 0.9476271294991674
Recall: 0.8649237154381247
F1: 0.9043886189297394

*Test*
Precision: 0.8752826197943258
Recall: 0.7015256912375051
F1: 0.7788305535725875

--

### KNN:
#### Scores:

Didn't give the expected results, weren't good enough.


