import pandas as pd
import json
import nltk

from typing import Optional
from fastapi import FastAPI
from DataModel import DataModel
from joblib import load, dump
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn import svm
from sklearn.metrics import precision_score, recall_score, f1_score


app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict")
def make_predictions(dataModel: List[DataModel]):
    lista = []
    for i in dataModel:
        lista.append(i.dict())
    df = pd.DataFrame(lista)
    df.columns = dataModel[0].columns()
    model = load("assets/modelo.joblib")
    result = model.predict(df)
    dic = {"resultado": result.tolist()}
    return dic

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}


@app.post("/retrain")
def make_retrain(dataModel: List[DataModel]):
    lista = []
    print(lista)
    for i in dataModel:
        lista.append(i.dict())
    df = pd.DataFrame(lista)
    df.columns = dataModel[0].columns()
    ret = retrain(df)
    return ret



def retrain(df):
    df.dropna()

    X_train, X_test, y_train, y_test = train_test_split(df, df["class"], test_size=0.2, random_state=1)

    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

    pipeline = Pipeline(
        steps = [
        ('vectorizer', TfidfVectorizer(tokenizer = tokenizer, stop_words = stop_words, lowercase = True, strip_accents= 'ascii')),
        ('classifier', svm.SVC(kernel = 'linear'))
    ])
    pipeline = pipeline.fit(X_train, y_train)
    y_train_tfidf_predict = pipeline.predict(X_train)
    y_test_tfidf_predict = pipeline.predict(X_test)

    filename = './assets/modelo.joblib'

    #Train F1
    precision_train = precision_score(y_train, y_train_tfidf_predict, pos_label = 'suicide')
    recall_train = recall_score(y_train, y_train_tfidf_predict, pos_label = 'suicide')
    f1_train = f1_score(y_train, y_train_tfidf_predict, pos_label = 'suicide')

    #Test F1
    precision_test = precision_score(y_test, y_test_tfidf_predict, pos_label = 'suicide')
    recall_test = recall_score(y_test, y_test_tfidf_predict, pos_label = 'suicide')
    f1_test = f1_score(y_test, y_test_tfidf_predict, pos_label = 'suicide')

    dump(pipeline, filename)
    ret = {"Precision-train": precision_train, "Recall-train": recall_train, "F1-train": f1_train,
            "Precision-test": precision_test, "Recall-test": recall_test, "F1-test": f1_test}
    json_object = json.dumps(ret, indent = 4)
    return json_object



def tokenizer(text: str):
    return word_tokenize(text)