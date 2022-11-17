import pandas as pd
import json
import nltk
from library_project import tokenizer

from typing import Optional
from fastapi import FastAPI
from DataModel import DataModel
from joblib import load, dump
from typing import List
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn import svm
from sklearn.metrics import precision_score, recall_score, f1_score
from fastapi.responses import HTMLResponse
from sklearn.feature_extraction.text import TfidfVectorizer

app = FastAPI()

@app.get("/", response_class= HTMLResponse)
def read_root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta http-equiv="X-UA-Compatible" content="IE=edge" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <link
        rel="stylesheet"
        href="https://cdn.jsdelivr.net/npm/bootstrap@4.3.1/dist/css/bootstrap.min.css"
        integrity="sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T"
        crossorigin="anonymous" />

        <title>Proyecto BI</title>
        <script lang="JavaScript">
        function sendBodyToPost(form){
            fetch('http://localhost:8000/predict',{
            method: 'POST',
            headers: {
                'Accept':'application/json',
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                "text":form.inputText.value,
                "classification":null
            })
            })
            .then(response => response.json())
            .then(response => console.log(JSON.stringify(response)))
        }
        </script>
    </head>
    <body>
        <div class="card text-center">
        <div class="card-header">Proyecto BI - API</div>
        <div class="card-body">
            <h5 class="card-title">Prevención de suicidios</h5>
            <p class="card-text">
            Ingrese un texto el cual quiera predecir si tiene intenciones de
            suicido
            </p>
            <form id="bi_project">
            <textarea
                class="form-control"
                name="inputText"
                rows="3"
                placeholder="Escriba aquí..."
                style="max-width: 60%; margin: auto"></textarea>
            <button
                type="submit"
                class="btn btn-primary mb-3"
                onclick="sendBodyToPost(this.form)">
                Confirm identity
            </button>
            </form>
        </div>
        <div class="card-footer text-muted">Grupo 10 - Los Bffis</div>
        </div>
    </body>
    </html>

    """


@app.post("/predict")
def make_predictions(dataModel: List[DataModel]):
    lista = []
    for i in dataModel:
        lista.append(i.dict())
    df = pd.DataFrame(lista)
    df.columns = dataModel[0].columns()

    model = load("assets/modelo.joblib")
    result = model.predict(df)
    print(result)
    dic = {"resultado": result.tolist()}
    dic = {"resultado": []}
    return dic
