import pandas as pd

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from joblib import load, dump
from nltk.tokenize import word_tokenize
from fastapi.responses import HTMLResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi.middleware.cors import CORSMiddleware





app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
            <form id="form" method="post" action="/predict">
            <textarea
                class="form-control"
                name="text"
                rows="3"
                placeholder="Escriba aquí..."
                style="max-width: 60%; margin: auto"></textarea>
            <button type="submit" class="btn btn-primary mb-3">Predict</button>
            </form>
        </div>
        <div class="card-footer text-muted">Grupo 10 - Los Bffis</div>
        </div>
    </body>
    </html>

    """


@app.post("/predict")
async def make_predictions(request: Request ):
    lista = [dict(jsonable_encoder(await request.form()))]
    print(lista)
    df = pd.DataFrame(lista)
    df.columns = ["text"]
    print(df)
    model = load("assets/modelo.joblib")
    result = model.predict(df)
    print(result)
    dic = {"resultado": result.tolist()}
    return dic
