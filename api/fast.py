import os
import joblib
import pandas as pd


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

#Load Model
model = joblib.load(os.path.join("assets", "model.joblib"))

# define a root `/` endpoint
@app.get("/")
def index():
    return {"Status": "Up and running"}

# Implement a /predict endpoint
@app.get("/predict")
def predict(acousticness: float,
            danceability: float,
            duration_ms: int,
            energy: float,
            explicit: int,
            id: object,
            instrumentalness: float,
            key: int,
            liveness: float,
            loudness: float,
            mode: int,
            name: object,
            release_date: object,
            speechiness: float,
            tempo: float,
            valence: float,
            artist: object):
    X_input = pd.DataFrame({'acousticness': acousticness,
                           'danceability': danceability,
                           'duration_ms': duration_ms,
                           'energy': energy,
                           'explicit': explicit,
                           'id': id,
                           'instrumentalness': instrumentalness,
                           'key': key,
                           'liveness': liveness,
                           'loudness': loudness,
                           'mode': mode,
                           'name': name,
                           'release_date': release_date,
                           'speechiness': speechiness,
                           'tempo': tempo,
                           'valence': valence,
                           'artist': artist}, index=['id'])
    # Make a prediction
    y_pred = model.predict(X_input)[0]
    print(y_pred)
    return {
        "artist": artist,
        "name": name,
        "popularity": int(y_pred)
        }
