from fastapi import FastAPI
import uvicorn
from typing import List
from pydantic import BaseModel
from model import predict_model, train_model
from genres import genre_mapping
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MusicFeatures(BaseModel):
    features: List[float]


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/predict")
async def predict_genre(music_features: MusicFeatures):
    features = music_features.features
    prediction = predict_model(features)
    print(features)
    return {"predicted_genre": prediction, "genre_mapping": genre_mapping.get(prediction)}


@app.get("/train")
async def train():
    return train_model()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
