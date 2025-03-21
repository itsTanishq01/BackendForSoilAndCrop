from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pickle
import uvicorn
import os

app = FastAPI(title="AgriSense Lite API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATHS = {
    "soil": "Soil.pkl",
    "crop": "Crop.pkl"
}

models = {}

@app.on_event("startup")
async def load_models():
    global models
    try:
        with open(MODEL_PATHS["soil"], "rb") as f:
            models["soil"] = pickle.load(f)
        with open(MODEL_PATHS["crop"], "rb") as f:
            models["crop"] = pickle.load(f)
        print("✅ Models loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading models: {e}")

@app.get("/")
async def home():
    return {
        "api_name": "AgriSense Lite API",
        "status": "active",
        "endpoints": {
            "/predict/soil": "POST - Predict soil health",
            "/predict/crop": "POST - Recommend a crop"
        }
    }

@app.post("/predict/soil")
async def predict_soil(data: dict):
    if models["soil"] is None:
        raise HTTPException(status_code=503, detail="Soil model is not available.")

    input_features = np.array([data[key] for key in data.keys()]).reshape(1, -1)
    prediction = models["soil"].predict(input_features)

    return {"soil_health_score": float(prediction[0])}

@app.post("/predict/crop")
async def recommend_crop(data: dict):
    if models["crop"] is None:
        raise HTTPException(status_code=503, detail="Crop model is not available.")

    input_features = np.array([data[key] for key in data.keys()]).reshape(1, -1)
    prediction = models["crop"].predict(input_features)

    return {"recommended_crop": str(prediction[0])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
