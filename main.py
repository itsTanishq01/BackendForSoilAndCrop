from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI(title="AgriSense API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all frontend requests
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
crop_model = joblib.load("Crop.pkl")
soil_health_model = joblib.load("Soil.pkl")

class SoilFeatures(BaseModel):
    Soil_Moisture_: float
    Bulk_Density_g_cm3: float
    Porosity_: float
    Water_Holding_Capacity_: float
    pH_Level: float
    Electrical_Conductivity_dS_m: float
    Organic_Carbon_: float
    Nitrogen_mg_kg: float
    Phosphorus_mg_kg: float
    Potassium_mg_kg: float
    Sulfur_mg_kg: float
    Calcium_mg_kg: float
    Magnesium_mg_kg: float
    Temperature_C: float
    Rainfall_mm: float
    Humidity_: float
    Solar_Radiation_W_m2: float

@app.get("/")
async def home():
    return {
        "api_name": "AgriSense API",
        "status": "active",
        "endpoints": {
            "/predict/soil": "POST - Predict soil health",
            "/predict/crop": "POST - Recommend a crop"
        }
    }

@app.post("/predict/soil")
async def predict_soil(features: SoilFeatures):
    data = np.array([[getattr(features, field) for field in features.__annotations__]])
    soil_health_score = soil_health_model.predict(data)[0]
    return {"Soil_Health_Score": soil_health_score}

@app.post("/predict/crop")
async def recommend_crop(features: SoilFeatures):
    data = np.array([[getattr(features, field) for field in features.__annotations__]])
    recommended_crop = crop_model.predict(data)[0]
    return {"Recommended_Crop": recommended_crop}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
