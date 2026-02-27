import logging
from pathlib import Path
from typing import List

import fastapi
import pandas as pd
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator

from challenge.model import DelayModel

logger = logging.getLogger(__name__)

app = fastapi.FastAPI()
model = DelayModel()

# sacadas del dataset — son las 23 aerolíneas que operan en SCL
VALID_OPERA = {
    "Aerolineas Argentinas", "Aeromexico", "Air Canada", "Air France",
    "Alitalia", "American Airlines", "Austral", "Avianca",
    "British Airways", "Copa Air", "Delta Air", "Gol Trans",
    "Grupo LATAM", "Iberia", "JetSmart SPA", "K.L.M.",
    "Lacsa", "Latin American Wings", "Oceanair Linhas Aereas",
    "Plus Ultra Lineas Aereas", "Qantas Airways", "Sky Airline",
    "United Airlines",
}


class FlightData(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

    @validator("OPERA")
    def validate_opera(cls, v):
        if v not in VALID_OPERA:
            raise ValueError(f"aerolínea desconocida: {v}")
        return v

    @validator("TIPOVUELO")
    def validate_tipo(cls, v):
        if v not in ("N", "I"):
            raise ValueError(f"TIPOVUELO debe ser N o I, recibido: {v}")
        return v

    @validator("MES")
    def validate_mes(cls, v):
        if not 1 <= v <= 12:
            raise ValueError(f"MES fuera de rango: {v}")
        return v


class PredictRequest(BaseModel):
    flights: List[FlightData]


@app.on_event("startup")
async def load_model():
    data_path = Path(__file__).resolve().parent.parent / "data" / "data.csv"
    if not data_path.exists():
        logger.warning("No se encontró data en %s, modelo no disponible", data_path)
        return
    data = pd.read_csv(str(data_path), low_memory=False)
    features, target = model.preprocess(data=data, target_column="delay")
    model.fit(features=features, target=target)
    logger.info("Modelo cargado, API lista")


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {"status": "OK"}


@app.post("/predict", status_code=200)
async def post_predict(request: PredictRequest) -> dict:
    df = pd.DataFrame([f.dict() for f in request.flights])
    features = model.preprocess(data=df)
    preds = model.predict(features=features)
    return {"predict": preds}


@app.exception_handler(fastapi.exceptions.RequestValidationError)
async def handle_validation_error(request, exc):
    return JSONResponse(status_code=400, content={"detail": str(exc)})
