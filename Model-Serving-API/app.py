import joblib
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import uvicorn
from starlette.responses import JSONResponse
import warnings

app = FastAPI()
templates = Jinja2Templates(directory="templates")
warnings.filterwarnings('ignore')

# Load the trained Linear Regression model
model = joblib.load("artifacts/LR.pickle")
scaler = joblib.load("artifacts/Scaler.pickle")

# Brand, Body, and Engine_Type options
brands = ["BMW", "Mercedes_Benz", "Mitsubishi", "Renault", "Volkswagen", "Toyota"]
bodies = ['sedan', 'crossover', 'vagon', 'other', 'hatch']
engine_types = ['Gas', 'Petrol', 'Diesel']


@app.get("/", response_class=HTMLResponse)
async def predict(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "brands": brands, "bodies": bodies,
                                                     "engine_types": engine_types})


@app.post("/predict/", response_class=JSONResponse)
async def predict(request: Request, mileage: float = Form(...), EngineV: float = Form(...),
                  Brand: str = Form(...), Body: str = Form(...), Engine_Type: str = Form(...)):
    brand_map = {brand: 1 if Brand == brand else 0 for brand in brands}
    body_map = {body: 1 if Body == body else 0 for body in bodies}
    engine_type_map = {engine_type: 1 if Engine_Type == engine_type else 0 for engine_type in engine_types}

    inputs = [mileage, EngineV] + list(brand_map.values()) + list(body_map.values()) + list(engine_type_map.values())

    # Reshape the inputs to a 2D array as the scaler expects 2D data
    inputs = [inputs]

    # Transform the inputs using the scaler
    scaled_inputs = scaler.transform(inputs)

    prediction = model.predict(scaled_inputs)[0]

    print("The cost of old car is ", np.exp(prediction))

    return {
        "prediction": np.exp(prediction),
        "unit": "USD"
    }


if __name__ == "__main__":
    uvicorn.run(app, port=9003, access_log=True)
