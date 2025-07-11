import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import matplotlib.pyplot as plt

app = FastAPI()

# Загрузка модели
try:
    model = joblib.load("models/best.pkl")
except Exception as e:
    raise RuntimeError(f"Ошибка загрузки модели или скейлера: {e}")

class InputData(BaseModel):
    Gender: str
    Alive: str
    Marital_Status: str
    Height: float
    Weight: float
    Origin: str
    Universe: str
    Identity: str

@app.post("/predict")
def predict(data: InputData):
    try:
        input_df = pd.DataFrame([data.dict()])
        
        # Предобработка данных аналогично тому, как это было сделано в обучении модели
        # Здесь нужно использовать тот же preprocessor, который был использован при обучении
        
        prediction = model.predict(input_df)  # Предполагается, что model - это Pipeline
        
        return {"predicted_value": int(prediction[0])}  # Возвращаем предсказанное значение как целое число
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Запуск приложения (если необходимо)
# uvicorn main:app --reload
