"""
FastAPI сервис для предсказания оттока клиентов банка.
Принимает JSON с данными клиента, возвращает вероятность оттока и бинарный прогноз.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
import json
import os

# ============================================================
# Инициализация приложения
# ============================================================
app = FastAPI(
    title="🏦 Bank Churn Prediction API",
    description="API-сервис для предсказания вероятности оттока клиента банка",
    version="1.0.0"
)

# ============================================================
# Загрузка модели и артефактов
# ============================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")

model = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
le_gender = joblib.load(os.path.join(MODELS_DIR, "label_encoder_gender.pkl"))

with open(os.path.join(MODELS_DIR, "metadata.json"), "r", encoding="utf-8") as f:
    metadata = json.load(f)

feature_names = metadata["feature_names"]
needs_scaling = metadata.get("needs_scaling", False)


# ============================================================
# Pydantic модели
# ============================================================
class ClientData(BaseModel):
    """Входные данные клиента для предсказания."""
    кредитный_рейтинг: float = Field(..., ge=300, le=900, description="Кредитный рейтинг (300-900)")
    город: str = Field(..., description="Город: Алматы, Астана или Атырау")
    пол: str = Field(..., description="Пол: Male или Female")
    возраст: float = Field(..., ge=18, le=100, description="Возраст (18-100)")
    стаж_в_банке: float = Field(..., ge=0, le=20, description="Стаж в банке (лет)")
    баланс_депозита: float = Field(0.0, ge=0, description="Баланс депозита")
    число_продуктов: float = Field(..., ge=1, le=4, description="Число банковских продуктов")
    есть_кредитка: float = Field(..., ge=0, le=1, description="Есть кредитка (0 или 1)")
    активный_клиент: float = Field(..., ge=0, le=1, description="Активный клиент (0 или 1)")
    оценочная_зарплата: float = Field(..., ge=0, description="Оценочная зарплата")

    class Config:
        json_schema_extra = {
            "example": {
                "кредитный_рейтинг": 650,
                "город": "Алматы",
                "пол": "Male",
                "возраст": 35,
                "стаж_в_банке": 5,
                "баланс_депозита": 100000.0,
                "число_продуктов": 2,
                "есть_кредитка": 1,
                "активный_клиент": 1,
                "оценочная_зарплата": 120000.0
            }
        }


class PredictionResponse(BaseModel):
    """Ответ сервиса с предсказанием."""
    probability: float = Field(..., description="Вероятность оттока (0-1)")
    prediction: int = Field(..., description="Бинарный прогноз (0 — остался, 1 — ушёл)")
    risk_level: str = Field(..., description="Уровень риска: Низкий / Средний / Высокий")


# ============================================================
# Функция предобработки
# ============================================================
def preprocess_client(data: ClientData) -> np.ndarray:
    """Преобразует данные клиента в формат для модели."""
    gender_encoded = le_gender.transform([data.пол])[0]

    cities = metadata["cities"]  # ['Алматы', 'Астана', 'Атырау']
    city_encoded = [1 if c == data.город else 0 for c in cities]

    features = [
        data.кредитный_рейтинг,
        gender_encoded,
        data.возраст,
        data.стаж_в_банке,
        data.баланс_депозита,
        data.число_продуктов,
        data.есть_кредитка,
        data.активный_клиент,
        data.оценочная_зарплата,
    ] + city_encoded

    features_array = np.array(features).reshape(1, -1)

    # Масштабирование если нужно (для LR)
    if needs_scaling:
        features_array = scaler.transform(features_array)

    return features_array


# ============================================================
# Эндпоинты
# ============================================================
@app.get("/", tags=["Info"])
def root():
    """Главная страница."""
    return {
        "service": "Bank Churn Prediction API",
        "version": "1.0.0",
        "model": metadata["best_model_name"],
        "metrics": metadata["metrics"],
        "endpoints": {
            "POST /predict": "Предсказание оттока",
            "GET /health": "Проверка здоровья",
            "GET /docs": "Документация API (Swagger)"
        }
    }


@app.get("/health", tags=["Health"])
def health_check():
    """Проверка работоспособности сервиса."""
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(client: ClientData):
    """
    Предсказание вероятности оттока клиента.

    Принимает JSON с данными клиента, возвращает:
    - probability: вероятность оттока (float, 0-1)
    - prediction: бинарный прогноз (0 или 1)
    - risk_level: уровень риска
    """
    try:
        if client.город not in metadata["cities"]:
            raise HTTPException(
                status_code=400,
                detail=f"Город '{client.город}' не поддерживается. Допустимые: {metadata['cities']}"
            )

        features = preprocess_client(client)

        probability = float(model.predict_proba(features)[0][1])
        prediction = int(probability >= 0.5)

        if probability < 0.3:
            risk_level = "🟢 Низкий"
        elif probability < 0.7:
            risk_level = "🟡 Средний"
        else:
            risk_level = "🔴 Высокий"

        return PredictionResponse(
            probability=round(probability, 4),
            prediction=prediction,
            risk_level=risk_level
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка предсказания: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
