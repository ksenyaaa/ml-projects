from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(
    title="Sentiment Analysis API",
    description="API для определения тональности текста на русском языке",
    version="1.0.0"
)

class TextRequest(BaseModel):
    text: str

print("Загрузка модели")
sentiment_pipeline = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment")
print("Модель успешно загружена!")

@app.post("/predict")
def predict_sentiment(request: TextRequest):
    input_text = request.text

    result = sentiment_pipeline(input_text)[0]

    return {
        "text": input_text,
        "label": result["label"],
        "confidence": round(result["score"], 4)
    }

@app.get("/")
def root():
    return {"message": "ML API is running! Use /predict to analyze text."}