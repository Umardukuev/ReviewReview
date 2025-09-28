from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Скачиваем необходимые данные NLTK при запуске
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

app = FastAPI(
    title="Movie Review Sentiment Analysis",
    description="API for analyzing sentiment of movie reviews",
    version="1.0.0"
)

# Настройка шаблонов
templates = Jinja2Templates(directory="templates")


class ReviewRequest(BaseModel):
    text: str


# Загрузка модели и векторизатора
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    print("Warning: Model files not found!")


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def predict_sentiment(request: ReviewRequest):
    if not model_loaded:
        return {"error": "Model not loaded", "sentiment": "unknown", "confidence": 0}

    processed_text = preprocess_text(request.text)
    text_vector = vectorizer.transform([processed_text])

    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]

    sentiment = "positive" if prediction == 1 else "negative"
    confidence = probability[1] if prediction == 1 else probability[0]

    return {
        'text': request.text,
        'sentiment': sentiment,
        'confidence': round(confidence * 100, 2),
        'probabilities': {
            'positive': round(probability[1] * 100, 2),
            'negative': round(probability[0] * 100, 2)
        }
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model_loaded
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)