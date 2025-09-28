from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pickle
import re
from nltk.corpus import stopwords

app = FastAPI(title="Sentiment Analysis API")

# Настройка шаблонов
templates = Jinja2Templates(directory="templates")


class ReviewRequest(BaseModel):
    text: str


# Загрузка модели и векторизатора
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)


@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
def predict_sentiment(request: ReviewRequest):
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)