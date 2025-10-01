import time

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pickle

from config import settings
from utils.text_processor import TextProcessor
from models.transformer_model import TransformerSentimentAnalyzer

app = FastAPI(
    title="Movie Review Sentiment Analysis"
)


class ReviewRequest(BaseModel):
    text: str
    model_type: Optional[str] = 'transformer'


text_processor = TextProcessor()
traditional_model = None
transformer_model = None

try:
    with open(settings.model_path, 'rb') as f:
        traditional_model = pickle.load(f)
    with open(settings.vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)
except FileNotFoundError:
    print("Traditional model file not found!")


try:
    transformer_model = TransformerSentimentAnalyzer()
except Exception as e:
    print(f'Transformer model loading failed: {e}')


@app.post("/predict")
async def predict_sentiment(request: ReviewRequest):
    start_time = time.time()

    if request.model_type == 'transformer' and transformer_model:
        result = transformer_model.predict(request.text)
        model_used = 'transformer'
    elif traditional_model:
        processed_text = text_processor.preprocess(request.text)
        text_vector = vectorizer.transform([processed_text])

        prediction = traditional_model.predict(text_vector)[0]
        probability = traditional_model.predict_proba(text_vector)[0]

        sentiment = "positive" if prediction == 1 else "negative"
        confidence = probability[1] if prediction == 1 else probability[0]

        result = {
            'text': request.text,
            'sentiment': sentiment,
            'confidence': round(confidence * 100, 2),
            'probabilities': {
                'positive': round(probability[1] * 100, 2),
                'negative': round(probability[0] * 100, 2)
            }
        }
        model_used = 'traditional'
    else:
        return {'error' : 'no model available'}

    processing_time = round(time.time() - start_time, 4)

    return {
        **result,
        'model_used' : model_used,
        'processing_time_seconds' : processing_time,
        'text_preview' : request.text[:100] + '...' if len(request.text) > 100 else request.text
    }



if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)