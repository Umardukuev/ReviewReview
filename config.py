import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_path: str = 'model.pkl'
    vectorizer_path: str = 'vectorizer.pkl'
    use_transformer: bool = False
    transformer_model: str = 'distilbert-base-uncased'

    class Config:
        env_file = '.env'


settings = Settings()