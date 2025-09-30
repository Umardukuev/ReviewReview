import re
import nltk
import spacy

class TextProcessor:
    def __init__(self, use_lemmatization=True):
        self.use_lemmatization = use_lemmatization
        if use_lemmatization:
            try:
                self.nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
            except OSError:
                raise RuntimeError("Please download: python -m spacy download en_core_web_sm")

    def preprocess(self, text):
        text = text.lower().strip()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s!?]', '', text)

        if self.use_lemmatization:
            doc = self.nlp(text)
            tokens = [token.lemma_ for token in doc if not token.is_stop and len(token.text) > 2]

        else:
            tokens = [word for word in text.split() if len(word) > 2]

        return ''.join(tokens)
