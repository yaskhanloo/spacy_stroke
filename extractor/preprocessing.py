import re
import spacy
from spacy.tokens import Doc
from typing import List, Dict

class TextPreprocessor:
    """Handles text cleaning and preprocessing for stroke reports."""
    
    def __init__(self):
        # Load German spaCy model
        try:
            self.nlp = spacy.load("de_core_news_sm")
        except OSError:
            print("German spaCy model not found. Please install with:")
            print("python -m spacy download de_core_news_sm")
            raise
    
    def clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common headers/footers patterns
        text = re.sub(r'(Befund|Diagnose|Impression):\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'(Dr\.|Prof\.|Oberarzt|Chefarzt)\s+\w+', '', text)
        
        # Strip and lowercase
        return text.strip().lower()
    
    def tokenize_and_tag(self, text: str) -> Doc:
        """Process text with spaCy for POS and NER."""
        return self.nlp(text)
        return self.nlp(text)
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract individual sentences."""
        doc = self.nlp(text)
        return [sent.text.strip() for sent in doc.sents]