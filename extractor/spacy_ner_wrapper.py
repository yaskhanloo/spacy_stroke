import spacy
from typing import List, Dict, Tuple

class SpacyNERExtractor:
    """Wrapper for spaCy NER to enhance keyword extraction."""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("de_core_news_sm")
        except OSError:
            print("German spaCy model not found. Please install with:")
            print("python -m spacy download de_core_news_sm")
            raise
    
    def extract_entities(self, text: str) -> List[Dict]:
        """Extract named entities using spaCy."""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char,
                'description': spacy.explain(ent.label_)
            })
        
        return entities
    
    def extract_noun_phrases(self, text: str) -> List[str]:
        """Extract noun phrases that might be medical terms."""
        doc = self.nlp(text)
        noun_phrases = []
        
        for chunk in doc.noun_chunks:
            # Filter for potentially medical terms
            if len(chunk.text) > 3 and not chunk.root.is_stop:
                noun_phrases.append(chunk.text)
        
        return noun_phrases
    
    def get_pos_tags(self, text: str) -> List[Tuple[str, str]]:
        """Get POS tags for all tokens."""
        doc = self.nlp(text)
        return [(token.text, token.pos_) for token in doc]