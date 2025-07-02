import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import joblib
from pathlib import Path
import json
import re
import spacy
from dataclasses import dataclass

@dataclass
class MLExtractionResult:
    """Container for ML extraction results."""
    category: str
    value: Optional[str]
    confidence: float
    features_used: List[str]
    model_name: str

class FeatureExtractor:
    """Extract features from text for ML models."""
    
    def __init__(self):
        try:
            self.nlp = spacy.load("de_core_news_sm")
        except OSError:
            print("Warning: German spaCy model not found. Install with: python -m spacy download de_core_news_sm")
            self.nlp = None
    
    def extract_text_features(self, text: str) -> Dict[str, Any]:
        """Extract various text features for ML training."""
        features = {}
        
        # Basic text statistics
        features['char_count'] = len(text)
        features['word_count'] = len(text.split())
        features['sentence_count'] = len(text.split('.'))
        features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
        
        # Medical term frequency
        medical_terms = [
            'anästhesie', 'sedierung', 'rtpa', 'heparin', 'thrombektomie',
            'tici', 'trevo', 'sofia', 'solitaire', 'perforation', 'blutung'
        ]
        
        for term in medical_terms:
            features[f'has_{term}'] = 1 if term in text.lower() else 0
            features[f'count_{term}'] = text.lower().count(term)
        
        # Time patterns
        time_pattern = r'\b([0-2]?[0-9]:[0-5][0-9])\b'
        time_matches = re.findall(time_pattern, text)
        features['time_count'] = len(time_matches)
        features['has_time'] = 1 if time_matches else 0
        
        # Numeric patterns
        numeric_pattern = r'\b\d+\b'
        numeric_matches = re.findall(numeric_pattern, text)
        features['numeric_count'] = len(numeric_matches)
        
        # Sentence structure features (if spaCy is available)
        if self.nlp:
            doc = self.nlp(text)
            features['noun_count'] = sum(1 for token in doc if token.pos_ == 'NOUN')
            features['verb_count'] = sum(1 for token in doc if token.pos_ == 'VERB')
            features['adj_count'] = sum(1 for token in doc if token.pos_ == 'ADJ')
            features['entity_count'] = len(doc.ents)
        
        return features
    
    def extract_context_features(self, text: str, target_category: str) -> Dict[str, Any]:
        """Extract context-specific features for a target category."""
        features = {}
        
        # Category-specific patterns
        category_patterns = {
            'anesthesia': [r'anästhesie', r'sedierung', r'narkose'],
            'medication': [r'rtpa', r'heparin', r'urokinase', r'alteplase'],
            'device': [r'trevo', r'sofia', r'solitaire', r'penumbra'],
            'tici_score': [r'tici\s*[0-3][abc]?'],
            'complications': [r'perforation', r'blutung', r'hämatom', r'komplikation']
        }
        
        patterns = category_patterns.get(target_category, [])
        
        for i, pattern in enumerate(patterns):
            matches = re.findall(pattern, text, re.IGNORECASE)
            features[f'pattern_{i}_count'] = len(matches)
            features[f'pattern_{i}_present'] = 1 if matches else 0
        
        # Context window features
        for pattern in patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                start = max(0, match.start() - 50)
                end = min(len(text), match.end() + 50)
                context = text[start:end]
                
                # Features from context
                features['context_word_count'] = len(context.split())
                features['context_has_time'] = 1 if re.search(r'\d{1,2}:\d{2}', context) else 0
                features['context_has_numeric'] = 1 if re.search(r'\d+', context) else 0
        
        return features

class StrokeMLExtractor:
    """Machine Learning-based extractor for stroke reports."""
    
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.feature_extractor = FeatureExtractor()
        self.categories = [
            'anesthesia', 'medication', 'device', 'treatment_method',
            'tici_score', 'complications'
        ]
        
    def prepare_training_data(self, reports_df: pd.DataFrame, 
                            ground_truth_df: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Prepare training data for each category.
        
        Args:
            reports_df: DataFrame with report texts
            ground_truth_df: DataFrame with ground truth labels
            
        Returns:
            Dictionary with training data for each category
        """
        training_data = {}
        
        # Merge reports and ground truth
        merged = reports_df.merge(ground_truth_df, on='report_id', how='inner')
        
        for category in self.categories:
            if category not in ground_truth_df.columns:
                continue
            
            X_texts = []
            X_features = []
            y = []
            
            for _, row in merged.iterrows():
                text = row['text']
                label = row[category]
                
                # Extract text and context features
                text_features = self.feature_extractor.extract_text_features(text)
                context_features = self.feature_extractor.extract_context_features(text, category)
                
                # Combine features
                combined_features = {**text_features, **context_features}
                
                X_texts.append(text)
                X_features.append(list(combined_features.values()))
                
                # Binary classification: has value or not
                y.append(1 if pd.notna(label) else 0)
            
            if X_texts:
                training_data[category] = {
                    'texts': X_texts,
                    'features': np.array(X_features),
                    'labels': np.array(y),
                    'feature_names': list(combined_features.keys())
                }
        
        return training_data
    
    def train_category_model(self, category: str, training_data: Dict) -> Dict[str, Any]:
        """Train ML model for a specific category."""
        
        if category not in training_data:
            return {'error': f'No training data for category: {category}'}
        
        data = training_data[category]
        X_texts = data['texts']
        X_features = data['features']
        y = data['labels']
        
        # Create text vectorizer
        vectorizer = TfidfVectorizer(
            max_features=1000,
            ngram_range=(1, 2),
            stop_words=None,  # Keep German stop words for medical context
            lowercase=True
        )
        
        X_text_vectors = vectorizer.fit_transform(X_texts)
        
        # Combine text vectors with hand-crafted features
        X_combined = np.hstack([X_text_vectors.toarray(), X_features])
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Try different models
        models = {
            'logistic_regression': LogisticRegression(random_state=42),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(probability=True, random_state=42)
        }
        
        best_model = None
        best_score = 0
        model_results = {}
        
        for model_name, model in models.items():
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Predict and evaluate
                y_pred = model.predict(X_test)
                score = model.score(X_test, y_test)
                
                model_results[model_name] = {
                    'accuracy': score,
                    'classification_report': classification_report(y_test, y_pred, output_dict=True)
                }
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    
            except Exception as e:
                model_results[model_name] = {'error': str(e)}
        
        # Store best model and vectorizer
        self.models[category] = best_model
        self.vectorizers[category] = vectorizer
        
        return {
            'category': category,
            'best_model': type(best_model).__name__ if best_model else None,
            'best_accuracy': best_score,
            'model_results': model_results,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
    
    def train_all_models(self, reports_df: pd.DataFrame, 
                        ground_truth_df: pd.DataFrame) -> Dict[str, Any]:
        """Train ML models for all categories."""
        
        print("🤖 Training ML models for stroke extraction...")
        
        # Prepare training data
        training_data = self.prepare_training_data(reports_df, ground_truth_df)
        
        results = {}
        
        for category in self.categories:
            print(f"  Training {category} model...")
            result = self.train_category_model(category, training_data)
            results[category] = result
            
            if 'error' in result:
                print(f"    ❌ Error: {result['error']}")
            else:
                print(f"    ✅ Best accuracy: {result['best_accuracy']:.3f}")
        
        return {
            'training_results': results,
            'total_categories': len(self.categories),
            'successful_models': len([r for r in results.values() if 'error' not in r])
        }
    
    def extract_with_ml(self, text: str, report_id: str = None) -> Dict[str, Any]:
        """Extract information using trained ML models."""
        
        if not self.models:
            return {'error': 'No models trained. Call train_all_models() first.'}
        
        results = {
            'report_id': report_id or 'unknown',
            'text_length': len(text)
        }
        
        # Extract features for this text
        text_features = self.feature_extractor.extract_text_features(text)
        
        for category in self.categories:
            if category not in self.models:
                results[category] = None
                continue
            
            model = self.models[category]
            vectorizer = self.vectorizers[category]
            
            try:
                # Extract context features for this category
                context_features = self.feature_extractor.extract_context_features(text, category)
                combined_features = {**text_features, **context_features}
                
                # Vectorize text
                text_vector = vectorizer.transform([text])
                
                # Combine with hand-crafted features
                feature_array = np.array(list(combined_features.values())).reshape(1, -1)
                X_combined = np.hstack([text_vector.toarray(), feature_array])
                
                # Predict
                prediction = model.predict(X_combined)[0]
                confidence = model.predict_proba(X_combined)[0].max()
                
                if prediction == 1:  # Has value
                    # For now, use rule-based extraction to get actual value
                    # In future, could train separate models for value extraction
                    results[category] = f"detected_with_confidence_{confidence:.3f}"
                else:
                    results[category] = None
                    
            except Exception as e:
                results[category] = f"error: {str(e)}"
        
        return results
    
    def save_models(self, model_dir: str = "models/"):
        """Save trained models to disk."""
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        saved_models = []
        
        for category in self.categories:
            if category in self.models and category in self.vectorizers:
                # Save model
                model_path = model_dir / f"{category}_model.pkl"
                joblib.dump(self.models[category], model_path)
                
                # Save vectorizer
                vectorizer_path = model_dir / f"{category}_vectorizer.pkl"
                joblib.dump(self.vectorizers[category], vectorizer_path)
                
                saved_models.append(category)
        
        # Save metadata
        metadata = {
            'categories': saved_models,
            'model_type': 'stroke_ml_extractor',
            'version': '1.0',
            'feature_extractor': 'custom_medical_features'
        }
        
        metadata_path = model_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Saved {len(saved_models)} models to {model_dir}")
        return saved_models
    
    def load_models(self, model_dir: str = "models/"):
        """Load trained models from disk."""
        model_dir = Path(model_dir)
        
        if not model_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Load metadata
        metadata_path = model_dir / "model_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            categories = metadata.get('categories', [])
        else:
            categories = self.categories
        
        loaded_models = []
        
        for category in categories:
            model_path = model_dir / f"{category}_model.pkl"
            vectorizer_path = model_dir / f"{category}_vectorizer.pkl"
            
            if model_path.exists() and vectorizer_path.exists():
                self.models[category] = joblib.load(model_path)
                self.vectorizers[category] = joblib.load(vectorizer_path)
                loaded_models.append(category)
        
        print(f"✅ Loaded {len(loaded_models)} models from {model_dir}")
        return loaded_models

def create_sample_training_data():
    """Create sample training data for ML model development."""
    
    # Sample training reports (larger set for ML)
    training_reports = [
        {
            'report_id': 'train_001',
            'text': '''Patient mit akutem ischämischem Schlaganfall. Allgemeinanästhesie eingeleitet.
                      Interventionsbeginn: 08:30 Uhr. rtPA bereits präklinisch verabreicht.
                      Mechanische Thrombektomie mit Trevo Stentretriever durchgeführt.
                      TICI 3 Rekanalisierung erreicht. Keine intraoperativen Komplikationen.'''
        },
        {
            'report_id': 'train_002',
            'text': '''67-jähriger Patient. Sedierung für die Intervention verwendet.
                      Prozedurstart: 09:45 Uhr. SOFIA Aspiration System eingesetzt.
                      Heparin zur Antikoagulation. Leichte Nachblutung beobachtet.
                      Finales TICI 2b Ergebnis erreicht.'''
        },
        {
            'report_id': 'train_003',
            'text': '''Lokalanästhesie für den minimal-invasiven Eingriff.
                      Beginn der Intervention: 14:15 Uhr. Solitaire FR Revascularization Device.
                      Urokinase als Thrombolytikum verwendet. Arterielle Perforation aufgetreten.
                      TICI 1 Rekanalisierung erreicht.'''
        },
        {
            'report_id': 'train_004',
            'text': '''Patient in Vollnarkose. Penumbra Aspiration System verwendet.
                      Start: 11:20 Uhr. Direkte Aspiration Thrombektomie.
                      Tenecteplase verabreicht. Erfolgreiche Rekanalisierung.
                      TICI 2c erreicht. Kein postoperatives Hämatom.'''
        },
        {
            'report_id': 'train_005',
            'text': '''Prozedur unter Sedierung. Catch Mini Device eingesetzt.
                      Interventionsbeginn: 13:45 Uhr. Alteplase Thrombolyse durchgeführt.
                      EmboTrap als Backup-Device. Minimale Blutung.
                      Exzellente TICI 3 Rekanalisierung.'''
        }
    ]
    
    # Corresponding ground truth for training
    training_ground_truth = [
        {
            'report_id': 'train_001',
            'anesthesia': 'allgemeinanästhesie',
            'medication': 'rtpa',
            'device': 'trevo',
            'treatment_method': 'mechanische thrombektomie',
            'tici_score': 'tici 3',
            'times': '08:30',
            'complications': None
        },
        {
            'report_id': 'train_002',
            'anesthesia': 'sedierung',
            'medication': 'heparin',
            'device': 'sofia',
            'treatment_method': 'aspiration',
            'tici_score': 'tici 2b',
            'times': '09:45',
            'complications': 'blutung'
        },
        {
            'report_id': 'train_003',
            'anesthesia': 'lokalanästhesie',
            'medication': 'urokinase',
            'device': 'solitaire',
            'treatment_method': None,
            'tici_score': 'tici 1',
            'times': '14:15',
            'complications': 'perforation'
        },
        {
            'report_id': 'train_004',
            'anesthesia': 'vollnarkose',
            'medication': 'tenecteplase',
            'device': 'penumbra',
            'treatment_method': 'aspiration thrombektomie',
            'tici_score': 'tici 2c',
            'times': '11:20',
            'complications': None
        },
        {
            'report_id': 'train_005',
            'anesthesia': 'sedierung',
            'medication': 'alteplase',
            'device': 'catch mini',
            'treatment_method': 'thrombolyse',
            'tici_score': 'tici 3',
            'times': '13:45',
            'complications': 'blutung'
        }
    ]
    
    # Create DataFrames
    reports_df = pd.DataFrame(training_reports)
    ground_truth_df = pd.DataFrame(training_ground_truth)
    
    # Save training data
    Path("data/training").mkdir(parents=True, exist_ok=True)
    reports_df.to_csv("data/training/training_reports.csv", index=False)
    ground_truth_df.to_csv("data/training/training_ground_truth.csv", index=False)
    
    print("✅ Sample training data created:")
    print(f"  📄 Reports: data/training/training_reports.csv ({len(reports_df)} reports)")
    print(f"  📊 Ground Truth: data/training/training_ground_truth.csv")
    
    return reports_df, ground_truth_df

def train_ml_models_example():
    """Example of training ML models."""
    
    print("🤖 ML Model Training Example")
    print("=" * 50)
    
    # Create sample training data
    reports_df, ground_truth_df = create_sample_training_data()
    
    # Initialize ML extractor
    ml_extractor = StrokeMLExtractor()
    
    # Train models
    training_results = ml_extractor.train_all_models(reports_df, ground_truth_df)
    
    print(f"\n📊 Training Summary:")
    print(f"  Total categories: {training_results['total_categories']}")
    print(f"  Successful models: {training_results['successful_models']}")
    
    # Save models
    saved_models = ml_extractor.save_models()
    
    # Test extraction
    test_text = """Patient mit Schlaganfall. Allgemeinanästhesie verwendet.
                   Beginn: 10:30 Uhr. rtPA verabreicht. Trevo System eingesetzt.
                   TICI 3 erreicht. Keine Komplikationen."""
    
    results = ml_extractor.extract_with_ml(test_text, "test_001")
    
    print(f"\n🧪 Test Extraction Results:")
    for category, value in results.items():
        if category not in ['report_id', 'text_length']:
            print(f"  {category}: {value}")
    
    return ml_extractor, training_results

if __name__ == "__main__":
    # Run ML training example
    ml_extractor, results = train_ml_models_example()
    
    print("\n🎉 ML model training complete!")
    print("💾 Models saved to models/ directory")
    print("🔬 Ready for advanced extraction!")