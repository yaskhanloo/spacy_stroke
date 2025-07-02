import pandas as pd
import csv
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
        # Medical terms based on German report terminology from CSV
        medical_terms = [
            'anästhesie', 'sedierung', 'rtpa', 'heparin', 'thrombektomie',
            'tici', 'trevo', 'sofia', 'solitaire', 'perforation', 'blutung',
            'aspirationskatheter', 'mikrokatheter', 'stentretriever', 'manöver',
            'thrombozytenaggregationshemmung', 'spasmolyse', 'gefässverschlüsse',
            'stenosen', 'reperfusionsergebnis', 'darstellung', 'komplikationen'
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
        
        # Category-specific patterns based on CSV variables
        category_patterns = {
            'anaesthesia': [r'intubationsnarkose', r'anästhesie', r'sedierung', r'narkose', r'lokalanästhesie', r'vollnarkose'],
            'aspiration_catheter_used': [r'aspirationskatheter', r'sofia', r'penumbra', r'catch'],
            'complications': [r'komplikationen', r'perforation', r'blutung', r'hämatom', r'nachblutung'],
            'end_time_intervention': [r'schleuse.*entfernt', r'ende.*intervention', r'\d{1,2}:\d{2}.*ende'],
            'extracranial_pta_stenting': [r'extrakranielle pta', r'extrakranielle.*stenting'],
            'guide_catheter_used': [r'guide.?katheter', r'führungskatheter'],
            'intracranial_pta_stenting': [r'intrakranielle pta', r'intrakranielle.*stenting'],
            'microcatheter_used': [r'mikrokatheter', r'microcatheter'],
            'number_recanalization_attempts': [r'anzahl.*manöver', r'\d+.*versuch', r'\d+.*manöver'],
            'periprocedural_antiplatelet': [r'thrombozytenaggregationshemmung', r'aspirin', r'clopidogrel'],
            'periprocedural_ia_thrombolysis': [r'ia.?thrombolyse', r'intra.?arterial.*thrombolyse', r'rtpa', r'alteplase'],
            'periprocedural_spasmolytic': [r'spasmolyse', r'vasospasmen', r'nimodipin'],
            'site_of_occlusion': [r'gefässverschlüsse', r'verschluss', r'okklusion', r'mca', r'ica'],
            'start_time_intervention': [r'schleuse.*aufgegeben', r'beginn.*intervention', r'\d{1,2}:\d{2}.*start'],
            'stenoses_cervical_arteries': [r'stenosen.*zervikalen', r'halsarterien.*stenose'],
            'stent_retriever_used': [r'stent.?retriever', r'trevo', r'solitaire', r'embotrap'],
            'tici_score': [r'tici\s*[0-3][abc]?', r'reperfusionsergebnis', r'rekanalisierung'],
            'technique_first_maneuver': [r'technik.*manöver', r'erste.*technik'],
            'visualisation_vessels': [r'darstellung.*gefässe', r'angiographie', r'dsa']
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
        # Categories based on intervention_report_variables_20250612.csv
        self.categories = [
            'anaesthesia', 'aspiration_catheter_used', 'complications',
            'end_time_intervention', 'extracranial_pta_stenting', 'guide_catheter_used',
            'intracranial_pta_stenting', 'microcatheter_used', 'number_recanalization_attempts',
            'periprocedural_antiplatelet', 'periprocedural_ia_thrombolysis', 'periprocedural_spasmolytic',
            'site_of_occlusion', 'start_time_intervention', 'stenoses_cervical_arteries',
            'stent_retriever_used', 'tici_score', 'technique_first_maneuver',
            'visualisation_vessels'
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

def load_csv_variables(csv_path: str = "intervention_report_variables_20250612.csv") -> Dict[str, str]:
    """Load variable definitions from CSV file."""
    variables = {}
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row.get('Variable (EN)') and row.get('German Report Term'):
                    variables[row['Variable (EN)']] = row['German Report Term']
    except FileNotFoundError:
        print(f"CSV file not found: {csv_path}")
        return {}
    
    return variables

def extract_csv_variables(text: str, report_id: str = None) -> Dict[str, Optional[str]]:
    """Extract variables from text based on intervention_report_variables_20250612.csv."""
    
    # extractor = StrokeMLExtractor()  # Not needed for pattern-based extraction
    results = {'report_id': report_id or 'unknown'}
    
    # Manual extraction patterns for each CSV variable
    extraction_patterns = {
        'anaesthesia': [
            (r'intubationsnarkose', 'Intubationsnarkose'),
            (r'allgemein.*anästhesie', 'Allgemeinanästhesie'),
            (r'lokal.*anästhesie', 'Lokalanästhesie'),
            (r'sedierung', 'Sedierung'),
            (r'vollnarkose', 'Vollnarkose')
        ],
        'aspiration_catheter_used': [
            (r'sofia', 'SOFIA'),
            (r'penumbra', 'Penumbra'),
            (r'catch.*mini', 'Catch Mini'),
            (r'aspirationskatheter', 'Aspirationskatheter')
        ],
        'guide_catheter_used': [
            (r'guide.*katheter', 'Guide-Katheter'),
            (r'führungskatheter', 'Führungskatheter')
        ],
        'microcatheter_used': [
            (r'mikrokatheter', 'Mikrokatheter'),
            (r'microcatheter', 'Microcatheter')
        ],
        'stent_retriever_used': [
            (r'trevo', 'Trevo'),
            (r'solitaire', 'Solitaire'),
            (r'embotrap', 'EmboTrap'),
            (r'stent.*retriever', 'Stent-Retriever')
        ],
        'tici_score': [
            (r'tici\s*([0-3][abc]?)', r'TICI \1'),
            (r'reperfusionsergebnis.*tici\s*([0-3][abc]?)', r'TICI \1')
        ],
        'start_time_intervention': [
            (r'schleuse.*aufgegeben.*?(\d{1,2}:\d{2})', r'\1'),
            (r'beginn.*intervention.*?(\d{1,2}:\d{2})', r'\1'),
            (r'start.*?(\d{1,2}:\d{2})', r'\1'),
            (r'interventionsbeginn.*?(\d{1,2}:\d{2})', r'\1')
        ],
        'end_time_intervention': [
            (r'schleuse.*entfernt.*?(\d{1,2}:\d{2})', r'\1'),
            (r'ende.*intervention.*?(\d{1,2}:\d{2})', r'\1'),
            (r'abschluss.*?(\d{1,2}:\d{2})', r'\1')
        ],
        'complications': [
            (r'perforation', 'Perforation'),
            (r'blutung', 'Blutung'),
            (r'hämatom', 'Hämatom'),
            (r'nachblutung', 'Nachblutung'),
            (r'komplikationen?', 'Komplikation')
        ],
        'periprocedural_ia_thrombolysis': [
            (r'ia.*thrombolyse', 'IA-Thrombolyse'),
            (r'rtpa', 'rtPA'),
            (r'alteplase', 'Alteplase'),
            (r'tenecteplase', 'Tenecteplase')
        ],
        'periprocedural_antiplatelet': [
            (r'thrombozytenaggregationshemmung', 'Thrombozytenaggregationshemmung'),
            (r'aspirin', 'Aspirin'),
            (r'clopidogrel', 'Clopidogrel')
        ],
        'number_recanalization_attempts': [
            (r'(\d+).*manöver', r'\1'),
            (r'anzahl.*manöver.*?(\d+)', r'\1')
        ],
        'site_of_occlusion': [
            (r'mca', 'MCA'),
            (r'ica', 'ICA'),
            (r'gefäßverschluss', 'Gefäßverschluss'),
            (r'verschluss', 'Verschluss')
        ]
    }
    
    text_lower = text.lower()
    
    for variable, patterns in extraction_patterns.items():
        results[variable] = None
        
        for pattern, replacement in patterns:
            match = re.search(pattern, text_lower, re.IGNORECASE)
            if match:
                if '\\1' in replacement:
                    results[variable] = re.sub(pattern, replacement, match.group(0), flags=re.IGNORECASE)
                else:
                    results[variable] = replacement
                break
    
    return results

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
    
    # Corresponding ground truth for training based on CSV variables
    training_ground_truth = [
        {
            'report_id': 'train_001',
            'anaesthesia': 'Allgemeinanästhesie',
            'stent_retriever_used': 'Trevo',
            'periprocedural_ia_thrombolysis': 'rtPA',
            'tici_score': 'TICI 3',
            'start_time_intervention': '08:30',
            'complications': None
        },
        {
            'report_id': 'train_002',
            'anaesthesia': 'Sedierung',
            'aspiration_catheter_used': 'SOFIA',
            'tici_score': 'TICI 2b',
            'start_time_intervention': '09:45',
            'complications': 'Nachblutung'
        },
        {
            'report_id': 'train_003',
            'anaesthesia': 'Lokalanästhesie',
            'stent_retriever_used': 'Solitaire',
            'periprocedural_ia_thrombolysis': 'Urokinase',
            'tici_score': 'TICI 1',
            'start_time_intervention': '14:15',
            'complications': 'Perforation'
        },
        {
            'report_id': 'train_004',
            'anaesthesia': 'Vollnarkose',
            'aspiration_catheter_used': 'Penumbra',
            'periprocedural_ia_thrombolysis': 'Tenecteplase',
            'tici_score': 'TICI 2c',
            'start_time_intervention': '11:20',
            'complications': None
        },
        {
            'report_id': 'train_005',
            'anaesthesia': 'Sedierung',
            'aspiration_catheter_used': 'Catch Mini',
            'periprocedural_ia_thrombolysis': 'Alteplase',
            'tici_score': 'TICI 3',
            'start_time_intervention': '13:45',
            'complications': 'Blutung'
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

def test_csv_variable_extraction():
    """Test the CSV variable extraction function."""
    
    print("📊 Testing CSV Variable Extraction")
    print("=" * 50)
    
    # Test with sample report text
    test_text = """Patient mit akutem ischämischem Schlaganfall. Allgemeinanästhesie eingeleitet.
                   Interventionsbeginn: 08:30 Uhr. rtPA bereits präklinisch verabreicht.
                   Mechanische Thrombektomie mit Trevo Stentretriever durchgeführt.
                   TICI 3 Rekanalisierung erreicht. Keine intraoperativen Komplikationen."""
    
    results = extract_csv_variables(test_text, "test_001")
    
    print(f"\n🧪 Extraction Results:")
    for variable, value in results.items():
        if value is not None:
            print(f"  ✅ {variable}: {value}")
        else:
            print(f"  ❌ {variable}: Not found")
    
    return results

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
    ml_extractor.save_models()
    
    # Test CSV variable extraction
    test_results = test_csv_variable_extraction()
    
    return ml_extractor, training_results, test_results

if __name__ == "__main__":
    # Test CSV variable extraction
    test_results = test_csv_variable_extraction()
    
    print("\n🎉 CSV variable extraction ready!")
    print("🔍 Use extract_csv_variables() function to extract variables from reports")
    
    # Example usage with different report
    print("\n📋 Example with different report:")
    sample_report = """Sedierung für Eingriff. Start 14:30 Uhr. 
                       SOFIA Aspiration verwendet. Perforation aufgetreten. 
                       TICI 2b erreicht."""
    
    example_results = extract_csv_variables(sample_report, "example_001")
    print("Results:")
    for var, val in example_results.items():
        if val:
            print(f"  ✅ {var}: {val}")