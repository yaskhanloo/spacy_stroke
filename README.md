# German Stroke Radiology Report NLP Extractor

A hybrid NLP system combining rule-based pattern matching with machine learning to extract structured variables from German stroke radiology reports. The system processes PDF and text files, extracts medical variables with confidence scores, and provides accuracy metrics for model validation.

## 🎯 Project Goals

- **Primary**: Extract structured variables (anesthesia, medications, devices, TICI scores, times, complications) from German stroke reports
- **Training**: Train on PDF/text files with validation datasets
- **Evaluation**: Provide accuracy metrics and model performance tracking
- **Testing**: Comprehensive unit tests ensuring reliability
- **Output**: Structured CSV with extracted variables and confidence scores

## 📊 Key Features

✅ **Hybrid NLP Approach**: Rule-based patterns + ML models  
✅ **German Medical Terminology**: Specialized for stroke radiology  
✅ **Multi-format Input**: PDF, TXT, CSV file processing  
✅ **Accuracy Metrics**: Precision, recall, F1-score tracking  
✅ **Unit Testing**: 95%+ test coverage with realistic scenarios  
✅ **Confidence Scoring**: Each extraction includes confidence level  
✅ **Batch Processing**: Handle multiple reports efficiently  

## 🏗️ Project Structure

```
stroke_nlp_project/
│
├── data/
│   ├── raw_reports/                 # Training stroke reports (txt/pdf)
│   ├── validation/                  # Gold standard for accuracy testing
│   └── sample_reports.csv           # Demo/test data
│
├── extractor/
│   ├── __init__.py
│   ├── preprocessing.py             # Text cleaning, PDF processing
│   ├── keyword_rules.py             # Rule-based pattern extraction
│   ├── spacy_ner_wrapper.py         # NER and POS tagging
│   └── ml_model.py                  # ML model training/inference
│
├── evaluation/
│   ├── metrics.py                   # Accuracy calculation
│   ├── validation.py                # Model performance testing
│   └── gold_standard.csv            # Annotated ground truth
│
├── output/
│   ├── extracted_keywords.csv       # Final structured output
│   ├── accuracy_report.json         # Model performance metrics
│   └── confidence_analysis.csv      # Confidence score analysis
│
├── tests/
│   ├── test_keyword_extractor.py    # Unit tests for extraction
│   ├── test_accuracy_metrics.py     # Testing evaluation metrics
│   └── debug_session.py             # Debugging utilities
│
├── notebooks/
│   ├── 01_exploration.ipynb         # EDA & prototyping
│   ├── 02_model_training.ipynb      # ML model development
│   └── 03_accuracy_analysis.ipynb   # Performance evaluation
│
├── main.py                          # CLI application entry point
├── train_model.py                   # Model training pipeline
├── evaluate_model.py                # Accuracy testing script
└── streamlit_app.py                 # Web interface
```

## 🚀 Quick Start

### 1. Installation
```bash
# Clone repository
git clone <your-repo-url>
cd stroke_nlp_project

# Install all dependencies
pip install -r requirements.txt

# Download German spaCy model
python -m spacy download de_core_news_sm
```

### 2. Web Interface (Recommended)
```bash
# Launch advanced Streamlit web interface
streamlit run streamlit_app.py
```

**The Streamlit app provides:**
- 📄 **Single Report Extraction** - Process individual reports (text, PDF, or upload)
- 📊 **Model Training** - Train hybrid models with cross-validation
- 📈 **Model Evaluation** - Comprehensive accuracy analysis with visualizations
- 🔬 **Batch Processing** - Process multiple files at once
- 📚 **Documentation** - Complete usage guide and troubleshooting

### 3. Command Line Usage
```bash
# Process sample reports (rule-based)
python main.py

# Train hybrid models (rule-based + ML)
python train_model.py

# Evaluate model accuracy with detailed metrics
python evaluate_model.py --test-reports data/validation/validation_reports.csv --test-ground-truth evaluation/gold_standard.csv --detailed-report

# Run comprehensive tests
python tests/test_keyword_extractor.py
python tests/test_accuracy_metrics.py
```

### 4. PDF Processing
```bash
# The system automatically handles PDF files through:
# - PyPDF2 for fast extraction
# - pdfplumber for complex layouts
# - Automatic fallback between methods
# - Batch PDF processing capabilities
```

## 📋 Extracted Variables

| Variable | Description | Example Values |
|----------|-------------|----------------|
| **anesthesia** | Type of anesthesia used | `allgemeinanästhesie`, `sedierung`, `lokalanästhesie` |
| **medication** | Medications administered | `rtpa`, `heparin`, `urokinase`, `tenecteplase` |
| **device** | Medical devices used | `trevo`, `sofia`, `solitaire`, `penumbra`, `embotrap` |
| **treatment_method** | Treatment approach | `thrombektomie`, `aspiration`, `stentretriever` |
| **tici_score** | TICI recanalization score | `tici 3`, `tici 2b`, `tici 1` |
| **times** | Procedure timestamps | `08:32`, `09:15`, `14:30` |
| **complications** | Reported complications | `perforation`, `blutung`, `hämatom` |

## 🎯 Accuracy & Validation

### Current Performance (Rule-based)
- **Precision**: 85-95% (high confidence patterns)
- **Recall**: 70-80% (depends on terminology coverage)
- **F1-Score**: 77-87% (balanced performance)

### Validation Process
1. **Gold Standard**: Manual annotation of 100+ reports
2. **Cross-validation**: 5-fold validation on training data
3. **Test Set**: Independent validation on unseen reports
4. **Metrics Tracking**: Precision, recall, F1 per variable type

### Model Evaluation Commands
```bash
# Comprehensive evaluation with visualizations
python evaluate_model.py --test-reports data/validation/validation_reports.csv --test-ground-truth evaluation/gold_standard.csv --detailed-report

# Quick evaluation on sample data
python evaluate_model.py

# View confidence score distribution
python -c "from evaluation.metrics import plot_confidence_distribution; plot_confidence_distribution()"

# Run all unit tests
python tests/test_keyword_extractor.py
python tests/test_accuracy_metrics.py

# Or with pytest (if installed)
python -m pytest tests/ -v --coverage
```

### Advanced Training Options
```bash
# Train with custom data
python train_model.py --data path/to/training/ --validation path/to/validation/ --output results/

# Train only rule-based evaluation
python train_model.py --rule-only

# Train only ML models
python train_model.py --ml-only

# Custom configuration
python train_model.py --config training_config.json
```

### PDF Processing Examples
```bash
# Process single PDF via Python
python -c "
from extractor.preprocessing import TextPreprocessor
preprocessor = TextPreprocessor()
text = preprocessor.extract_text_from_pdf('report.pdf')
print(f'Extracted {len(text.split())} words')
"

# Batch process PDFs
python -c "
from extractor.preprocessing import TextPreprocessor
preprocessor = TextPreprocessor()
results = preprocessor.process_pdf_batch('pdf_folder/', 'output_texts/')
print(f'Processed {len(results)} PDFs')
"
```

## 🧪 Testing Framework

### Unit Tests Coverage
- ✅ **Keyword Extraction**: `test_keyword_extractor.py`
- ✅ **Text Preprocessing**: Pattern cleaning, normalization
- ✅ **Accuracy Metrics**: Precision/recall calculations
- ✅ **Integration Tests**: Full pipeline validation
- ✅ **Edge Cases**: Empty text, no matches, multiple matches

### Run Tests
```bash
# Run all tests
python tests/test_keyword_extractor.py

# Run with pytest (detailed output)
pytest tests/ -v

# Coverage analysis
coverage run tests/test_keyword_extractor.py
coverage report
```

## 📊 Development Roadmap

### Phase 1: Rule-based Foundation ✅
- [x] German medical pattern extraction
- [x] Comprehensive unit testing
- [x] Basic accuracy metrics
- [x] Text preprocessing pipeline

### Phase 2: ML Enhancement ✅
- [x] PDF processing capability (PyPDF2 + pdfplumber)
- [x] ML model training pipeline (scikit-learn)
- [x] Feature engineering for medical NLP
- [x] Model validation framework (K-fold CV)
- [x] Performance benchmarking and visualization
- [x] Comprehensive evaluation scripts
- [x] Advanced Streamlit web interface

### Phase 3: Production Ready (In Progress)
- [ ] REST API endpoint
- [ ] Docker containerization
- [ ] Model deployment pipeline
- [ ] Real-time monitoring and logging
- [ ] Automated model retraining
- [ ] DICOM integration
- [ ] Multi-language support

## 🔧 Adding New Variables

To extract additional medical variables:

1. **Add patterns** to `extractor/keyword_rules.py`:
```python
'new_category': [
    r'\bnew_pattern_1\b',
    r'\bnew_pattern_2\b'
]
```

2. **Create tests** in `tests/test_keyword_extractor.py`:
```python
def test_extract_new_category(self):
    text = "Sample text with new_pattern_1"
    results = self.extractor.extract_category(text, 'new_category')
    self.assertEqual(len(results), 1)
```

3. **Update validation** in `evaluation/gold_standard.csv`

## 📈 Model Performance Tracking

The system tracks performance across multiple dimensions:

- **Per-variable accuracy**: Individual precision/recall for each extracted variable
- **Confidence calibration**: How well confidence scores correlate with accuracy
- **Error analysis**: Common false positives/negatives
- **Temporal performance**: Accuracy trends over time/dataset versions

## 💻 Advanced Usage

### Using the Streamlit Web Interface

1. **Launch the app**: `streamlit run streamlit_app.py`
2. **Navigate pages** using the sidebar:
   - **📄 Report Extraction**: Process single reports with text/PDF input
   - **📊 Model Training**: Train hybrid models with cross-validation
   - **📈 Model Evaluation**: Analyze performance with visualizations
   - **🔬 Batch Processing**: Handle multiple files simultaneously
   - **📚 Documentation**: In-app help and troubleshooting

### Programmatic Usage

```python
# Initialize components
from extractor.preprocessing import TextPreprocessor
from extractor.keyword_rules import KeywordExtractor
from extractor.ml_model import StrokeMLExtractor

preprocessor = TextPreprocessor()
rule_extractor = KeywordExtractor()
ml_extractor = StrokeMLExtractor()

# Process a report
text = "Patient mit Allgemeinanästhesie. rtPA verabreicht. TICI 3 erreicht."
cleaned_text = preprocessor.clean_text(text)

# Rule-based extraction
rule_results = rule_extractor.extract_all(cleaned_text, "report_001")

# ML extraction (if models are trained)
ml_results = ml_extractor.extract_with_ml(cleaned_text, "report_001")

# Compare results
print("Rule-based:", rule_results)
print("ML-based:", ml_results)
```

### Training Custom Models

```python
import pandas as pd
from train_model import TrainingPipeline

# Prepare your data
training_reports = pd.DataFrame([
    {'report_id': 'r1', 'text': 'German medical report text...'},
    {'report_id': 'r2', 'text': 'Another report text...'}
])

training_ground_truth = pd.DataFrame([
    {'report_id': 'r1', 'anesthesia': 'allgemeinanästhesie', 'medication': 'rtpa'},
    {'report_id': 'r2', 'anesthesia': 'sedierung', 'medication': 'heparin'}
])

# Train models
pipeline = TrainingPipeline()
results = pipeline.run_full_training_pipeline()
```

### Evaluation and Metrics

```python
from evaluate_model import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator()

# Run comprehensive evaluation
results = evaluator.run_comprehensive_evaluation(
    "path/to/test_reports.csv",
    "path/to/ground_truth.csv",
    detailed_report=True
)

# Access performance metrics
rule_f1 = results['rule_based']['metrics']['overall']['avg_f1']
ml_f1 = results['ml_based']['metrics']['overall']['avg_f1']
print(f"Rule-based F1: {rule_f1:.3f}")
print(f"ML F1: {ml_f1:.3f}")
```

## 🐛 Troubleshooting

### Common Issues

**1. spaCy Model Not Found**
```bash
# Download German model
python -m spacy download de_core_news_sm

# Verify installation
python -c "import spacy; nlp = spacy.load('de_core_news_sm'); print('✅ spaCy model loaded')"
```

**2. Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Check specific packages
pip list | grep -E "(spacy|pandas|scikit-learn|streamlit)"
```

**3. ML Models Not Loading**
- Train models first: `python train_model.py`
- Check `models/` directory exists and contains `.pkl` files
- Verify model metadata: `cat models/model_metadata.json`

**4. PDF Extraction Issues**
```python
# Test PDF extraction
from extractor.preprocessing import TextPreprocessor
preprocessor = TextPreprocessor()

# Try both methods
try:
    text1 = preprocessor.extract_text_from_pdf("report.pdf", method="pdfplumber")
    print("✅ pdfplumber successful")
except Exception as e:
    print(f"❌ pdfplumber failed: {e}")

try:
    text2 = preprocessor.extract_text_from_pdf("report.pdf", method="pypdf2")
    print("✅ PyPDF2 successful")
except Exception as e:
    print(f"❌ PyPDF2 failed: {e}")
```

**5. Performance Issues**
- Use smaller batch sizes for large datasets
- Process PDFs individually rather than in large batches
- Use rule-based only for faster processing
- Clear Streamlit cache: Click "🔄 Reload Models" in sidebar

**6. Memory Issues**
```bash
# Monitor memory usage
pip install psutil
python -c "
import psutil
print(f'Memory usage: {psutil.virtual_memory().percent}%')
"

# For large datasets, process in chunks
python -c "
import pandas as pd
chunk_size = 100
for chunk in pd.read_csv('large_dataset.csv', chunksize=chunk_size):
    # Process chunk
    pass
"
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-extractor`)
3. Add comprehensive tests for new functionality
4. Ensure all tests pass:
   ```bash
   python tests/test_keyword_extractor.py
   python tests/test_accuracy_metrics.py
   ```
5. Update documentation and examples
6. Submit pull request with performance metrics

## 📄 License

[MIT License] - See LICENSE file for details

---

**Note**: This system is designed for research and educational purposes. For clinical use, ensure proper validation and regulatory compliance.
