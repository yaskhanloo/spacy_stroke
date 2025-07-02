#!/usr/bin/env python3
"""
Advanced Streamlit app for German Stroke Radiology Report NLP Extraction.

Features:
- Rule-based and ML model extraction
- PDF processing
- Model training and evaluation
- Accuracy metrics visualization
- Batch processing
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os
import json
import tempfile
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO

# Add the current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from extractor.preprocessing import TextPreprocessor
    from extractor.keyword_rules import KeywordExtractor
    from extractor.spacy_ner_wrapper import SpacyNERExtractor
    from extractor.ml_model import StrokeMLExtractor, create_sample_training_data
    from evaluation.metrics import AccuracyMetrics
    from evaluation.validation import ModelValidator, create_sample_validation_set
    from train_model import TrainingPipeline
    from evaluate_model import ModelEvaluator
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Please ensure all dependencies are installed: pip install -r requirements.txt")

# Configure Streamlit page
st.set_page_config(
    page_title="🧠 Advanced Stroke NLP Extractor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-metric {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .warning-metric {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_processors():
    """Load and cache the NLP processors."""
    try:
        preprocessor = TextPreprocessor()
        keyword_extractor = KeywordExtractor()
        ner_extractor = SpacyNERExtractor()
        ml_extractor = StrokeMLExtractor()
        
        # Try to load ML models if available
        try:
            ml_extractor.load_models("models/")
            ml_available = True
        except:
            ml_available = False
            
        return preprocessor, keyword_extractor, ner_extractor, ml_extractor, ml_available
    except Exception as e:
        st.error(f"Error loading NLP models: {e}")
        st.info("Please install the German spaCy model: `python -m spacy download de_core_news_sm`")
        return None, None, None, None, False

@st.cache_data
def load_sample_data():
    """Load sample validation data."""
    try:
        reports_df, truth_df = create_sample_validation_set()
        return reports_df, truth_df
    except Exception as e:
        st.error(f"Error loading sample data: {e}")
        return None, None

def extract_text_from_pdf_files(uploaded_files, preprocessor):
    """Extract text from uploaded PDF files."""
    extracted_texts = []
    
    for uploaded_file in uploaded_files:
        try:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            # Extract text
            extracted_text = preprocessor.extract_text_from_pdf(tmp_path)
            cleaned_text = preprocessor.clean_text(extracted_text)
            
            extracted_texts.append({
                'filename': uploaded_file.name,
                'raw_text': extracted_text,
                'cleaned_text': cleaned_text,
                'word_count': len(cleaned_text.split()) if cleaned_text else 0
            })
            
            # Clean up temporary file
            os.unlink(tmp_path)
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
            extracted_texts.append({
                'filename': uploaded_file.name,
                'error': str(e),
                'raw_text': None,
                'cleaned_text': None
            })
    
    return extracted_texts

def process_single_report(text, report_id, preprocessor, rule_extractor, ml_extractor, use_ml):
    """Process a single report with both rule-based and ML extraction."""
    # Preprocess text
    cleaned_text = preprocessor.clean_text(text)
    
    # Rule-based extraction
    rule_results = rule_extractor.extract_all(cleaned_text, report_id)
    
    # ML extraction if available
    ml_results = None
    if use_ml and ml_extractor.models:
        try:
            ml_results = ml_extractor.extract_with_ml(cleaned_text, report_id)
        except Exception as e:
            st.warning(f"ML extraction failed: {e}")
    
    return {
        'cleaned_text': cleaned_text,
        'rule_results': rule_results,
        'ml_results': ml_results,
        'original_text': text
    }

def display_extraction_results(results, method_name):
    """Display extraction results in a formatted way."""
    if not results:
        st.warning(f"No {method_name} results available")
        return
    
    st.subheader(f"🎯 {method_name} Results")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        # Left column
        if results.get('anesthesia'):
            st.success(f"**Anesthesia:** {results['anesthesia']}")
        
        if results.get('medication'):
            st.info(f"**Medication:** {results['medication']}")
        
        if results.get('device'):
            st.info(f"**Device:** {results['device']}")
        
        if results.get('treatment_method'):
            st.info(f"**Treatment:** {results['treatment_method']}")
    
    with col2:
        # Right column
        if results.get('tici_score'):
            st.warning(f"**TICI Score:** {results['tici_score']}")
        
        if results.get('times'):
            st.info(f"**Times:** {results['times']}")
        
        if results.get('complications'):
            st.error(f"**Complications:** {results['complications']}")
    
    # Show as DataFrame
    results_df = pd.DataFrame([results])
    st.dataframe(results_df, use_container_width=True)

def create_performance_visualization(metrics_data):
    """Create performance visualization plots."""
    if not metrics_data:
        return None
    
    # Extract metrics for plotting
    categories = []
    f1_scores = []
    precision_scores = []
    recall_scores = []
    
    for category, metrics in metrics_data.items():
        if category != 'overall' and isinstance(metrics, dict):
            categories.append(category.title())
            f1_scores.append(metrics.get('f1', 0))
            precision_scores.append(metrics.get('precision', 0))
            recall_scores.append(metrics.get('recall', 0))
    
    if not categories:
        return None
    
    # Create radar chart
    fig = go.Figure()
    
    # Add F1 scores
    fig.add_trace(go.Scatterpolar(
        r=f1_scores,
        theta=categories,
        fill='toself',
        name='F1 Score',
        line_color='blue'
    ))
    
    # Add Precision scores
    fig.add_trace(go.Scatterpolar(
        r=precision_scores,
        theta=categories,
        fill='toself',
        name='Precision',
        line_color='green'
    ))
    
    # Add Recall scores
    fig.add_trace(go.Scatterpolar(
        r=recall_scores,
        theta=categories,
        fill='toself',
        name='Recall',
        line_color='red'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        title="Model Performance by Category",
        showlegend=True
    )
    
    return fig

def main():
    # Main title
    st.markdown('<h1 class="main-header">🧠 Advanced German Stroke NLP Extractor</h1>', unsafe_allow_html=True)
    st.markdown("**Hybrid NLP System: Rule-Based + Machine Learning + PDF Processing**")
    
    # Load processors
    preprocessor, rule_extractor, ner_extractor, ml_extractor, ml_available = load_processors()
    
    if not all([preprocessor, rule_extractor, ner_extractor]):
        st.error("Failed to load core NLP components. Please check your installation.")
        st.stop()
    
    # Sidebar navigation
    with st.sidebar:
        st.header("🚀 Navigation")
        
        page = st.selectbox(
            "Choose a page:",
            [
                "📄 Report Extraction",
                "📊 Model Training",
                "📈 Model Evaluation", 
                "🔬 Batch Processing",
                "📚 Documentation"
            ]
        )
        
        st.markdown("---")
        st.header("ℹ️ System Status")
        
        # System status indicators
        st.success("✅ Rule-based Model")
        if ml_available:
            st.success("✅ ML Models Loaded")
        else:
            st.warning("⚠️ ML Models Not Available")
        
        st.info(f"📦 spaCy Model: {'✅' if ner_extractor else '❌'}")
        
        st.markdown("---")
        st.header("📝 Quick Actions")
        
        if st.button("🔄 Reload Models"):
            st.cache_resource.clear()
            st.rerun()
        
        if st.button("📊 Load Sample Data"):
            st.session_state.load_sample = True

    # Main content based on selected page
    if page == "📄 Report Extraction":
        show_extraction_page(preprocessor, rule_extractor, ml_extractor, ml_available)
    
    elif page == "📊 Model Training":
        show_training_page()
    
    elif page == "📈 Model Evaluation":
        show_evaluation_page()
    
    elif page == "🔬 Batch Processing":
        show_batch_processing_page(preprocessor, rule_extractor, ml_extractor, ml_available)
    
    elif page == "📚 Documentation":
        show_documentation_page()

def show_extraction_page(preprocessor, rule_extractor, ml_extractor, ml_available):
    """Show the main extraction interface."""
    
    st.header("📄 Single Report Extraction")
    
    # Input options
    input_method = st.radio(
        "Choose input method:",
        ["📝 Text Input", "📁 Upload PDF", "📄 Upload Text File"]
    )
    
    input_text = ""
    
    if input_method == "📝 Text Input":
        # Sample text button
        col1, col2 = st.columns([3, 1])
        
        with col2:
            if st.button("📋 Load Sample"):
                sample_text = """Patient mit akutem ischämischem Schlaganfall. Allgemeinanästhesie eingeleitet.
Interventionsbeginn: 08:30 Uhr. rtPA bereits präklinisch verabreicht.
Mechanische Thrombektomie mit Trevo Stentretriever durchgeführt.
TICI 3 Rekanalisierung erreicht. Keine intraoperativen Komplikationen."""
                st.session_state.sample_text = sample_text
        
        input_text = st.text_area(
            "Enter German stroke report:",
            value=st.session_state.get('sample_text', ''),
            height=200,
            placeholder="Paste your German medical report here..."
        )
    
    elif input_method == "📁 Upload PDF":
        uploaded_pdf = st.file_uploader("Upload PDF file", type=['pdf'])
        
        if uploaded_pdf:
            with st.spinner("Extracting text from PDF..."):
                try:
                    # Save uploaded file temporarily
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(uploaded_pdf.read())
                        tmp_path = tmp_file.name
                    
                    # Extract text
                    extracted_text = preprocessor.extract_text_from_pdf(tmp_path)
                    input_text = extracted_text
                    
                    # Show extraction preview
                    st.success(f"✅ Extracted {len(extracted_text.split())} words from PDF")
                    
                    with st.expander("📖 Preview extracted text"):
                        st.text_area("Extracted text:", extracted_text[:1000] + "..." if len(extracted_text) > 1000 else extracted_text, height=150)
                    
                    # Clean up
                    os.unlink(tmp_path)
                    
                except Exception as e:
                    st.error(f"PDF extraction failed: {e}")
    
    elif input_method == "📄 Upload Text File":
        uploaded_txt = st.file_uploader("Upload text file", type=['txt'])
        
        if uploaded_txt:
            input_text = uploaded_txt.read().decode('utf-8')
            st.success(f"✅ Loaded {len(input_text.split())} words from text file")
    
    # Extraction options
    st.subheader("🔧 Extraction Options")
    
    col1, col2 = st.columns(2)
    with col1:
        use_rule_based = st.checkbox("🔧 Rule-based Extraction", value=True)
    with col2:
        use_ml = st.checkbox("🤖 ML Extraction", value=ml_available, disabled=not ml_available)
    
    # Process button
    if st.button("🔍 Extract Information", type="primary") and input_text.strip():
        with st.spinner("Processing report..."):
            
            results = process_single_report(
                input_text, "manual_input", preprocessor, 
                rule_extractor, ml_extractor, use_ml
            )
            
            st.session_state.extraction_results = results
    
    # Display results
    if 'extraction_results' in st.session_state:
        results = st.session_state.extraction_results
        
        st.header("📊 Extraction Results")
        
        # Create tabs for different results
        tabs = []
        if use_rule_based:
            tabs.append("🔧 Rule-based")
        if use_ml and results['ml_results']:
            tabs.append("🤖 ML Model")
        tabs.extend(["📝 Text Processing", "📤 Export"])
        
        tab_objects = st.tabs(tabs)
        tab_index = 0
        
        # Rule-based results
        if use_rule_based:
            with tab_objects[tab_index]:
                display_extraction_results(results['rule_results'], "Rule-based")
            tab_index += 1
        
        # ML results
        if use_ml and results['ml_results']:
            with tab_objects[tab_index]:
                display_extraction_results(results['ml_results'], "ML Model")
            tab_index += 1
        
        # Text processing tab
        with tab_objects[tab_index]:
            st.subheader("🧹 Text Processing")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Original Text:**")
                st.text_area("", results['original_text'][:500] + "..." if len(results['original_text']) > 500 else results['original_text'], height=200, disabled=True)
            
            with col2:
                st.write("**Cleaned Text:**")
                st.text_area("", results['cleaned_text'][:500] + "..." if len(results['cleaned_text']) > 500 else results['cleaned_text'], height=200, disabled=True)
        
        tab_index += 1
        
        # Export tab
        with tab_objects[tab_index]:
            st.subheader("💾 Export Results")
            
            # Prepare export data
            export_data = {}
            if use_rule_based:
                export_data.update(results['rule_results'])
            
            if use_ml and results['ml_results']:
                for key, value in results['ml_results'].items():
                    if key not in export_data:
                        export_data[f"ml_{key}"] = value
            
            export_df = pd.DataFrame([export_data])
            
            # Download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                csv = export_df.to_csv(index=False)
                st.download_button(
                    label="📄 Download CSV",
                    data=csv,
                    file_name="stroke_extraction_results.csv",
                    mime="text/csv"
                )
            
            with col2:
                json_data = json.dumps(export_data, indent=2, ensure_ascii=False)
                st.download_button(
                    label="📄 Download JSON",
                    data=json_data,
                    file_name="stroke_extraction_results.json",
                    mime="application/json"
                )

def show_training_page():
    """Show the model training interface."""
    
    st.header("📊 Model Training")
    
    st.info("🚀 Train hybrid NLP models on your stroke radiology reports")
    
    # Training options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📁 Training Data")
        
        use_sample_data = st.checkbox("Use sample training data", value=True)
        
        if not use_sample_data:
            training_reports = st.file_uploader("Training reports CSV", type=['csv'])
            training_ground_truth = st.file_uploader("Training ground truth CSV", type=['csv'])
        
        validation_folds = st.slider("Cross-validation folds", 3, 10, 5)
    
    with col2:
        st.subheader("🔧 Model Options")
        
        train_rule_based = st.checkbox("Evaluate rule-based model", value=True)
        train_ml_models = st.checkbox("Train ML models", value=True)
        generate_plots = st.checkbox("Generate performance plots", value=True)
    
    # Start training
    if st.button("🚀 Start Training", type="primary"):
        with st.spinner("Training models... This may take a few minutes."):
            
            try:
                # Initialize training pipeline
                config = {
                    'training': {'validation_folds': validation_folds},
                    'models': {
                        'rule_based': train_rule_based,
                        'ml_models': train_ml_models
                    },
                    'evaluation': {'plot_metrics': generate_plots}
                }
                
                pipeline = TrainingPipeline(config)
                
                # Run training
                training_results = pipeline.run_full_training_pipeline()
                
                st.session_state.training_results = training_results
                
                st.success("✅ Training completed successfully!")
                
            except Exception as e:
                st.error(f"Training failed: {e}")
                st.exception(e)
    
    # Display training results
    if 'training_results' in st.session_state:
        results = st.session_state.training_results
        
        st.header("📈 Training Results")
        
        # Overall performance metrics
        if 'comparison' in results:
            comparison = results['comparison']
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'rule_based' in comparison:
                    st.metric(
                        "Rule-based F1",
                        f"{comparison['rule_based']['overall_f1']:.3f}",
                        delta=None
                    )
            
            with col2:
                if 'ml_based' in comparison:
                    st.metric(
                        "ML Model F1",
                        f"{comparison['ml_based']['overall_f1']:.3f}",
                        delta=None
                    )
            
            with col3:
                if 'best_model' in comparison:
                    st.metric(
                        "Best Model",
                        comparison['best_model']['name'].title(),
                        delta=None
                    )
            
            # Recommendations
            if 'recommendations' in comparison:
                st.subheader("💡 Recommendations")
                for rec in comparison['recommendations']:
                    st.write(f"• {rec}")

def show_evaluation_page():
    """Show the model evaluation interface."""
    
    st.header("📈 Model Evaluation")
    
    st.info("🔬 Evaluate model performance against ground truth data")
    
    # Evaluation options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Test Data")
        
        use_sample_test = st.checkbox("Use sample test data", value=True)
        
        if not use_sample_test:
            test_reports = st.file_uploader("Test reports CSV", type=['csv'])
            test_ground_truth = st.file_uploader("Test ground truth CSV", type=['csv'])
    
    with col2:
        st.subheader("🔧 Evaluation Options")
        
        evaluate_rule_based = st.checkbox("Evaluate rule-based model", value=True)
        evaluate_ml = st.checkbox("Evaluate ML model", value=True)
        generate_detailed_report = st.checkbox("Generate detailed analysis", value=True)
    
    # Run evaluation
    if st.button("📊 Run Evaluation", type="primary"):
        with st.spinner("Evaluating models..."):
            
            try:
                if use_sample_test:
                    # Load sample validation data
                    reports_df, truth_df = load_sample_data()
                    
                    if reports_df is not None and truth_df is not None:
                        evaluator = ModelEvaluator()
                        
                        # Save sample data temporarily
                        with tempfile.TemporaryDirectory() as temp_dir:
                            reports_path = f"{temp_dir}/test_reports.csv"
                            truth_path = f"{temp_dir}/test_ground_truth.csv"
                            
                            reports_df.to_csv(reports_path, index=False)
                            truth_df.to_csv(truth_path, index=False)
                            
                            # Run evaluation
                            evaluation_results = evaluator.run_comprehensive_evaluation(
                                reports_path, truth_path, generate_detailed_report
                            )
                            
                            st.session_state.evaluation_results = evaluation_results
                
                st.success("✅ Evaluation completed!")
                
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                st.exception(e)
    
    # Display evaluation results
    if 'evaluation_results' in st.session_state:
        results = st.session_state.evaluation_results
        
        st.header("📊 Evaluation Results")
        
        # Performance metrics
        if 'comparison' in results:
            comparison = results['comparison']
            
            # Create performance visualization
            if 'rule_based' in results and 'error' not in results['rule_based']:
                rule_metrics = results['rule_based']['metrics']
                fig = create_performance_visualization(rule_metrics)
                
                if fig:
                    st.plotly_chart(fig, use_container_width=True)
            
            # Summary table
            st.subheader("📋 Performance Summary")
            
            summary_data = []
            
            if 'rule_based' in comparison:
                summary_data.append({
                    'Model': 'Rule-based',
                    'F1 Score': f"{comparison['rule_based']['f1_score']:.3f}",
                    'Precision': f"{comparison['rule_based']['precision']:.3f}",
                    'Recall': f"{comparison['rule_based']['recall']:.3f}",
                    'Processing Time (s)': f"{comparison['rule_based']['avg_processing_time']:.4f}"
                })
            
            if 'ml_based' in comparison:
                summary_data.append({
                    'Model': 'ML-based',
                    'F1 Score': f"{comparison['ml_based']['f1_score']:.3f}",
                    'Precision': f"{comparison['ml_based']['precision']:.3f}",
                    'Recall': f"{comparison['ml_based']['recall']:.3f}",
                    'Processing Time (s)': f"{comparison['ml_based']['avg_processing_time']:.4f}"
                })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)

def show_batch_processing_page(preprocessor, rule_extractor, ml_extractor, ml_available):
    """Show the batch processing interface."""
    
    st.header("🔬 Batch Processing")
    
    st.info("📁 Process multiple files at once (PDF, TXT, or CSV)")
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload files for batch processing",
        type=['pdf', 'txt', 'csv'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} files uploaded")
        
        # Processing options
        col1, col2 = st.columns(2)
        
        with col1:
            use_rule_based = st.checkbox("🔧 Rule-based extraction", value=True)
        with col2:
            use_ml = st.checkbox("🤖 ML extraction", value=ml_available, disabled=not ml_available)
        
        # Process files
        if st.button("🚀 Process All Files", type="primary"):
            with st.spinner("Processing files..."):
                
                batch_results = []
                
                for i, uploaded_file in enumerate(uploaded_files):
                    st.write(f"Processing {uploaded_file.name}...")
                    
                    try:
                        # Extract text based on file type
                        if uploaded_file.name.endswith('.pdf'):
                            # Save temporarily and extract
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                                tmp_file.write(uploaded_file.read())
                                tmp_path = tmp_file.name
                            
                            text = preprocessor.extract_text_from_pdf(tmp_path)
                            os.unlink(tmp_path)
                            
                        elif uploaded_file.name.endswith('.txt'):
                            text = uploaded_file.read().decode('utf-8')
                            
                        elif uploaded_file.name.endswith('.csv'):
                            # Handle CSV with text column
                            df = pd.read_csv(uploaded_file)
                            if 'text' in df.columns:
                                # Process each row
                                for idx, row in df.iterrows():
                                    text = row['text']
                                    report_id = row.get('report_id', f"{uploaded_file.name}_{idx}")
                                    
                                    result = process_single_report(
                                        text, report_id, preprocessor,
                                        rule_extractor, ml_extractor, use_ml
                                    )
                                    
                                    result['filename'] = uploaded_file.name
                                    result['row_index'] = idx
                                    batch_results.append(result)
                                
                                continue
                            else:
                                st.error(f"CSV file {uploaded_file.name} missing 'text' column")
                                continue
                        
                        # Process single file
                        result = process_single_report(
                            text, uploaded_file.name, preprocessor,
                            rule_extractor, ml_extractor, use_ml
                        )
                        
                        result['filename'] = uploaded_file.name
                        batch_results.append(result)
                        
                    except Exception as e:
                        st.error(f"Error processing {uploaded_file.name}: {e}")
                
                st.session_state.batch_results = batch_results
                st.success(f"✅ Processed {len(batch_results)} items")
        
        # Display batch results
        if 'batch_results' in st.session_state:
            results = st.session_state.batch_results
            
            st.header("📊 Batch Processing Results")
            
            # Create summary table
            summary_data = []
            
            for result in results:
                row = {'filename': result['filename']}
                
                if use_rule_based and result['rule_results']:
                    row.update({f"rule_{k}": v for k, v in result['rule_results'].items() 
                               if k not in ['report_id', 'text_length']})
                
                if use_ml and result.get('ml_results'):
                    row.update({f"ml_{k}": v for k, v in result['ml_results'].items() 
                               if k not in ['report_id', 'text_length']})
                
                summary_data.append(row)
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                st.dataframe(summary_df, use_container_width=True)
                
                # Download batch results
                csv_data = summary_df.to_csv(index=False)
                st.download_button(
                    label="📄 Download Batch Results",
                    data=csv_data,
                    file_name="batch_stroke_extraction_results.csv",
                    mime="text/csv"
                )

def show_documentation_page():
    """Show documentation and help."""
    
    st.header("📚 Documentation")
    
    # Quick start guide
    st.subheader("🚀 Quick Start Guide")
    
    st.markdown("""
    ### 1. Single Report Extraction
    - Go to **📄 Report Extraction**
    - Input text manually, upload PDF, or upload text file
    - Choose extraction method (Rule-based, ML, or both)
    - Click **🔍 Extract Information**
    
    ### 2. Model Training
    - Go to **📊 Model Training**
    - Configure training options
    - Click **🚀 Start Training**
    - Review performance metrics and recommendations
    
    ### 3. Model Evaluation
    - Go to **📈 Model Evaluation**
    - Upload test data or use sample data
    - Click **📊 Run Evaluation**
    - Analyze detailed performance metrics
    
    ### 4. Batch Processing
    - Go to **🔬 Batch Processing**
    - Upload multiple PDF, TXT, or CSV files
    - Choose extraction methods
    - Download results as CSV
    """)
    
    # Extracted variables
    st.subheader("📋 Extracted Variables")
    
    variables_df = pd.DataFrame([
        {'Variable': 'anesthesia', 'Description': 'Type of anesthesia used', 'Examples': 'allgemeinanästhesie, sedierung, lokalanästhesie'},
        {'Variable': 'medication', 'Description': 'Medications administered', 'Examples': 'rtpa, heparin, urokinase, tenecteplase'},
        {'Variable': 'device', 'Description': 'Medical devices used', 'Examples': 'trevo, sofia, solitaire, penumbra'},
        {'Variable': 'treatment_method', 'Description': 'Treatment approach', 'Examples': 'thrombektomie, aspiration, stentretriever'},
        {'Variable': 'tici_score', 'Description': 'TICI recanalization score', 'Examples': 'tici 3, tici 2b, tici 1'},
        {'Variable': 'times', 'Description': 'Procedure timestamps', 'Examples': '08:32, 09:15, 14:30'},
        {'Variable': 'complications', 'Description': 'Reported complications', 'Examples': 'perforation, blutung, hämatom'}
    ])
    
    st.dataframe(variables_df, use_container_width=True)
    
    # Installation instructions
    st.subheader("🔧 Installation & Setup")
    
    st.code("""
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download German spaCy model
python -m spacy download de_core_news_sm

# 3. Run Streamlit app
streamlit run streamlit_app.py
    """, language="bash")
    
    # Command line usage
    st.subheader("💻 Command Line Usage")
    
    st.code("""
# Train models
python train_model.py --data data/training/ --validation data/validation/

# Evaluate models
python evaluate_model.py --test-reports data/test/reports.csv --test-ground-truth data/test/truth.csv

# Run tests
python tests/test_keyword_extractor.py
python tests/test_accuracy_metrics.py
    """, language="bash")
    
    # Troubleshooting
    st.subheader("🔍 Troubleshooting")
    
    with st.expander("Common Issues"):
        st.markdown("""
        **spaCy model not found:**
        - Run: `python -m spacy download de_core_news_sm`
        
        **Import errors:**
        - Ensure all dependencies are installed: `pip install -r requirements.txt`
        
        **ML models not loading:**
        - Train models first using the training page or command line
        
        **PDF extraction fails:**
        - Try different PDF files or convert to text manually
        
        **Performance issues:**
        - Use smaller batch sizes for large datasets
        - Consider using rule-based only for faster processing
        """)

# Footer
st.markdown("---")
st.markdown("**🧠 Advanced German Stroke NLP Extractor** | Built with Streamlit, spaCy & scikit-learn")

if __name__ == "__main__":
    main()