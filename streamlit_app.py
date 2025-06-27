# ===== streamlit_app.py =====
import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import os

# Add the current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from extractor.preprocessing import TextPreprocessor
from extractor.keyword_rules import KeywordExtractor
from extractor.spacy_ner_wrapper import SpacyNERExtractor

# Configure Streamlit page
st.set_page_config(
    page_title="Stroke Report Keyword Extractor",
    page_icon="🧠",
    layout="wide"
)

@st.cache_resource
def load_processors():
    """Load and cache the NLP processors."""
    try:
        preprocessor = TextPreprocessor()
        keyword_extractor = KeywordExtractor()
        ner_extractor = SpacyNERExtractor()
        return preprocessor, keyword_extractor, ner_extractor
    except Exception as e:
        st.error(f"Error loading NLP models: {e}")
        st.info("Please install the German spaCy model: `python -m spacy download de_core_news_sm`")
        return None, None, None

def process_text(text, preprocessor, keyword_extractor, ner_extractor):
    """Process a single text and return results."""
    # Preprocess
    cleaned_text = preprocessor.clean_text(text)
    
    # Extract keywords
    keyword_results = keyword_extractor.extract_all(cleaned_text)
    
    # Extract NER entities
    entities = ner_extractor.extract_entities(cleaned_text)
    
    # Extract noun phrases
    noun_phrases = ner_extractor.extract_noun_phrases(cleaned_text)
    
    return {
        'cleaned_text': cleaned_text,
        'keyword_results': keyword_results,
        'entities': entities,
        'noun_phrases': noun_phrases
    }

def main():
    # Title and description
    st.title("🧠 Stroke Report Keyword Extractor")
    st.markdown("**Classical NLP Pipeline for German Medical Reports**")
    
    # Load processors
    preprocessor, keyword_extractor, ner_extractor = load_processors()
    
    if not all([preprocessor, keyword_extractor, ner_extractor]):
        st.stop()
    
    # Sidebar with instructions
    with st.sidebar:
        st.header("ℹ️ Instructions")
        st.markdown("""
        1. **Enter Text**: Paste a German stroke report in the text area
        2. **Process**: Click the button to extract keywords
        3. **Review**: Check the results in different tabs
        4. **Download**: Export results as CSV
        
        **Categories Extracted:**
        - Anesthesia types
        - Medications
        - Treatment methods  
        - Medical devices
        - TICI scores
        - Times
        - Complications
        """)
        
        st.header("📝 Sample Text")
        if st.button("Load Sample Report"):
            sample_text = """
            Patient wurde in Allgemeinanästhesie behandelt. Beginn der Intervention um 08:32 Uhr.
            Verwendung des Trevo Stentretriever Systems. rtPA wurde um 07:45 verabreicht.
            Mechanische Thrombektomie mit SOFIA Katheter durchgeführt.
            Finales Ergebnis: TICI 3. Keine Komplikationen aufgetreten.
            """
            st.session_state.sample_text = sample_text
    
    # Main input area
    st.header("📄 Input Report")
    
    # Get initial text from sample if available
    initial_text = st.session_state.get('sample_text', '')
    
    # Text input
    input_text = st.text_area(
        "Enter German stroke report:",
        value=initial_text,
        height=200,
        placeholder="Paste your German medical report here..."
    )
    
    # File upload option
    st.subheader("📁 Or Upload Files")
    uploaded_files = st.file_uploader(
        "Upload text files",
        type=['txt'],
        accept_multiple_files=True
    )
    
    # Process button
    if st.button("🔍 Extract Keywords", type="primary"):
        if input_text.strip():
            with st.spinner("Processing report..."):
                # Process the input text
                results = process_text(input_text, preprocessor, keyword_extractor, ner_extractor)
                st.session_state.results = results
                st.session_state.processed_text = input_text
        elif uploaded_files:
            with st.spinner("Processing uploaded files..."):
                all_results = []
                for uploaded_file in uploaded_files:
                    text = uploaded_file.read().decode('utf-8')
                    result = process_text(text, preprocessor, keyword_extractor, ner_extractor)
                    result['filename'] = uploaded_file.name
                    all_results.append(result)
                st.session_state.uploaded_results = all_results
        else:
            st.warning("Please enter text or upload files to process.")
    
    # Display results if available
    if 'results' in st.session_state:
        st.header("📊 Extraction Results")
        results = st.session_state.results
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["Keywords", "Text Processing", "Named Entities", "Export"])
        
        with tab1:
            st.subheader("🎯 Extracted Keywords")
            
            # Display in columns
            col1, col2 = st.columns(2)
            
            with col1:
                keyword_results = results['keyword_results']
                
                # Anesthesia
                if keyword_results.get('anesthesia'):
                    st.success(f"**Anesthesia:** {keyword_results['anesthesia']}")
                
                # Medication
                if keyword_results.get('medication'):
                    st.info(f"**Medication:** {keyword_results['medication']}")
                
                # Treatment method
                if keyword_results.get('treatment_method'):
                    st.info(f"**Treatment:** {keyword_results['treatment_method']}")
                
                # Device
                if keyword_results.get('device'):
                    st.info(f"**Device:** {keyword_results['device']}")
            
            with col2:
                # TICI Score
                if keyword_results.get('tici_score'):
                    st.warning(f"**TICI Score:** {keyword_results['tici_score']}")
                
                # Times
                if keyword_results.get('times'):
                    st.info(f"**Times:** {keyword_results['times']}")
                
                # Complications
                if keyword_results.get('complications'):
                    st.error(f"**Complications:** {keyword_results['complications']}")
            
            # Full results table
            st.subheader("📋 Complete Results")
            results_df = pd.DataFrame([keyword_results])
            st.dataframe(results_df, use_container_width=True)
        
        with tab2:
            st.subheader("🧹 Text Processing")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Original Text:**")
                st.text_area("", st.session_state.processed_text, height=200, disabled=True)
            
            with col2:
                st.write("**Cleaned Text:**")
                st.text_area("", results['cleaned_text'], height=200, disabled=True)
            
            # Noun phrases
            st.subheader("📝 Extracted Noun Phrases")
            if results['noun_phrases']:
                for i, phrase in enumerate(results['noun_phrases'][:10]):  # Show first 10
                    st.code(phrase)
            else:
                st.info("No significant noun phrases found.")
        
        with tab3:
            st.subheader("🏷️ Named Entities (spaCy)")
            
            if results['entities']:
                entities_df = pd.DataFrame(results['entities'])
                st.dataframe(entities_df, use_container_width=True)
                
                # Entity visualization
                st.subheader("📊 Entity Distribution")
                entity_counts = entities_df['label'].value_counts()
                st.bar_chart(entity_counts)
            else:
                st.info("No named entities found.")
        
        with tab4:
            st.subheader("💾 Export Results")
            
            # Convert to DataFrame for export
            export_df = pd.DataFrame([results['keyword_results']])
            
            # CSV download
            csv = export_df.to_csv(index=False)
            st.download_button(
                label="📄 Download as CSV",
                data=csv,
                file_name="stroke_report_extraction.csv",
                mime="text/csv"
            )
            
            # JSON download
            import json
            json_data = json.dumps(results['keyword_results'], indent=2, ensure_ascii=False)
            st.download_button(
                label="📄 Download as JSON",
                data=json_data,
                file_name="stroke_report_extraction.json",
                mime="application/json"
            )
            
            st.info("💡 Tip: You can process multiple reports and combine the CSV files for batch analysis.")
    
    # Display uploaded file results
    if 'uploaded_results' in st.session_state:
        st.header("📁 Batch Processing Results")
        
        uploaded_results = st.session_state.uploaded_results
        
        # Create summary DataFrame
        summary_data = []
        for result in uploaded_results:
            summary_row = result['keyword_results'].copy()
            summary_row['filename'] = result['filename']
            summary_data.append(summary_row)
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True)
        
        # Download batch results
        batch_csv = summary_df.to_csv(index=False)
        st.download_button(
            label="📄 Download Batch Results as CSV",
            data=batch_csv,
            file_name="batch_stroke_reports_extraction.csv",
            mime="text/csv"
        )
    
    # Footer
    st.markdown("---")
    st.markdown("**🧪 Stroke Report Keyword Extractor** | Built with Classical NLP & spaCy")

if __name__ == "__main__":
    main()