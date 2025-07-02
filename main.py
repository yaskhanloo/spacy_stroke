import os
import pandas as pd
from pathlib import Path
from typing import Dict
from extractor.preprocessing import TextPreprocessor
from extractor.keyword_rules import KeywordExtractor
from extractor.spacy_ner_wrapper import SpacyNERExtractor

class StrokeReportProcessor:
    """Main processor for stroke radiology reports."""
    
    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.keyword_extractor = KeywordExtractor()
        self.ner_extractor = SpacyNERExtractor()
        
        # Ensure output directory exists
        Path("output").mkdir(exist_ok=True)
    
    def process_single_report(self, text: str, report_id: str = None) -> Dict:
        """Process a single report and extract keywords."""
        # Preprocess text
        cleaned_text = self.preprocessor.clean_text(text)
        
        # Extract keywords using rules
        results = self.keyword_extractor.extract_all(cleaned_text, report_id)
        
        # Add NER entities for additional context
        entities = self.ner_extractor.extract_entities(cleaned_text)
        results['ner_entities'] = [ent['text'] for ent in entities if ent['label'] in ['PER', 'ORG', 'MISC']]
        
        return results
    
    def process_reports_from_folder(self, folder_path: str) -> pd.DataFrame:
        """Process all text files in a folder."""
        folder = Path(folder_path)
        results = []
        
        for file_path in folder.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                result = self.process_single_report(text, file_path.stem)
                results.append(result)
                print(f"Processed: {file_path.name}")
                
            except Exception as e:
                print(f"Error processing {file_path.name}: {e}")
        
        return pd.DataFrame(results)
    
    def process_csv_reports(self, csv_path: str, text_column: str, id_column: str = None) -> pd.DataFrame:
        """Process reports from a CSV file."""
        df = pd.read_csv(csv_path)
        results = []
        
        for idx, row in df.iterrows():
            text = row[text_column]
            report_id = row[id_column] if id_column else f"report_{idx}"
            
            try:
                result = self.process_single_report(text, report_id)
                results.append(result)
                print(f"Processed report: {report_id}")
                
            except Exception as e:
                print(f"Error processing report {report_id}: {e}")
        
        return pd.DataFrame(results)
    
    def save_results(self, df: pd.DataFrame, output_path: str = "output/extracted_keywords.csv"):
        """Save results to CSV."""
        df.to_csv(output_path, index=False)
        print(f"Results saved to: {output_path}")
        
# ===== Example Usage & Demo =====
def create_sample_data():
    """Create sample German stroke reports for testing."""
    sample_reports = [
        {
            'id': 'report_001',
            'text': """
            Patient wurde in Allgemeinanästhesie behandelt. Beginn der Intervention um 08:32 Uhr.
            Verwendung des Trevo Stentretriever Systems. rtPA wurde um 07:45 verabreicht.
            Mechanische Thrombektomie mit SOFIA Katheter durchgeführt.
            Finales Ergebnis: TICI 3. Keine Komplikationen aufgetreten.
            """
        },
        {
            'id': 'report_002', 
            'text': """
            Sedierung für die Prozedur. Start: 09:15 Uhr mit Aspiration using Penumbra System.
            Catch Mini device eingesetzt. Urokinase als Thrombolytikum verwendet.
            Leichte Blutung nach der Intervention beobachtet. TICI 2b erreicht.
            """
        },
        {
            'id': 'report_003',
            'text': """
            Lokale Anästhesie für den Eingriff. Beginn: 10:20. 
            Solitaire Stentretriever verwendet für die mechanische Rekanalisation.
            Heparin antikoagulation. Embotrap als backup device.
            Perforation der Gefäßwand aufgetreten. TICI 1 Ergebnis.
            """
        }
    ]
    
    # Create directory structure
    Path("data/raw_reports").mkdir(parents=True, exist_ok=True)
    
    # Save as individual text files
    for report in sample_reports:
        with open(f"data/raw_reports/{report['id']}.txt", 'w', encoding='utf-8') as f:
            f.write(report['text'])
    
    # Save as CSV
    df = pd.DataFrame(sample_reports)
    df.to_csv("data/sample_reports.csv", index=False)
    
    print("Sample data created in data/ folder")

def main():
    """Main execution function."""
    print("🧪 Stroke Report Keyword Extractor")
    print("=" * 40)
    
    # Create sample data if it doesn't exist
    if not Path("data").exists():
        print("Creating sample data...")
        create_sample_data()
    
    # Initialize processor
    processor = StrokeReportProcessor()
    
    # Process reports from folder
    print("\n📁 Processing reports from folder...")
    if Path("data/raw_reports").exists():
        results_df = processor.process_reports_from_folder("data/raw_reports")
        processor.save_results(results_df)
        
        print("\n📊 Results Summary:")
        # Display key CSV variables that are most likely to be present
        display_columns = ['report_id', 'anaesthesia', 'stent_retriever_used', 'aspiration_catheter_used', 
                          'periprocedural_ia_thrombolysis', 'tici_score', 'start_time_intervention', 'complications']
        available_columns = [col for col in display_columns if col in results_df.columns]
        print(results_df[available_columns].to_string())
    
    # Also process CSV if it exists
    if Path("data/sample_reports.csv").exists():
        print("\n📄 Processing CSV reports...")
        csv_results = processor.process_csv_reports("data/sample_reports.csv", "text", "id")
        processor.save_results(csv_results, "output/csv_extracted_keywords.csv")

if __name__ == "__main__":
    main()

# ===== Setup Instructions =====
"""
🚀 SETUP INSTRUCTIONS:

1. Install dependencies:
   pip install spacy pandas streamlit jupyter
   python -m spacy download de_core_news_sm

2. Create project structure:
   mkdir stroke_nlp_project
   cd stroke_nlp_project
   mkdir -p data/raw_reports extractor output notebooks

3. Save this code as separate files according to the structure

4. Run the project:
   python main.py

5. Optional - Create Streamlit app:
   streamlit run streamlit_app.py

The processor will:
- Create sample German stroke reports if none exist
- Extract keywords using rule-based patterns
- Use spaCy for additional NER and POS tagging
- Output structured CSV files with all extracted information
- Provide confidence scores and context for each extraction

Key Features:
✅ German medical terminology support
✅ Rule-based extraction with high precision
✅ spaCy integration for enhanced NLP
✅ Extensible pattern system
✅ CSV and folder input support
✅ Structured output with confidence scores
✅ Context preservation for manual review
"""