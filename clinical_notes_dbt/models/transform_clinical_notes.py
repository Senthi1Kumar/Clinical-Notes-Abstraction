# models/transform_clinical_notes.py

# As of now the DBT-python models supports only populart data platforms:
# Snowflake, BigQuery and Databricks - https://docs.getdbt.com/docs/build/python-models#specific-data-platforms


#------
from typing import Dict, List, Any
import torch
from gliner import GLiNER

class ClinicalNotesTransformer:
    def __init__(self, model_path: str = 'urchade/gliner_medium-v2.1'):
        """
        Initialize GLINER model for medical entity extraction
        
        Args:
            model_path (str): Path to pre-trained GLINER model
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = GLiNER.from_pretrained(model_path)
    
    def extract_medical_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract medical entities using GLINER
        
        Args:
            text (str): Input clinical note text
        
        Returns:
            Dict of extracted medical entities
        """
        try:
            # Custom medical entity types
            entity_types = [
                'medication', 
                'diagnosis', 
                'symptom', 
                'procedure', 
                'body_part'
            ]
            
            # Perform entity extraction
            entities = self.model.predict_entities(
                text, 
                labels=entity_types,
                threshold=0.5
            )
            
            # Organize extracted entities
            extracted_entities = {
                entity_type: [
                    ent['text'] for ent in entities 
                    if ent['label'] == entity_type
                ] for entity_type in entity_types
            }
            
            return {
                'entities': extracted_entities,
                'entity_count': sum(len(v) for v in extracted_entities.values())
            }
        
        except Exception as e:
            print(f"Entity extraction error: {e}")
            return {'entities': {}, 'entity_count': 0}
    
    def preprocess_clinical_note(self, text: str) -> Dict[str, Any]:
        """
        Comprehensive preprocessing of clinical note
        
        Args:
            text (str): Input clinical note
        
        Returns:
            Dict of preprocessed features
        """
        # Text basic preprocessing
        cleaned_text = text.lower().strip()
        
        # Extract medical entities
        medical_entities = self.extract_medical_entities(text)
        
        return {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'text_length': len(cleaned_text),
            'medical_entities': medical_entities['entities'],
            'entity_count': medical_entities['entity_count']
        }

def model(dbt, session):
    """
    dbt Python model for clinical notes transformation
    
    Args:
        dbt: dbt context
        session: Database session
    
    Returns:
        Transformed DataFrame
    """
    # Initialize transformer
    transformer = ClinicalNotesTransformer()

    dbt.config(
        # materialized = "table",
        packages = ["torch", "gliner", "typing"]
    )
    
    # Fetch raw clinical notes
    # df = dbt.ref('clinical_notes_staging') 
    df = dbt.source('clinical_notes_source', 'clinical_notes')
    
    # Apply preprocessing to notes
    df['processed_note'] = df['note'].apply(transformer.preprocess_clinical_note)
    
    # Flatten processed data
    df['cleaned_text'] = df['processed_note'].apply(lambda x: x['cleaned_text'])
    df['text_length'] = df['processed_note'].apply(lambda x: x['text_length'])
    df['medical_entities'] = df['processed_note'].apply(lambda x: x['medical_entities'])
    df['entity_count'] = df['processed_note'].apply(lambda x: x['entity_count'])
    
    return df