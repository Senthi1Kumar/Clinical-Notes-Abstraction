import os
import datetime
import json
import logging
import psycopg
import torch
from gliner import GLiNER
from dotenv import load_dotenv
from tqdm.auto import tqdm
from typing import Dict, List, Any

class ClinicalNotesProcessor:
    def __init__(
        self, 
        model_path: str = 'gliner-community/gliner_small-v2.5',
        output_directory: str = 'processed_notes'
    ):
        # Logging Configuration
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename='clinical_data_processing.log'
        )
        
        self.logger = logging.getLogger(__name__)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = GLiNER.from_pretrained(model_path).to(self.device)
        
        # Output
        os.makedirs(output_directory, exist_ok=True)
        self.output_path = output_directory

    def process_clinical_corpus(
        self, 
        database_connection,
        batch_size: int = 100
    ):
        cursor = database_connection.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM clinical_notes")
        total_records = cursor.fetchone()[0]
        
        cursor.execute("SELECT idx, note FROM clinical_notes")
        
        with tqdm(
            total=total_records, 
            desc="📊 Medical Corpus Processing", 
            unit="document",
            dynamic_ncols=True,
            colour='green'
        ) as progress_bar:
            processed_batch = []
            
            for note_id, note_text in cursor:
                try:
                    processed_note = self._transform_clinical_document(
                        note_id, 
                        note_text
                    )
                    processed_batch.append(processed_note)
                    
                    # Batch Serialization
                    if len(processed_batch) >= batch_size:
                        self._serialize_batch(processed_batch)
                        processed_batch.clear()
                    
                    progress_bar.update(1)
                
                except Exception as computational_anomaly:
                    self.logger.error(f"Processing Disruption: {computational_anomaly}")
            
            if processed_batch:
                self._serialize_batch(processed_batch)

    def _transform_clinical_document(
        self, 
        document_id: int, 
        document_text: str
    ) -> Dict[str, Any]:
        cleaned_text = self._normalize_unicode_characters(document_text.lower().strip())
        medical_entities = self._extract_medical_entities(document_text)
        
        return {
            'id': document_id,
            'original_text': document_text,
            'cleaned_text': cleaned_text,
            'text_length': len(cleaned_text),
            'medical_entities': medical_entities['entities'],
            'entity_metrics': {
                'total_entities': medical_entities['entity_count'],
                'semantic_density': medical_entities['entity_count'] / len(cleaned_text)
            }
        }
    
    def _normalize_unicode_characters(self, text: str) -> str:
        unicode_mappings = {
            '\u2019': "'",  # Right single quotation mark
            '\u2018': "'",  # Left single quotation mark
            '\u201c': '"',  # Left double quotation mark
            '\u201d': '"'   # Right double quotation mark
        }
        
        normalized_text = text
        for unicode_char, replacement in unicode_mappings.items():
            normalized_text = normalized_text.replace(unicode_char, replacement)
        
        return normalized_text

    def _extract_medical_entities(self, text: str) -> Dict[str, Any]:
        entity_types = [
            'medication', 'diagnosis', 'symptom', 
            'procedure', 'body_part'
        ]
        
        try:
            entities = self.model.predict_entities(
                text, 
                labels=entity_types,
                threshold=0.5
            )
            
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
        
        except Exception as extraction_anomaly:
            self.logger.error(f"Entity Extraction Disruption: {extraction_anomaly}")
            return {'entities': {}, 'entity_count': 0}

    def _serialize_batch(self, processed_batch: List[Dict[str, Any]]):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(
            self.output_path, 
            f"clinical_notes_batch_{timestamp}.json"
        )
        
        with open(output_filename, 'w', encoding='utf-8') as serialization_stream:
            json.dump(processed_batch, serialization_stream, indent=2)


load_dotenv()
connection = psycopg.connect(
    dbname=os.getenv('DB_NAME'),
    user=os.getenv('DB_USER'),
    password=os.getenv('DB_PASSWORD'),
    host=os.getenv('DB_HOST')
)

processor = ClinicalNotesProcessor(
    output_directory='processed_medical_notes'
)
processor.process_clinical_corpus(connection)