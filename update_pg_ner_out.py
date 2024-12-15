import os
import json
import pandas as pd
import psycopg
import logging
from tqdm import tqdm
from typing import Dict, Any, List
from dotenv import load_dotenv

class ClinicalNERMetadataUpdater:
    def __init__(self):
        load_dotenv()
        self.connection = psycopg.connect(
            dbname=os.getenv('DB_NAME'),
            user=os.getenv('DB_USER'),
            password=os.getenv('DB_PASSWORD'),
            host=os.getenv('DB_HOST')
        )
        
        # Computational Trace Management
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _augment_postgres_schema(self):
        alter_statements = [
            "ALTER TABLE clinical_notes ADD COLUMN IF NOT EXISTS text_length INTEGER",
            "ALTER TABLE clinical_notes ADD COLUMN IF NOT EXISTS total_entities INTEGER",
            "ALTER TABLE clinical_notes ADD COLUMN IF NOT EXISTS semantic_density FLOAT",
            
            # Entity Category Columns
            "ALTER TABLE clinical_notes ADD COLUMN IF NOT EXISTS medications TEXT[]",
            "ALTER TABLE clinical_notes ADD COLUMN IF NOT EXISTS diagnoses TEXT[]",
            "ALTER TABLE clinical_notes ADD COLUMN IF NOT EXISTS symptoms TEXT[]",
            "ALTER TABLE clinical_notes ADD COLUMN IF NOT EXISTS procedure TEXT[]",
            "ALTER TABLE clinical_notes ADD COLUMN IF NOT EXISTS body_parts TEXT[]"
        ]
        
        with self.connection.cursor() as cursor:
            for statement in alter_statements:
                try:
                    cursor.execute(statement)
                except Exception as schema_evolution_anomaly:
                    self.logger.warning(f"Schema Evolution Disruption: {schema_evolution_anomaly}")
            
            self.connection.commit()

    def process_ner_metadata(self, processed_notes_directory: str):
        self._augment_postgres_schema()
        
        # Computational Artifact Discovery
        json_files = [
            f for f in os.listdir(processed_notes_directory) 
            if f.endswith('.json')
        ]
        
        with tqdm(
            total=len(json_files), 
            desc="🔬 NER Metadata Synchronization",
            colour='green'
        ) as progress_bar:
            for json_file in json_files:
                file_path = os.path.join(processed_notes_directory, json_file)
                
                with open(file_path, 'r', encoding='utf-8') as file_stream:
                    batch_records = json.load(file_stream)
                
                self._update_postgres_records(batch_records)
                progress_bar.update(1)

    def _update_postgres_records(self, batch_records: List[Dict[str, Any]]):
        update_query = """
        UPDATE clinical_notes
        SET 
            text_length = %s,
            total_entities = %s,
            semantic_density = %s,
            medications = %s,
            diagnoses = %s,
            symptoms = %s,
            procedure = %s,
            body_parts = %s
        WHERE idx = %s
        """
        
        with self.connection.cursor() as cursor:
            for record in batch_records:
                # try:
                cursor.execute(update_query, (
                    record.get('text_length', None),
                    record['entity_metrics']['total_entities'],
                    record['entity_metrics']['semantic_density'],
                    record['medical_entities']['medication'],
                    record['medical_entities']['diagnosis'],
                    record['medical_entities']['symptom'],
                    record['medical_entities']['procedure'],
                    record['medical_entities']['body_part'],
                    record['id']
                ))
            
            self.connection.commit()
        
    def display_ner_metadata(self, limit: int = 1):
        query = """
        SELECT * FROM clinical_notes LIMIT %s
        """

        df = pd.read_sql_query(query, self.connection, params=(limit,))

        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_colwidth', None)

        print('\nClinical notes table along with NER data in Postgres\n')
        print(df.to_string(index=False))

        self.logger.info(f"Viewed {len(df)} clinical_notes table")

        return df



def main():
    metadata_updater = ClinicalNERMetadataUpdater()
    metadata_updater.process_ner_metadata('processed_medical_notes')
    metadata_updater.display_ner_metadata()

if __name__ == "__main__":
    main()