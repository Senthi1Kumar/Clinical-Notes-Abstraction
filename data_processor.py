import os
import logging
import pandas as pd  # noqa: F401
import psycopg # type: ignore
from psycopg import sql # type: ignore
from dotenv import load_dotenv
from datasets import load_dataset

class PostgresDataIngestor:
    def __init__(self):
        load_dotenv()

        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            filename='clinical_data_import.log'
        )
        self.logger = logging.getLogger(__name__)

    def _get_db_connection(self):
        """
        Establish DB connection using env vars

        Returns:
            pyscopg2 connection object
        """
        try:
            connection = psycopg.connect(
                dbname = os.getenv('DB_NAME'),
                user = os.getenv('DB_USER'),
                password = os.getenv('DB_PASSWORD'),
                host = os.getenv('DB_HOST')
            )
            self.logger.info("DB connection established successfully!")
            return connection
        except psycopg.Error as e:
            self.logger.error(f'Error connecting to PostgreSQL: {e}')
            raise

    def create_clinical_notes_table(self, connection):
        """
        Create table to store clinical notes if not exists

        Args:
            connection: DB connection object
        """
        with connection.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS clinical_notes (
                idx INTEGER,
                note TEXT,
                full_note TEXT,
                conversation TEXT,
                summary TEXT,
                PRIMARY KEY (idx)
                )
                """
            )
            connection.commit()
        
        self.logger.info("Clinical notes table created/verified")

    def bulk_insert_dataset(self, dataset, chunk_size=3000):
        """
        Bulk insert dataset into DB

        Args:
            dataset: HF dataset
            chunk_size: No. of rows per chunk
        """
        df = dataset.to_pandas()

        # Establish DB connection
        connection = self._get_db_connection() 

        try:
            self.create_clinical_notes_table(connection)

            # insert query
            insert_query = sql.SQL(
                """
                INSERT INTO clinical_notes
                (idx, note, full_note, conversation, summary)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (idx) DO NOTHING
                """
            )

            # chunked insertion
            with connection.cursor() as cursor:
                for i in range(0, len(df), chunk_size):
                    chunk = df.iloc[i:i+chunk_size]

                    # prepare data for insertion 
                    data_to_insert = chunk[['idx', 'note', 'full_note', 'conversation', 'summary']].values.tolist()

                    # execute batch insert
                    cursor.executemany(insert_query, data_to_insert)
                    connection.commit()

                    self.logger.info(f'Inserted chunk {i//chunk_size + 1}')

            self.logger.info("Dataset successfully uploaded")

        except Exception as e:
            connection.rollback()
            self.logger.error(f"Insertion failed: {e}")

        finally:
            connection.close()

def main():
    # load dataset
    ds = load_dataset("AGBonnet/augmented-clinical-notes")

    # initialize ingestor
    ingestor = PostgresDataIngestor()

    # upload dataset
    ingestor.bulk_insert_dataset(ds['train'])

if __name__ == "__main__":
    main()

