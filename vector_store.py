# vector_store.py
import logging
from typing import List, Dict, Optional
import psycopg2
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
import json
from datetime import datetime
from config.settings import get_settings

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.settings = get_settings()
        self.conn = None
        self._setup_db()

    def _setup_db(self):
        """Inicjalizacja połączenia i stworzenie tabel."""
        try:
            self.conn = psycopg2.connect(self.settings.database.service_url)
            register_vector(self.conn)
            
            with self.conn.cursor() as cur:
                # Tworzenie tabel jeśli nie istnieją
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS dockling_documents (
                        id SERIAL PRIMARY KEY,
                        file_name TEXT NOT NULL,
                        file_path TEXT NOT NULL,
                        content_type TEXT NOT NULL,
                        processed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        metadata JSONB DEFAULT '{}'::jsonb,
                        embedding vector(1536)
                    );
                    
                    CREATE TABLE IF NOT EXISTS dockling_chunks (
                        id SERIAL PRIMARY KEY,
                        document_id INTEGER REFERENCES dockling_documents(id),
                        chunk_text TEXT NOT NULL,
                        chunk_metadata JSONB DEFAULT '{}'::jsonb,
                        embedding vector(1536),
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    -- Indeksy dla przyspieszenia wyszukiwania
                    CREATE INDEX IF NOT EXISTS idx_dockling_docs_metadata ON dockling_documents USING GIN (metadata);
                    CREATE INDEX IF NOT EXISTS idx_dockling_chunks_metadata ON dockling_chunks USING GIN (chunk_metadata);
                """)
                self.conn.commit()
                
        except Exception as e:
            logger.error(f"Error setting up database: {e}")
            raise

    async def store_document(self, doc_data: Dict) -> Optional[int]:
        """Zapisuje dokument i jego chunki do bazy."""
        try:
            with self.conn.cursor() as cur:
                # Zapisz główny dokument
                cur.execute("""
                    INSERT INTO dockling_documents 
                    (file_name, file_path, content_type, metadata, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """, (
                    doc_data['file_name'],
                    str(doc_data['file_path']),
                    doc_data['content_type'],
                    json.dumps(doc_data['metadata']),
                    doc_data['embedding']
                ))
                
                doc_id = cur.fetchone()[0]
                
                # Zapisz chunki
                chunk_data = [(
                    doc_id,
                    chunk['text'],
                    json.dumps(chunk['metadata']),
                    chunk['embedding']
                ) for chunk in doc_data['chunks']]
                
                execute_values(cur, """
                    INSERT INTO dockling_chunks 
                    (document_id, chunk_text, chunk_metadata, embedding)
                    VALUES %s
                """, chunk_data)
                
                self.conn.commit()
                return doc_id
                
        except Exception as e:
            self.conn.rollback()
            logger.error(f"Error storing document: {e}")
            return None

    async def search_similar(
        self,
        query_embedding: List[float],
        limit: int = 5,
        metadata_filter: Optional[Dict] = None
    ) -> List[Dict]:
        """Wyszukiwanie podobnych chunków."""
        try:
            with self.conn.cursor() as cur:
                where_clause = "WHERE 1=1"
                params = [query_embedding]
                
                if metadata_filter:
                    for key, value in metadata_filter.items():
                        where_clause += f" AND chunk_metadata->>{key} = %s"
                        params.append(value)
                
                cur.execute(f"""
                    SELECT 
                        c.chunk_text,
                        c.chunk_metadata,
                        d.file_name,
                        d.metadata as doc_metadata,
                        1 - (c.embedding <=> %s::vector) as similarity
                    FROM dockling_chunks c
                    JOIN dockling_documents d ON c.document_id = d.id
                    {where_clause}
                    ORDER BY c.embedding <=> %s::vector
                    LIMIT {limit}
                """, params * 2)  # params * 2 bo używamy embedingu dwa razy
                
                results = []
                for row in cur.fetchall():
                    results.append({
                        'text': row[0],
                        'chunk_metadata': row[1],
                        'file_name': row[2],
                        'doc_metadata': row[3],
                        'similarity': row[4]
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
        
    async def get_document_chunks(self, document_id):
        """
        Pobiera wszystkie chunki dla danego dokumentu.
        
        Args:
            document_id: ID dokumentu
            
        Returns:
            Lista chunków dokumentu
        """
        try:
            with self.conn.cursor() as cur:
                # Pobierz chunki dla dokumentu
                cur.execute("""
                    SELECT 
                        id,
                        chunk_text as text,
                        chunk_metadata as metadata
                    FROM 
                        dockling_chunks
                    WHERE 
                        document_id = %s
                    ORDER BY 
                        id
                """, (document_id,))
                
                chunks = []
                for row in cur.fetchall():
                    metadata = json.loads(row[2]) if row[2] else {}
                    chunks.append({
                        "id": row[0],
                        "text": row[1],
                        "metadata": metadata
                    })
                return chunks
        except Exception as e:
            logger.error(f"Error retrieving document chunks: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []



    def __del__(self):
        """Zamknij połączenie przy usuwaniu obiektu."""
        if self.conn:
            self.conn.close()