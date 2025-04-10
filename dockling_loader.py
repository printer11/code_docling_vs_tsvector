from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker
from docling_core.types.doc import DoclingDocument
from pathlib import Path
from typing import Dict, List, Optional
import logging
import asyncio
import os
import json
from datetime import datetime
from openai import AsyncOpenAI
from config.settings import get_settings
from vector_store import VectorStore
from utils.tokenizer import OpenAITokenizerWrapper
import logging.handlers
import traceback

# Helper functions
def extract_text_with_fallbacks(file_path: str) -> str:
    """
    Extract text from a PDF using multiple fallback methods.
    Tries PyPDF2 first, then pdfplumber if PyPDF2 doesn't yield good results.
    """
    original_text = ""
    
    # First try with PyPDF2
    try:
        import PyPDF2
        logger.info(f"Attempting PyPDF2 fallback for text extraction from {file_path}")
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            fallback_text = ""
            for page in reader.pages:
                page_text = page.extract_text() or ""
                if page_text:
                    fallback_text += page_text + "\n\n"
                    
        if len(fallback_text.strip()) > len(original_text.strip()):
            logger.info(f"Used PyPDF2 fallback for text extraction - got {len(fallback_text)} chars")
            return fallback_text
        
    except Exception as e:
        logger.error(f"PyPDF2 fallback text extraction failed: {e}")
    
    # If PyPDF2 fails or produces poor results, try pdfplumber
    try:
        import pdfplumber
        logger.info(f"Attempting pdfplumber fallback for text extraction from {file_path}")
        with pdfplumber.open(file_path) as pdf:
            plumber_text = ""
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text:
                    plumber_text += page_text + "\n\n"
                    
        if len(plumber_text.strip()) > len(original_text.strip()):
            logger.info(f"Used pdfplumber fallback for text extraction - got {len(plumber_text)} chars")
            return plumber_text
            
    except Exception as e:
        logger.error(f"pdfplumber fallback text extraction failed: {e}")
    
    logger.error("All fallback text extraction methods failed")
    return ""

# Konfiguracja ścieżek logowania
current_dir = Path(__file__).parent
project_dir = current_dir.parent
log_dir = project_dir / 'logs'
log_dir.mkdir(exist_ok=True)
log_file = log_dir / 'dockling.log'

# Konfiguracja loggera
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Handler dla pliku z rotacją
file_handler = logging.handlers.RotatingFileHandler(
    str(log_file),
    maxBytes=10485760,  # 10MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setLevel(logging.DEBUG)

# Handler dla konsoli
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Format logów
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Konfiguracja loggera
logger.handlers = []
logger.addHandler(file_handler)
logger.addHandler(console_handler)
logger.propagate = False

class DocklingMetadataEnhancer:
    def __init__(self, client: AsyncOpenAI):
        self.settings = get_settings()
        self.client = client
        logger.info("Initialized DocklingMetadataEnhancer")

    async def enhance_chunk_metadata(self, chunk, doc_type: str = None) -> Dict:
        """Rozszerza metadane chunka o dodatkowe informacje."""
        try:
            logger.debug(f"Enhancing metadata for chunk with doc_type: {doc_type}")
            
            # Podstawowe metadane z docklinga
            base_metadata = {
                "filename": getattr(chunk.meta.origin, 'filename', None),
                "page_numbers": await self._extract_page_numbers(chunk),
                "title": chunk.meta.headings[0] if (hasattr(chunk.meta, 'headings') and chunk.meta.headings) else None,
            }

            # Ekstrakcja sekcji
            section = await self._extract_section(chunk)
            if section:
                base_metadata["section"] = section

            # Analiza semantyczna chunka
            semantic_metadata = await self._analyze_chunk_content(chunk.text, doc_type)
            base_metadata.update(semantic_metadata)

            # Dodanie timestampu i typu dokumentu
            base_metadata.update({
                "processed_at": datetime.now().isoformat(),
                "doc_type": doc_type
            })

            logger.debug(f"Enhanced metadata: {json.dumps(base_metadata, indent=2)}")
            return base_metadata

        except Exception as e:
            logger.error(f"Error enhancing chunk metadata: {str(e)}")
            logger.error(traceback.format_exc())
            return {}

    async def _extract_section(self, chunk) -> Optional[str]:
        """Ekstrakcja nazwy sekcji z chunka."""
        try:
            # Próba ekstrakcji z meta.headings
            if hasattr(chunk.meta, 'headings') and chunk.meta.headings:
                return chunk.meta.headings[-1]

            # Próba ekstrakcji z doc_items
            if hasattr(chunk.meta, 'doc_items'):
                for item in chunk.meta.doc_items:
                    if hasattr(item, 'section'):
                        return item.section

            # Jeśli nie znaleziono sekcji, analizujemy tekst
            return await self._analyze_section_from_text(chunk.text)

        except Exception as e:
            logger.error(f"Error extracting section: {str(e)}")
            return None

    async def _analyze_section_from_text(self, text: str) -> Optional[str]:
        """Analiza tekstu w celu identyfikacji sekcji."""
        try:
            prompt = """Przeanalizuj poniższy fragment tekstu i określ nazwę sekcji dokumentu, do której należy.
            Zwróć TYLKO nazwę sekcji, bez dodatkowego tekstu. Jeśli nie można określić sekcji, zwróć NULL.
            
            Fragment tekstu:
            {text}
            """

            response = await self.client.chat.completions.create(
                model=self.settings.openai.default_model,
                messages=[
                    {"role": "system", "content": "Jesteś ekspertem w analizie struktury dokumentów."},
                    {"role": "user", "content": prompt.format(text=text[:500])}
                ],
                temperature=0.3
            )

            section = response.choices[0].message.content.strip()
            return None if section.lower() in ['null', 'none', 'brak'] else section

        except Exception as e:
            logger.error(f"Error analyzing section from text: {str(e)}")
            return None

    async def _analyze_chunk_content(self, text: str, doc_type: str = None) -> Dict:
        """Analiza semantyczna zawartości chunka."""
        try:
            analysis_prompt = f"""Przeanalizuj poniższy fragment tekstu i zwróć:
            1. Lista 3-5 słów kluczowych
            2. Główny temat fragmentu
            3. Typ zawartości (np. definicja, analiza, wniosek, rekomendacja)
            
            {f'Dokument został wcześniej sklasyfikowany jako: {doc_type}' if doc_type else ''}
            
            Zwróć wynik w formacie JSON z polami:
            {{
                "keywords": ["słowo1", "słowo2", ...],
                "main_topic": "główny temat",
                "content_type": "typ zawartości"
            }}
            
            Fragment tekstu:
            {text}
            """

            response = await self.client.chat.completions.create(
                model=self.settings.openai.default_model,
                messages=[
                    {"role": "system", "content": "Jesteś ekspertem w analizie tekstu."},
                    {"role": "user", "content": analysis_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.3
            )

            content = response.choices[0].message.content
            return json.loads(content)

        except Exception as e:
            logger.error(f"Error analyzing chunk content: {str(e)}")
            return {
                "keywords": [],
                "main_topic": None,
                "content_type": None
            }

    async def _extract_page_numbers(self, chunk) -> List[int]:
        """Ekstrakcja numerów stron z chunka."""
        try:
            if not hasattr(chunk.meta, 'doc_items'):
                return []
                
            page_numbers = []
            for item in chunk.meta.doc_items:
                for prov in item.prov:
                    if hasattr(prov, 'page_no'):
                        page_numbers.append(prov.page_no)
            
            return sorted(set(page_numbers))
            
        except Exception as e:
            logger.error(f"Error extracting page numbers: {str(e)}")
            return []

class DoclingLoader:
    def __init__(self):
        self.settings = get_settings()
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.vector_store = VectorStore()
        self.converter = DocumentConverter()
        self.tokenizer = OpenAITokenizerWrapper()
        self.metadata_enhancer = DocklingMetadataEnhancer(self.client)
        
        # Inicjalizacja katalogów
        self.processed_dir = Path("data/processed")
        self.pending_dir = Path("data/pending")
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.pending_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Initialized DoclingLoader")

    def list_pending_files(self) -> List[Path]:
        """Zwraca listę plików oczekujących na przetworzenie."""
        files = [
            f for f in self.pending_dir.glob("*")
            if f.is_file() and f.suffix.lower() in {'.pdf', '.docx', '.txt'}
        ]
        logger.info(f"Found {len(files)} pending files")
        return files

    async def process_file(self, file_path: Path) -> Dict:
        """Przetwarza pojedynczy plik."""
        try:
            logger.info(f"Starting processing file: {file_path}")
            
            # Konwersja dokumentu
            result = self.converter.convert(str(file_path))
            if not result.document:
                raise Exception(f"Failed to convert document: {file_path}")
            
            doc = result.document
            doc_type = await self._detect_document_type(doc)
            logger.info(f"Document type detected: {doc_type}")
            
            # Przetwarzanie chunków
            processed_chunks = await self._prepare_chunks(doc, doc_type)
            logger.info(f"Processed {len(processed_chunks)} chunks")

            # Przygotowanie metadanych dokumentu
            metadata = {
                "original_filename": file_path.name,
                "mime_type": result.input.format.value,
                "processed_date": datetime.now().isoformat(),
                "document_type": doc_type,
                "page_count": result.input.page_count,
                "total_chunks": len(processed_chunks)
            }

            # Przygotowanie danych do zapisu
            doc_data = {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "content_type": result.input.format.value,
                "metadata": metadata,
                "embedding": processed_chunks[0]["embedding"] if processed_chunks else [],
                "chunks": processed_chunks
            }

            # Zapis do bazy danych
            doc_id = await self.vector_store.store_document(doc_data)
            if not doc_id:
                raise Exception("Failed to store document in database")

            # Przeniesienie pliku do processed
            new_path = self.processed_dir / file_path.name
            file_path.rename(new_path)
            logger.info(f"Successfully processed and moved file: {file_path}")

            return {
                "status": "success",
                "document_id": doc_id,
                "metadata": metadata
            }

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "error": str(e)
            }

    async def _detect_document_type(self, doc: DoclingDocument) -> str:
        """Wykrywa typ dokumentu używając GPT."""
        try:
            text = doc.export_to_text()
            
            # More robust fallback when Dockling fails to extract text properly
            if "missing-text" in text or len(text.strip()) < 100:
                file_path = getattr(doc.meta.origin, 'filename', None)
                if file_path and file_path.lower().endswith('.pdf'):
                    fallback_text = extract_text_with_fallbacks(str(file_path))
                    if len(fallback_text.strip()) > len(text.strip()):
                        text = fallback_text
            
            prompt = """Przeanalizuj poniższy tekst i określ typ dokumentu. 
            Zwróć DOKŁADNIE jeden z następujących typów (zwróć dokładnie jedno słowo bez żadnego dodatkowego tekstu):
            OPINIA - jeśli to dokument zawierający opinie/rekomendacje
            ANALIZA - jeśli to opracowanie analityczne/raport
            REGULACJA - jeśli to dokument prawny/regulacyjny
            ETYKIETA - jeśli to etykieta produktu
            UNKNOWN - jeśli nie można jednoznacznie określić typu

            Tekst:
            {text}
            """

            response = await self.client.chat.completions.create(
                model=self.settings.openai.default_model,
                messages=[
                    {"role": "system", "content": "Jesteś ekspertem w klasyfikacji dokumentów."},
                    {"role": "user", "content": prompt.format(text=text[:2000])}
                ],
                temperature=0.3
            )

            doc_type = response.choices[0].message.content.strip().upper()
            logger.info(f"Detected document type: {doc_type}")
            return doc_type

        except Exception as e:
            logger.error(f"Error detecting document type: {str(e)}")
            return "UNKNOWN"

    async def _prepare_chunks(self, doc: DoclingDocument, doc_type: str) -> List[Dict]:
        """Przygotowuje chunki dokumentu z rozszerzonymi metadanymi."""
        try:
            logger.info("Creating HybridChunker")
            chunker = HybridChunker(
                tokenizer=self.tokenizer,
                max_tokens=8191,
                merge_peers=True
            )

            logger.info("Starting document chunking")
            raw_chunks = list(chunker.chunk(dl_doc=doc))
            logger.debug(f"Created {len(raw_chunks)} raw chunks")

            processed_chunks = []
            for i, chunk in enumerate(raw_chunks, 1):
                try:
                    logger.debug(f"Processing chunk {i}/{len(raw_chunks)}")
                    
                    # Generowanie embeddingu
                    chunk_embedding = await self._generate_embedding(chunk.text)
                    
                    # Rozszerzanie metadanych
                    chunk_metadata = await self.metadata_enhancer.enhance_chunk_metadata(
                        chunk=chunk,
                        doc_type=doc_type
                    )
                    
                    processed_chunks.append({
                        "text": chunk.text,
                        "metadata": chunk_metadata,
                        "embedding": chunk_embedding
                    })
                    
                    logger.debug(f"Successfully processed chunk {i}")
                    
                except Exception as chunk_error:
                    logger.error(f"Error processing chunk {i}: {str(chunk_error)}")
                    logger.error(traceback.format_exc())
                    continue

            logger.info(f"Successfully processed {len(processed_chunks)} chunks")
            return processed_chunks

        except Exception as e:
            logger.error(f"Error in _prepare_chunks: {str(e)}")
            logger.error(traceback.format_exc())
            raise

    async def _generate_embedding(self, text: str) -> List[float]:
        """Generuje embedding dla tekstu używając OpenAI."""
        try:
            response = await self.client.embeddings.create(
                model=self.settings.openai.embedding_model,
                input=text
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise

    async def search(self, query: str, limit: int = 5, 
                    metadata_filter: Optional[Dict] = None) -> List[Dict]:
        """Wyszukuje podobne dokumenty."""
        query_embedding = await self._generate_embedding(query)
        return await self.vector_store.search_similar(
            query_embedding, 
            limit=limit,
            metadata_filter=metadata_filter
        )