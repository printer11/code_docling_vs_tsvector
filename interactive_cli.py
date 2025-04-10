import asyncio
import logging
from pathlib import Path
import os
import sys
import readline  # Dodaje historię i edycję linii
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
import time
from typing import List, Dict, Any, Optional
import traceback
import re
import json
import argparse
from rich.text import Text
from rich.markup import escape

# Nowe importy - dodaj je na początku pliku
from translation_utils import SimpleTranslator
from text_search import find_text_in_document
from ranking import rank_search_results
from two_layer_search import TwoLayerSearch

# Załaduj zmienne środowiskowe
load_dotenv()

# Konfiguracja logowania
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "dockling.log"

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Handler dla pliku z rotacją
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)

# Handler dla konsoli
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Format logów
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Dodanie handlerów do loggera
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Sprawdź czy używamy Dockling czy pgvector
USE_DOCKLING = os.getenv("USE_DOCKLING", "true").lower() == "true"

# Sprawdź flagi uruchomienia
CHECK_LOADER_ONLY = "--check-loader-only" in sys.argv
USE_ENHANCED = "--use-enhanced" in sys.argv or CHECK_LOADER_ONLY

# Konfiguracja wyszukiwania
if USE_DOCKLING:
    # Używamy ulepszonej lub standardowej wersji DoclingLoader
    if USE_ENHANCED:
        try:
            from dockling_loader_enhanced import EnhancedDoclingLoader as Searcher
            logger.info("Używam ulepszonej wersji DoclingLoader z deduplikacją")
            if CHECK_LOADER_ONLY:
                # Tylko sprawdzamy loader i kończymy
                sys.exit(0)
        except ImportError as e:
            logger.error(f"Nie można zaimportować EnhancedDoclingLoader: {e}")
            if CHECK_LOADER_ONLY:
                sys.exit(1)
            from dockling_loader import DoclingLoader as Searcher
            logger.info("Używam standardowej wersji DoclingLoader (enhanced niedostępny)")
    else:
        # Używamy standardowej wersji
        from dockling_loader import DoclingLoader as Searcher
        logger.info("Używam standardowej wersji DoclingLoader")
else:
    # Import klasy do wyszukiwania w PostgreSQL
    from postgres_vector_search import PostgresVectorSearch

    class Searcher:
        def __init__(self):
            self.db_url = os.getenv("TIMESCALE_SERVICE_URL")
            if not self.db_url:
                raise ValueError("TIMESCALE_SERVICE_URL not found in environment variables")
            self.pg_search = None
            
        async def initialize(self):
            self.pg_search = PostgresVectorSearch(self.db_url)
            await self.pg_search.initialize()
            
        async def search(self, query, limit=5, metadata_filter=None):
            if not self.pg_search:
                await self.initialize()
                
            # Generowanie embeddingu
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = await client.embeddings.create(
                model="text-embedding-3-small",
                input=query,
                encoding_format="float"
            )
            embedding = response.data[0].embedding
            
            # Wyszukiwanie z deduplikacją
            raw_results = await self.pg_search.search_similar(embedding, limit=limit*3)
            
            # Grupowanie wyników według dokumentów źródłowych
            grouped_results = {}
            for result in raw_results:
                title = result['title']
                
                if title not in grouped_results or result['similarity'] > grouped_results[title]['similarity']:
                    grouped_results[title] = result
            
            # Konwersja zgrupowanych wyników do listy i sortowanie według podobieństwa
            formatted_results = []
            for title, result in grouped_results.items():
                formatted_results.append({
                    'file_name': title,
                    'similarity': result['similarity'],
                    'doc_metadata': result['doc_metadata'],
                    'text': result['summary'] or result.get('text', '')
                })
            
            # Sortowanie według podobieństwa
            formatted_results.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Ograniczenie do żądanego limitu
            return formatted_results[:limit]
            
        async def close(self):
            if hasattr(self, 'pg_search') and self.pg_search:
                await self.pg_search.close()

# Konfiguracja Rich
console = Console()

async def generate_response(query, results, context=None):
    """Generuje odpowiedź na zapytanie użytkownika na podstawie znalezionych dokumentów."""
    from openai import AsyncOpenAI
    import asyncio
    import time
    
    logger.info(f"Generowanie odpowiedzi dla zapytania: '{query}'")
    
    client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    # Ograniczamy długość tekstu dla każdego dokumentu aby zmniejszyć zużycie tokenów
    max_content_length = 1000  # Maksymalna długość tekstu dla każdego dokumentu
    
    # Przygotowanie kontekstu z fragmentów
    fragments_context = "\n\n".join([
        f"Document: {result['file_name']}\n"
        f"Similarity: {result['similarity']:.2f}\n"
        f"Type: {result['doc_metadata'].get('document_type', 'Unknown') if isinstance(result['doc_metadata'], dict) else 'Unknown'}\n"
        f"Content: {result['text'][:max_content_length]}{'...' if len(result['text']) > max_content_length else ''}"
        for result in results
    ])
    
    # Dodaj dodatkowy kontekst, jeśli istnieje
    full_context = context + "\n\n" + fragments_context if context else fragments_context
    
    prompt = f"""
    Jako asystent ds. badań regulacyjnych w firmie zajmującej się prawem żywnościowym, odpowiedz na pytanie użytkownika.
    Bazuj wyłącznie na informacjach z dostarczonych dokumentów. Użytkownik pyta o wcześniejsze analizy i opinie.
    
    Użytkownik pyta: {query}
    
    Dokumenty, które mogą zawierać odpowiedź:
    {full_context}
    
    Jeśli nie możesz udzielić pełnej odpowiedzi na podstawie dostarczonych dokumentów, wyjaśnij co wiesz
    i jakich informacji brakuje. Podaj nazwy dokumentów, na których bazujesz.
    """
    
    # Dodanie obsługi timeout
    start_time = time.time()
    
    try:
        with console.status("[bold green]Wysyłanie zapytania do API OpenAI...[/]") as status:
            # Ustawienie timeout na 30 sekund
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "Jesteś asystentem ds. badań regulacyjnych w firmie zajmującej się prawem żywnościowym. Odpowiadasz na pytania dotyczące wcześniejszych analiz i opinii."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                ),
                timeout=30.0
            )
            
        elapsed_time = time.time() - start_time
        logger.info(f"Odpowiedź otrzymana w {elapsed_time:.2f} sekund")
        console.print(f"[green]Odpowiedź otrzymana w {elapsed_time:.2f} sekund[/]")
        return response.choices[0].message.content.strip()
        
    except asyncio.TimeoutError:
        logger.error("Upłynął limit czasu oczekiwania na odpowiedź z API OpenAI")
        console.print("[bold red]Upłynął limit czasu oczekiwania na odpowiedź z API OpenAI.[/]")
        return "Upłynął limit czasu oczekiwania na odpowiedź z API OpenAI. Spróbuj ponownie z krótszym zapytaniem lub mniejszą liczbą dokumentów."
    except Exception as e:
        logger.error(f"Błąd podczas generowania odpowiedzi: {str(e)}")
        console.print(f"[bold red]Wystąpił błąd: {str(e)}[/]")
        return f"Wystąpił błąd podczas generowania odpowiedzi: {str(e)}"

def highlight_text(text, search_term, color="yellow"):
    """Podkreśla szukany tekst w tekście, zachowując formatowanie rich."""
    if not search_term:
        return Text(text)
        
    escaped_text = escape(text)
    pattern = re.escape(search_term)
    highlighted = re.sub(
        f'({pattern})', 
        f'[bold {color}]\\1[/bold {color}]', 
        escaped_text, 
        flags=re.IGNORECASE
    )
    return highlighted

# Zmodyfikowana funkcja do wyszukiwania tekstu
async def find_in_documents(searcher, results, search_term, context_size=2, pre_context_size=500, post_context_size=1500):
    """
    Wyszukuje konkretny tekst w znalezionych dokumentach.
    
    Args:
        searcher: Obiekt wyszukiwania
        results: Lista wyników wyszukiwania
        search_term: Fraza do wyszukania
        context_size: Liczba chunków kontekstu przed i po
        pre_context_size: Liczba znaków kontekstu przed znalezionym dopasowaniem (domyślnie 500)
        post_context_size: Liczba znaków kontekstu po znalezionym dopasowaniu (domyślnie 1500)
    """
    from text_search import find_text_in_document, find_text_in_file
    from translation_utils import SimpleTranslator
    
    # Inicjalizacja tłumacza
    translator = SimpleTranslator()
    
    # Generuj warianty zapytania - zachowujemy funkcjonalność multilingual
    search_variants = translator.get_multilingual_variants(search_term)
    logger.debug(f"Szukam '{search_term}' z wariantami {search_variants} w {len(results)} dokumentach")
    
    found_chunks = []
    
    # Przeszukaj każdy dokument
    for result_idx, result in enumerate(results):
        doc_id = result.get('id') or result.get('document_id')
        file_name = result.get('file_name') or result.get('title')
        
        logger.debug(f"Dokument {result_idx+1}: {file_name}")
        
        # Spróbuj najpierw znaleźć oryginalny plik dla pełnego tekstu
        full_text = None
        original_file_path = None
        try:
            processed_dir = Path("data/processed")
            pending_dir = Path("data/pending")
            
            possible_paths = list(processed_dir.glob(f"*{file_name}*")) + list(pending_dir.glob(f"*{file_name}*"))
            
            if possible_paths:
                file_path = possible_paths[0]
                original_file_path = file_path
                logger.info(f"Znaleziono oryginalny plik dla wyszukiwania: {file_path}")
                
                # Użyj nowej funkcji find_text_in_file, która obsługuje różne typy plików
                for variant in search_variants:
                    file_matches = find_text_in_file(
                        str(file_path),
                        variant,
                        pre_context_size=pre_context_size,
                        post_context_size=post_context_size,
                        fuzzy_match=True,
                        aggressive_normalization=False  # Używamy mniej agresywnej normalizacji - zachowuje polskie znaki
                    )
                    
                    if file_matches:
                        logger.info(f"Znaleziono {len(file_matches)} wystąpień '{variant}' w pliku {file_path}")
                        
                        for match in file_matches:
                            # More comprehensive logging for each match
                            logger.debug(f"Match found: text='{match['match_text']}', position={match['position']}")
                            logger.debug(f"Search term was: '{variant}'")
                            logger.debug(f"Context (50 chars): '...{match['context'][max(0, match['position']-25):match['position']+25]}...'")
                            
                            found_chunks.append({
                                'result_idx': result_idx + 1,
                                'file_name': file_name,
                                'file_path': str(file_path),
                                'current_chunk_idx': 0,
                                'search_term': variant,
                                'chunks': [{
                                    'text': match['context'],
                                    'metadata': {
                                        'match_position': match['position'],
                                        'match_text': match['match_text']
                                    }
                                }],
                                'doc_metadata': result.get('doc_metadata', {})
                            })
                        
                        # Jeśli znaleziono dopasowania, przechodzimy do następnego dokumentu
                        continue
        except Exception as e:
            logger.warning(f"Nie udało się wczytać oryginalnego pliku: {e}")
            full_text = None
        
        # Jeśli nie znaleziono pliku lub nie znaleziono dopasowań w pliku, 
        # przeszukaj tekst z wyniku wyszukiwania semantycznego
        for variant in search_variants:
            # Użyj tekstu z wyniku wyszukiwania
            search_text = result.get('text', '')
            
            # Użyj ulepszonej funkcji find_text_in_document
            matches = find_text_in_document(
                {'text': search_text, 'document_id': doc_id, 'title': file_name},
                variant,
                pre_context_size=pre_context_size,
                post_context_size=post_context_size,
                fuzzy_match=True,
                aggressive_normalization=False  # Używamy mniej agresywnej normalizacji
            )
            
            if matches:
                logger.info(f"Znaleziono {len(matches)} wystąpień '{variant}' w tekście dokumentu {file_name}")
                
                for match in matches:
                    # More comprehensive logging for match
                    logger.debug(f"Match found: text='{match['match_text']}', position={match['position']}")
                    logger.debug(f"Search term was: '{variant}'")
                    logger.debug(f"Context (50 chars): '...{match['context'][max(0, match['position']-25):match['position']+25]}...'")
                    
                    found_chunks.append({
                        'result_idx': result_idx + 1,
                        'file_name': file_name,
                        'file_path': original_file_path,
                        'current_chunk_idx': 0,
                        'search_term': variant,
                        'chunks': [{
                            'text': match['context'],
                            'metadata': {
                                'match_position': match['position'],
                                'match_text': match['match_text']
                            }
                        }],
                        'doc_metadata': result.get('doc_metadata', {})
                    })
    
    return found_chunks

async def search_across_hits(searcher, query_results, search_term, pre_context_size=500, post_context_size=1500):
    """
    Funkcja przeszukuje wszystkie pliki znalezione w wyszukiwaniu semantycznym pod kątem określonej frazy.
    
    Args:
        searcher: Obiekt wyszukiwania
        query_results: Wyniki wyszukiwania semantycznego
        search_term: Fraza do wyszukania
        pre_context_size: Liczba znaków kontekstu przed znalezionym dopasowaniem
        post_context_size: Liczba znaków kontekstu po znalezionym dopasowaniu
        
    Returns:
        Lista wyników wyszukiwania z kontekstem
    """
    from text_search import find_text_in_file
    
    logger.info(f"Przeszukiwanie wszystkich wyników pod kątem frazy: '{search_term}'")
    results = []
    
    # Znajdź oryginalne pliki dla wszystkich wyników
    file_paths = []
    for result in query_results:
        file_name = result.get('file_name', '')
        if not file_name:
            continue
            
        # Sprawdź, czy plik istnieje w katalogach processed lub pending
        processed_dir = Path("data/processed")
        pending_dir = Path("data/pending")
        
        possible_paths = list(processed_dir.glob(f"*{file_name}*")) + list(pending_dir.glob(f"*{file_name}*"))
        
        if possible_paths:
            file_path = possible_paths[0]
            logger.debug(f"Znaleziono plik: {file_path}")
            file_paths.append((str(file_path), result))
    
    # Przeszukaj wszystkie znalezione pliki
    for file_path, result in file_paths:
        try:
            # Użyj funkcji find_text_in_file
            matches = find_text_in_file(
                file_path,
                search_term,
                pre_context_size=pre_context_size,
                post_context_size=post_context_size,
                fuzzy_match=True,
                aggressive_normalization=False  # Mniej agresywna normalizacja
            )
            
            if matches:
                logger.info(f"Znaleziono {len(matches)} wystąpień w pliku {file_path}")
                
                for match in matches:
                    similarity = result.get('similarity', 0)
                    
                    # More comprehensive logging for each match
                    logger.debug(f"Match found: text='{match['match_text']}', position={match['position']}")
                    logger.debug(f"Search term was: '{search_term}'")
                    logger.debug(f"Context (50 chars): '...{match['context'][max(0, match['position']-25):match['position']+25]}...'")
                    
                    results.append({
                        'file_path': file_path,
                        'file_name': os.path.basename(file_path),
                        'context': match['context'],
                        'match_position': match['position'],
                        'match_text': match['match_text'],
                        'similarity': similarity,
                        'doc_metadata': result.get('doc_metadata', {})
                    })
        except Exception as e:
            logger.error(f"Błąd podczas przeszukiwania pliku {file_path}: {e}", exc_info=True)
    
    # Sortuj wyniki według podobieństwa
    results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
    
    logger.info(f"Znaleziono łącznie {len(results)} wystąpień frazy '{search_term}' we wszystkich plikach")
    return results

# Nowa funkcja do zaawansowanego wyszukiwania
async def search_results_for_keyword(keyword, search_results, pre_context_size=500, post_context_size=1500):
    """
    Przeszukuje wyniki wyszukiwania pod kątem słowa kluczowego.
    
    Args:
        keyword: Słowo kluczowe do wyszukania
        search_results: Lista wyników wyszukiwania z wcześniejszego zapytania
        pre_context_size: Liczba znaków kontekstu przed znalezionym dopasowaniem (domyślnie 500)
        post_context_size: Liczba znaków kontekstu po znalezionym dopasowaniu (domyślnie 1500)
        
    Returns:
        Lista dokumentów zawierających słowo kluczowe wraz z fragmentami
    """
    logger.info(f"Przeszukiwanie wyników wyszukiwania dla słowa kluczowego: '{keyword}'")
    
    results = []
    
    for result in search_results:
        try:
            file_name = result['file_name']
            
            # Spróbuj najpierw odnaleźć oryginalny plik na dysku
            processed_dir = Path("data/processed")
            pending_dir = Path("data/pending")
            
            possible_paths = list(processed_dir.glob(f"*{file_name}*")) + list(pending_dir.glob(f"*{file_name}*"))
            
            if possible_paths:
                file_path = possible_paths[0]
                logger.info(f"Znaleziono oryginalny plik: {file_path}")
                
                # Konwertuj dokument na tekst
                from docling.document_converter import DocumentConverter
                converter = DocumentConverter()
                doc_result = converter.convert(str(file_path))
                
                # Pobierz tekst dokumentu
                full_text = doc_result.document.export_to_text()
                
                # Sprawdź czy zawiera słowo kluczowe (niewrażliwe na wielkość liter)
                if keyword.lower() in full_text.lower():
                    logger.info(f"Znaleziono dopasowanie w pliku: {file_path}")
                    
                    # Użyj zmodyfikowanej funkcji find_text_in_document dla lepszego kontekstu
                    matches = find_text_in_document(
                        {'text': full_text, 'document_id': file_path.name, 'title': file_name},
                        keyword,
                        pre_context_size=pre_context_size,
                        post_context_size=post_context_size,
                        fuzzy_match=True,
                        aggressive_normalization=True
                    )
                    
                    if matches:
                        for match in matches:
                            # More comprehensive logging for each match
                            logger.debug(f"Match found: text='{match['match_text']}', position={match['position']}")
                            logger.debug(f"Search term was: '{keyword}'")
                            logger.debug(f"Context (50 chars): '...{match['context'][max(0, match['position']-25):match['position']+25]}...'")
                            
                            results.append({
                                'file_path': str(file_path),
                                'file_name': file_name,
                                'context': match['context'],
                                'match_position': match['position'],
                                'match_text': match['match_text'],
                                'full_text': full_text,
                                'similarity': result.get('similarity', 0)
                            })
                    else:
                        # Fallback - jeśli find_text_in_document nie zwrócił wyników
                        index = full_text.lower().find(keyword.lower())
                        start = max(0, index - pre_context_size)
                        end = min(len(full_text), index + len(keyword) + post_context_size)
                        context = full_text[start:end]
                        
                        results.append({
                            'file_path': str(file_path),
                            'file_name': file_name,
                            'context': context,
                            'match_position': index,
                            'match_text': full_text[index:index+len(keyword)],
                            'full_text': full_text,
                            'similarity': result.get('similarity', 0)
                        })
            else:
                # Jeśli nie znaleziono pliku, sprawdź tekst w wynikach wyszukiwania
                text = result.get('text', '')
                # Sprawdź czy zawiera słowo kluczowe z użyciem agresywnej normalizacji
                from text_search import normalize_text
                text_norm = normalize_text(text, aggressive=True)
                keyword_norm = normalize_text(keyword, aggressive=True)
                if keyword_norm in text_norm:
                    logger.info(f"Znaleziono dopasowanie w tekście wynikowym dla: {file_name}")
                    
                    # Użyj zmodyfikowanej funkcji find_text_in_document dla lepszego kontekstu
                    matches = find_text_in_document(
                        {'text': text, 'document_id': result.get('id', ''), 'title': file_name},
                        keyword,
                        pre_context_size=pre_context_size,
                        post_context_size=post_context_size,
                        fuzzy_match=True,
                        aggressive_normalization=True
                    )
                    
                    if matches:
                        for match in matches:
                            results.append({
                                'file_path': None,
                                'file_name': file_name,
                                'context': match['context'],
                                'match_position': match['position'],
                                'match_text': match['match_text'],
                                'full_text': text,
                                'similarity': result.get('similarity', 0)
                            })
                    else:
                        # Fallback - jeśli find_text_in_document nie zwrócił wyników
                        index = text.lower().find(keyword.lower())
                        start = max(0, index - pre_context_size)
                        end = min(len(text), index + len(keyword) + post_context_size)
                        context = text[start:end]
                        
                        results.append({
                            'file_path': None,
                            'file_name': file_name,
                            'context': context,
                            'match_position': index,
                            'match_text': text[index:index+len(keyword)],
                            'full_text': text,
                            'similarity': result.get('similarity', 0)
                        })
                
        except Exception as e:
            logger.error(f"Błąd podczas przetwarzania wyniku {result.get('file_name')}: {str(e)}", exc_info=True)
    
    # Sortuj wyniki według podobieństwa
    results.sort(key=lambda x: x.get('similarity', 0), reverse=True)
    
    logger.info(f"Znaleziono {len(results)} wyników zawierających słowo kluczowe '{keyword}'")
    return results

async def search_all_files_for_keyword(keyword, base_dirs=None):
    """
    Przeszukuje wszystkie dokumenty w określonych katalogach pod kątem słowa kluczowego.
    
    Args:
        keyword: Słowo kluczowe do wyszukania
        base_dirs: Lista katalogów do przeszukania. Domyślnie data/processed i data/pending
        
    Returns:
        Lista dokumentów zawierających słowo kluczowe wraz z fragmentami
    """
    if base_dirs is None:
        base_dirs = [Path("data/processed"), Path("data/pending")]
    
    logger.info(f"Przeszukiwanie wszystkich plików dla słowa kluczowego: '{keyword}'")
    
    results = []
    
    # Normalizacja słowa kluczowego
    from text_search import normalize_text
    keyword_norm = normalize_text(keyword, aggressive=True)
    
    for base_dir in base_dirs:
        if not base_dir.exists():
            logger.warning(f"Katalog {base_dir} nie istnieje")
            continue
            
        # Znajdź wszystkie pliki PDF i DOCX
        pdf_files = list(base_dir.glob("*.pdf"))
        docx_files = list(base_dir.glob("*.docx"))
        
        all_files = pdf_files + docx_files
        logger.info(f"Znaleziono {len(all_files)} plików w katalogu {base_dir}")
        
        for file_path in all_files:
            try:
                # Wczytaj dokument
                logger.debug(f"Analizowanie pliku: {file_path}")
                
                # Konwertuj dokument na tekst
                from docling.document_converter import DocumentConverter
                converter = DocumentConverter()
                doc_result = converter.convert(str(file_path))
                
                # Pobierz tekst dokumentu
                full_text = doc_result.document.export_to_text()
                
                # Sprawdź czy zawiera słowo kluczowe z użyciem agresywnej normalizacji
                text_norm = normalize_text(full_text, aggressive=True)
                if keyword_norm in text_norm:
                    logger.info(f"Znaleziono dopasowanie w pliku: {file_path}")
                    
                    # Użyj funkcji find_text_in_document z agresywną normalizacją
                    matches = find_text_in_document(
                        {'text': full_text, 'document_id': file_path.name, 'title': file_path.name},
                        keyword,
                        pre_context_size=500, 
                        post_context_size=1500,
                        fuzzy_match=True,
                        aggressive_normalization=True
                    )
                    
                    if matches:
                        for match in matches:
                            results.append({
                                'file_path': str(file_path),
                                'file_name': file_path.name,
                                'context': match['context'],
                                'match_position': match['position'],
                                'match_text': match['match_text'],
                                'full_text': full_text
                            })
                    else:
                        # Fallback - jeśli find_text_in_document nie zwrócił wyników
                        # Znajdź kontekst dla słowa kluczowego
                        index = text_norm.find(keyword_norm)
                        if index >= 0:
                            # Przybliżona pozycja w oryginalnym tekście
                            approx_index = min(index, len(full_text)-1)
                            start = max(0, approx_index - 500)
                            end = min(len(full_text), approx_index + 1500)
                            context = full_text[start:end]
                            
                            results.append({
                                'file_path': str(file_path),
                                'file_name': file_path.name,
                                'context': context,
                                'full_text': full_text
                            })
                    
            except Exception as e:
                logger.error(f"Błąd podczas przetwarzania pliku {file_path}: {str(e)}", exc_info=True)
    
    logger.info(f"Znaleziono {len(results)} plików zawierających słowo kluczowe '{keyword}'")
    return results

async def advanced_search(searcher, query, limit=5, context_size=100, 
                   pre_context_size=500, post_context_size=1500):
    """
    Wykonuje zaawansowane wyszukiwanie z wykorzystaniem dwuwarstwowego podejścia.
    
    Args:
        searcher: Obiekt wyszukiwania (Dockling lub PostgreSQL)
        query: Zapytanie użytkownika
        limit: Maksymalna liczba dokumentów do zwrócenia
        context_size: Rozmiar kontekstu w znakach (symetryczny, gdy nie używane są pre/post_context_size)
        pre_context_size: Liczba znaków kontekstu przed dopasowaniem (domyślnie 500)
        post_context_size: Liczba znaków kontekstu po dopasowaniu (domyślnie 1500)
    
    Returns:
        Dict zawierający wyniki wyszukiwania
    """
    logger.info(f"Zaawansowane wyszukiwanie dla zapytania: '{query}' z limitem {limit}")
    
    # Inicjalizacja tłumacza
    translator = SimpleTranslator()
    
    # Wygeneruj warianty zapytania
    query_variants = translator.get_multilingual_variants(query)
    logger.debug(f"Warianty zapytania: {query_variants}")
    
    # Wyszukiwanie semantyczne dla wszystkich wariantów
    all_results = []
    for variant in query_variants:
        variant_results = await searcher.search(variant, limit=limit*2)
        all_results.extend(variant_results)
    
    # Deduplikacja wyników
    seen_ids = set()
    unique_results = []
    for result in all_results:
        result_id = result.get('id') or result.get('document_id')
        if result_id and result_id not in seen_ids:
            seen_ids.add(result_id)
            unique_results.append(result)
    
    # Ogranicz do limitu, sortując według podobieństwa
    top_results = sorted(unique_results, key=lambda x: x.get('similarity', 0), reverse=True)[:limit]
    
    # Znajdź konkretne wzmianki w dokumentach z asymetrycznym kontekstem
    all_matches = []
    for doc in top_results:
        for variant in query_variants:
            # Użyj funkcji find_text_in_document z modułu text_search z asymetrycznym kontekstem
            matches = find_text_in_document(
                doc, 
                variant, 
                context_size=context_size,
                pre_context_size=pre_context_size,
                post_context_size=post_context_size
            )
            
            for match in matches:
                match['query_variant'] = variant
                match['document'] = doc
                
            all_matches.extend(matches)
    
    # Oceń i uszereguj wyniki
    ranked_matches = rank_search_results(all_matches, query)
    
    return {
        'query': query,
        'variants': query_variants,
        'documents': top_results,
        'matches': ranked_matches,
        'num_documents': len(top_results),
        'num_matches': len(ranked_matches)
    }

def display_advanced_results(console, search_results):
    """Wyświetla wyniki zaawansowanego wyszukiwania."""
    if not search_results or not search_results.get('matches'):
        console.print("[yellow]Nie znaleziono szukanego tekstu w dokumentach.[/]")
        return
        
    # Podsumowanie wyników
    console.print(Panel(
        f"Znaleziono {search_results['num_matches']} dopasowań w {search_results['num_documents']} dokumentach",
        title="Wyniki wyszukiwania",
        style="green"
    ))
    
    # Jeśli były warianty zapytania, pokaż je
    if len(search_results.get('variants', [])) > 1:
        variants = ", ".join(search_results['variants'])
        console.print(f"[blue]Użyte warianty wyszukiwania:[/] {variants}\n")
    
    # Wyświetl dopasowania
    for i, match in enumerate(search_results['matches'][:10], 1):  # Pokaż top 10
        document_title = match.get('document_title', '')
        context = match.get('context', '')
        match_text = match.get('match_text', '')
        rank = match.get('rank', 0)
        variant = match.get('query_variant', '')
        
        # Nagłówek dopasowania
        header = f"{i}. {document_title}"
        if variant != search_results['query']:
            header += f" [blue](znaleziono dla: '{variant}')[/]"
        
        console.print(f"\n[bold cyan]{header}[/] [yellow](Trafność: {rank:.2f})[/]")
        
        # Highlight kontekstu
        if context and match_text:
            context_parts = context.split(match_text)
            if len(context_parts) > 1:
                highlighted = f"{context_parts[0]}[bold on yellow]{match_text}[/bold on yellow]{context_parts[1]}"
                console.print(highlighted)
            else:
                console.print(context)
        else:
            console.print(context)
        
        console.print("---")

def create_results_table(results):
    """Tworzy tabelę z wynikami wyszukiwania."""
    table = Table(title="Znalezione dokumenty")
    
    table.add_column("Nr", style="cyan")
    table.add_column("Dokument", style="green")
    table.add_column("Podobieństwo", style="yellow")
    table.add_column("Typ", style="magenta")
    
    for i, result in enumerate(results, 1):
        doc_type = result['doc_metadata'].get('document_type', 'Unknown') if isinstance(result['doc_metadata'], dict) else 'Unknown'
        table.add_row(
            str(i),
            result['file_name'],
            f"{result['similarity']:.2f}",
            doc_type
        )
    
    return table
def display_found_chunks(console, found_chunks):
    """Wyświetla znalezione fragmenty z kontekstem i podświetleniem."""
    if not found_chunks:
        console.print("[yellow]Nie znaleziono szukanego tekstu w dokumentach.[/]")
        return
        
    for find_result in found_chunks:
        file_name = find_result['file_name']
        result_idx = find_result['result_idx']
        doc_type = find_result.get('doc_metadata', {}).get('document_type', 'Unknown')
        search_term = find_result['search_term']
        file_path = find_result.get('file_path', 'Unknown')
        
        console.print(f"\n[bold cyan]Dokument {result_idx}:[/] [green]{file_name}[/] [magenta](Typ: {doc_type})[/]")
        if file_path and file_path != 'Unknown':
            console.print(f"[dim]Ścieżka: {file_path}[/]")
        
        for i, chunk in enumerate(find_result['chunks']):
            chunk_text = chunk.get('text', '')
            
            # Pobierz informacje o dopasowaniu
            match_position = chunk.get('metadata', {}).get('match_position', -1)
            match_text = chunk.get('metadata', {}).get('match_text', search_term)
            
            # Wyświetl informacje o fragmencie
            console.print(f"[bold white on blue]>>> ZNALEZIONO ({i+1}/{len(find_result['chunks'])}) <<<[/]")
            
            # Podświetl fragment z kontekstem
            if match_position >= 0 and match_text:
                # Podziel tekst na fragmenty przed i po dopasowaniu dla podświetlenia
                before_match = chunk_text[:match_position]
                after_match = chunk_text[match_position + len(match_text):]
                
                # Wyświetl z podświetleniem
                console.print(f"{before_match}[bold on yellow]{match_text}[/bold on yellow]{after_match}")
            else:
                # Jeśli nie mamy dokładnej pozycji, próbujemy podświetlić termin wyszukiwania
                highlighted_text = highlight_text(chunk_text, search_term)
                console.print(highlighted_text)
            
            console.print("---")

        
        for i, chunk in enumerate(find_result['chunks']):
            chunk_text = chunk.get('text', '')
            is_current = i == (find_result['current_chunk_idx'] - find_result['chunks'][0]['metadata'].get('index', 0))
            
            # Informacje o bieżącym fragmencie
            chunk_info = []
            if 'page_numbers' in chunk.get('metadata', {}):
                pages = chunk['metadata']['page_numbers']
                chunk_info.append(f"Strona: {', '.join(map(str, pages))}")
            
            if 'section' in chunk.get('metadata', {}):
                chunk_info.append(f"Sekcja: {chunk['metadata']['section']}")
                
            chunk_info_str = " | ".join(chunk_info) if chunk_info else ""
            
            # Wyświetl informacje o fragmencie
            if is_current:
                console.print(f"[bold white on blue]>>> ZNALEZIONO ({chunk_info_str}) <<<[/]")
                highlighted_text = highlight_text(chunk_text, search_term)
                console.print(highlighted_text)
            else:
                console.print(f"[dim]{chunk_info_str}[/]")
                console.print(f"[dim]{chunk_text}[/]")
            
            console.print("---")
            

async def main():
    """Główna funkcja interaktywnego interfejsu."""
    console.print(Panel.fit("Document Search - CLI", style="bold green"))
    console.print("Wpisz 'exit' lub 'quit' aby zakończyć. 'help' wyświetli pomoc.\n")
    
    # Inicjalizacja wyszukiwania
    searcher = Searcher()
    
    # Globalne ustawienia wyszukiwania
    limit = 5
    context = None
    query_history = []
    
    try:
        while True:
            try:
                query = console.input("[bold blue]Zapytanie:[/] ")
                
                if not query.strip():
                    continue
                    
                if query.lower() in ('exit', 'quit'):
                    break
                    
                if query.lower() == 'help':
                    console.print(Panel("""
                    Dostępne komendy:
                    - [bold]exit[/] / [bold]quit[/] - zakończenie programu
                    - [bold]help[/] - wyświetlenie pomocy
                    - [bold]limit N[/] - zmiana liczby wyświetlanych wyników (np. limit 3)
                    - [bold]context[/] - ustawienie stałego kontekstu dla wszystkich zapytań
                    - [bold]show N[/] - pokazanie pełnej treści dokumentu o numerze N
                    - [bold]find tekst[/] - wyszukuje konkretny tekst w znalezionych dokumentach i pokazuje fragmenty z kontekstem
                    - [bold]search tekst[/] - zaawansowane wyszukiwanie z obsługą wariantów i tłumaczeń
                    - [bold]search-across tekst[/] - przeszukuje wszystkie znalezione dokumenty dla dokładnego ciągu znaków
                    - [bold]search-all tekst[/] - przeszukuje tylko aktualne wyniki wyszukiwania pod kątem słowa kluczowego
                    - [bold]search-every tekst[/] - przeszukuje wszystkie pliki w katalogach pod kątem słowa kluczowego
                    - [bold]history[/] - pokazanie historii zapytań
                    - [bold]analyze N[/] - analiza dokumentu o numerze N
                    - [bold]analyze all[/] - analiza wszystkich dokumentów
                    
                    Przykładowe zapytania:
                    - "Jakie są główne wnioski EFSA dotyczące glukozaminy?"
                    - "Czy opisywaliśmy składnik chaga?"
                    - "Pokaż informacje o kakao w proszku"
                    - "find E211" - wyszuka wszystkie wystąpienia E211 w znalezionych dokumentach
                    - "search sorbinian potasu" - zaawansowane wyszukiwanie z tłumaczeniem i wariantami
                    - "search-across Crocus sativus" - znajdzie dokładne wystąpienia "Crocus sativus" w dokumentach
                    - "search-all Crocus sativus" - szuka w aktualnych wynikach wyszukiwania
                    - "search-every Crocus sativus" - szuka we wszystkich plikach w katalogach
                    """, title="Pomoc"))
                    continue
                    
                if query.lower().startswith('limit '):
                    try:
                        limit = int(query.split()[1])
                        logger.info(f"Limit wyników zmieniony na: {limit}")
                        console.print(f"[green]Limit wyników zmieniony na: {limit}[/]")
                    except (IndexError, ValueError):
                        console.print("[red]Błąd: Podaj prawidłową liczbę[/]")
                    continue
                    
                if query.lower() == 'context':
                    console.print("[yellow]Podaj kontekst (Enter pusty aby anulować):[/]")
                    context_input = console.input()
                    if context_input:
                        context = context_input
                        logger.info(f"Ustawiono kontekst: {context}")
                        console.print("[green]Kontekst ustawiony[/]")
                    continue
                    
                if query.lower() == 'history':
                    if query_history:
                        history_table = Table(title="Historia zapytań")
                        history_table.add_column("Nr", style="cyan")
                        history_table.add_column("Zapytanie", style="green")
                        
                        for i, hist_query in enumerate(query_history, 1):
                            history_table.add_row(str(i), hist_query)
                        
                        console.print(history_table)
                    else:
                        console.print("[yellow]Historia zapytań jest pusta[/]")
                    continue
                
                # Nowe polecenie: zaawansowane wyszukiwanie
                if query.lower().startswith('search '):
                    search_term = query[7:].strip()
                    if not search_term:
                        console.print("[red]Błąd: Podaj tekst do wyszukania[/]")
                        continue
                    
                    logger.info(f"Zaawansowane wyszukiwanie dla zapytania: '{search_term}'")
                    console.print(f"[bold]Zaawansowane wyszukiwanie '{search_term}'...[/]")
                    
                    with console.status("[bold green]Wyszukiwanie...[/]"):
                        search_results = await advanced_search(
                            searcher, 
                            search_term,
                            limit=limit,
                            context_size=150,
                            pre_context_size=500,
                            post_context_size=1500
                        )
                    
                    # Wyświetl wyniki
                    display_advanced_results(console, search_results)
                    continue
                # search-across zostało przeniesione do pętli poleceń poniżej, gdzie action jest zdefiniowane
                
                # Zapisz zapytanie w historii
                query_history.append(query)
                logger.info(f"Nowe zapytanie: '{query}'")
                
                with console.status("[bold green]Wyszukiwanie...[/]"):
                    # Wyszukiwanie dokumentów
                    logger.info(f"Wyszukiwanie dokumentów dla zapytania: '{query}' z limitem {limit}")
                    results = await searcher.search(query, limit=limit)
                    logger.info(f"Znaleziono {len(results)} dokumentów")
                    
                if not results:
                    logger.warning("Nie znaleziono pasujących dokumentów")
                    console.print("[yellow]Nie znaleziono pasujących dokumentów[/]")
                    continue
                
                # Wyświetlenie wyników
                table = create_results_table(results)
                console.print(table)
                
                # Czekamy na decyzję użytkownika
                while True:
                    action = console.input("[bold blue]Polecenie ([bold]show N[/] / [bold]find tekst[/] / [bold]search-across tekst[/] / [bold]search-all tekst[/] / [bold]search-every tekst[/] / [bold]analyze N[/] / [bold]analyze all[/] / [bold]Enter[/] aby kontynuować):[/] ")
                    
                    if not action:
                        break  # Kontynuuj z nowym zapytaniem
                        
                    if action.lower().startswith('search-across '):
                        try:
                            search_term = action[13:].strip()
                            if not search_term:
                                console.print("[red]Błąd: Podaj tekst do wyszukania[/]")
                                continue
                                
                            logger.info(f"Wyszukiwanie frazy '{search_term}' we wszystkich znalezionych dokumentach")
                            console.print(f"[bold]Przeszukiwanie wszystkich plików dla frazy '{search_term}'...[/]")
                            
                            with console.status("[bold green]Przeszukiwanie wszystkich plików...[/]"):
                                cross_results = await search_across_hits(
                                    searcher, 
                                    results, 
                                    search_term,
                                    pre_context_size=500,
                                    post_context_size=1500
                                )
                            
                            if not cross_results:
                                console.print("[yellow]Nie znaleziono wystąpień frazy w żadnym pliku.[/]")
                                continue
                            
                            # Wyświetl wyniki
                            console.print(Panel(f"Znaleziono {len(cross_results)} wystąpień frazy '{search_term}' we wszystkich plikach", style="green"))
                            
                            for i, result in enumerate(cross_results, 1):
                                file_name = result['file_name']
                                context = result['context']
                                match_text = result['match_text']
                                match_position = result['match_position']
                                
                                # Podziel tekst, aby podświetlić znalezioną frazę
                                before_match = context[:match_position - context.find(context.strip())]
                                after_match = context[match_position + len(match_text):]
                                
                                console.print(f"\n[bold cyan]{i}. {file_name}[/] [yellow](Trafność: {result['similarity']:.2f})[/]")
                                
                                # Wyświetl kontekst z podświetleniem
                                console.print(f"{before_match}[bold on yellow]{match_text}[/bold on yellow]{after_match}")
                                console.print("---")
                        except Exception as e:
                            logger.error(f"Błąd podczas przeszukiwania plików: {str(e)}", exc_info=True)
                            console.print(f"[red]Wystąpił błąd podczas przeszukiwania: {str(e)}[/]")
                    
                    elif action.lower().startswith('search-all '):
                        try:
                            search_term = action[11:].strip()
                            if not search_term:
                                console.print("[red]Błąd: Podaj tekst do wyszukania[/]")
                                continue
                                
                            logger.info(f"Wyszukiwanie dla słowa kluczowego: '{search_term}'")
                            
                            # Przeszukaj tylko aktualne wyniki wyszukiwania
                            console.print(f"[bold]Przeszukiwanie aktualnych wyników wyszukiwania dla '{search_term}'...[/]")
                            
                            with console.status("[bold green]Przeszukiwanie aktualnych wyników wyszukiwania...[/]"):
                                # Przeszukaj tylko aktualne wyniki
                                file_results = await search_results_for_keyword(search_term, results)
                            
                            if not file_results:
                                console.print("[yellow]Nie znaleziono dopasowań w aktualnych wynikach.[/]")
                                continue
                        except Exception as e:
                            logger.error(f"Błąd podczas przeszukiwania plików: {str(e)}", exc_info=True)
                            console.print(f"[red]Wystąpił błąd podczas przeszukiwania: {str(e)}[/]")
                                
                    elif action.lower().startswith('search-every '):
                        try:
                            search_term = action[13:].strip()
                            if not search_term:
                                console.print("[red]Błąd: Podaj tekst do wyszukania[/]")
                                continue
                                
                            logger.info(f"Wyszukiwanie dla słowa kluczowego: '{search_term}' we wszystkich plikach")
                            console.print(f"[bold]Przeszukiwanie wszystkich plików dla '{search_term}'...[/]")
                            
                            with console.status("[bold green]Przeszukiwanie wszystkich plików w katalogach data/processed i data/pending...[/]"):
                                # Przeszukaj wszystkie pliki
                                file_results = await search_all_files_for_keyword(search_term)
                            
                            if not file_results:
                                console.print("[yellow]Nie znaleziono plików zawierających szukany tekst.[/]")
                                continue
                            
                            # Posortuj wyniki - najpierw sprawdź czy fraza jest w nazwie pliku
                            from text_search import normalize_text
                            for result in file_results:
                                file_name_norm = normalize_text(result['file_name'].lower(), aggressive=True)
                                search_term_norm = normalize_text(search_term.lower(), aggressive=True)
                                result['exact_name_match'] = search_term_norm in file_name_norm
                            
                            # Sortuj: najpierw pliki z frazą w nazwie, potem reszta
                            file_results = sorted(file_results, key=lambda x: (not x.get('exact_name_match', False)))
                            
                            # Wyświetl wyniki
                            console.print(Panel(f"Znaleziono {len(file_results)} plików zawierających tekst '{search_term}'", style="green"))
                            
                            for i, result in enumerate(file_results, 1):
                                file_name = result['file_name']
                                context = result['context']
                                
                                # Highlight kontekstu
                                context_parts = context.lower().split(search_term.lower())
                                if len(context_parts) > 1:
                                    # Rekonstruuj kontekst z podświetleniem
                                    highlighted_context = ""
                                    for i, part in enumerate(context_parts):
                                        if i > 0:
                                            # Znajdź oryginalny fragment (z zachowaniem wielkości liter)
                                            idx = context.lower().find(part, len(highlighted_context))
                                            if idx > 0:
                                                original_term = context[idx-len(search_term):idx]
                                                highlighted_context += f"[bold on yellow]{original_term}[/bold on yellow]"
                                        
                                        if part:
                                            highlighted_context += part
                                else:
                                    highlighted_context = context
                                
                                # Wyświetl informacje o pliku z podświetleniem
                                console.print(f"\n[bold cyan]{i}. {file_name}[/]")
                                console.print(highlighted_context)
                                console.print("---")
                            
                            # Pytanie o analizę konkretnego pliku
                            analyze_input = console.input("[bold yellow]Analizować konkretny plik? Podaj numer (Enter aby pominąć):[/] ")
                            if analyze_input.strip() and analyze_input.isdigit():
                                file_idx = int(analyze_input) - 1
                                if 0 <= file_idx < len(file_results):
                                    selected_file = file_results[file_idx]
                                    file_path = selected_file['file_path']
                                    file_name = selected_file['file_name']
                                    full_text = selected_file['full_text']
                                    
                                    logger.info(f"Analiza pliku: {file_path}")
                                    
                                    # Zapytanie o analizę OpenAI
                                    console.print(f"[yellow]Wyszukiwanie fragmentów związanych z '{search_term}' w dokumencie. Ta operacja zużywa tokeny OpenAI.[/]")
                                    proceed = console.input("[bold yellow]Czy kontynuować? (y/n):[/] ").lower()
                                    
                                    if proceed in ('y', 'yes', 't', 'tak'):
                                        # Pobierz ostatnie zapytanie użytkownika
                                        last_query = query_history[-1] if query_history else search_term
                                        
                                        # Analizuj dokument z OpenAI
                                        from openai import AsyncOpenAI
                                        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                                        
                                        # Prompt do dogłębnej analizy dokumentów o żywności
                                        system_prompt = """Jesteś precyzyjnym narzędziem do wyszukiwania i analizy informacji w dokumentach prawnych i naukowych dotyczących żywności.
                                        
                                        Przeprowadź analizę dokumentu i udziel szczegółowej odpowiedzi w następujących sekcjach:
                                        
                                        # 1. ODPOWIEDŹ NA ZAPYTANIE
                                        Znajdź 2-3 DOKŁADNE CYTATY z dokumentu, które najlepiej odpowiadają na zapytanie użytkownika, wraz z kontekstem.
                                        
                                        ## Fragment 1
                                        ```
                                        [dokładny cytat z dokumentu]
                                        ```
                                        
                                        ## Fragment 2
                                        ```
                                        [dokładny cytat z dokumentu]
                                        ```
                                        
                                        # 2. SKŁADNIKI ŻYWIENIOWE I SUBSTANCJE WYMIENIONE W DOKUMENCIE
                                        Wymień wszystkie zidentyfikowane składniki żywieniowe, dodatki do żywności, substancje chemiczne, suplementy diety, witaminy, minerały, 
                                        ekstrakty roślinne i inne składniki aktywne biologicznie znalezione w dokumencie. Gdzie to możliwe, podaj:
                                        - Nazwę składnika (zarówno polską jak i łacińską/chemiczną jeśli występuje)
                                        - Numery E (jeśli są to dodatki do żywności)
                                        - Kontekst, w jakim składnik jest wymieniony
                                        - Dozwolone ilości lub ograniczenia, jeśli są podane
                                        
                                        Dla każdego znalezionego składnika podaj krótki cytat z dokumentu jako dowód.
                                        
                                        # 3. KLUCZOWE INFORMACJE REGULACYJNE
                                        Wymień wszystkie znalezione informacje o:
                                        - Przepisach prawnych i rozporządzeniach
                                        - Ograniczeniach lub dozwolonych zastosowaniach
                                        - Zaleceniach organów regulacyjnych (np. EFSA, GIS)
                                        - Opiniach ekspertów
                                        
                                        Jeśli nie znajdziesz żadnych fragmentów dla którejś z sekcji, napisz: "Nie znaleziono informacji na ten temat."
                                        """
                                        
                                        # Przesyłamy cały tekst dokumentu do analizy API
                                        user_prompt = f"Zapytanie: '{search_term}'\n\nDokument '{file_name}':\n\n{full_text}"
                                        
                                        logger.info(f"Wysyłanie zapytania do OpenAI dla dokumentu '{file_name}'")
                                        logger.debug(f"System prompt:\n{system_prompt}")
                                        logger.debug(f"User prompt (początek):\n{user_prompt[:500]}...")
                                        
                                        with console.status("[bold green]Wyszukiwanie fragmentów w dokumencie...[/]"):
                                            response = await client.chat.completions.create(
                                                model="gpt-4o",
                                                messages=[
                                                    {"role": "system", "content": system_prompt},
                                                    {"role": "user", "content": user_prompt}
                                                ],
                                                temperature=0.3
                                            )
                                            fragments = response.choices[0].message.content
                                        
                                        # Logowanie odpowiedzi
                                        logger.info(f"Otrzymano odpowiedź od OpenAI dla dokumentu '{file_name}'")
                                        logger.debug(f"Pełna odpowiedź OpenAI:\n{fragments}")
                                        
                                        # Pokaż znalezione fragmenty
                                        console.print(Panel(Markdown(fragments), title=f"Znalezione fragmenty w dokumencie: {file_name}"))
                                        
                                        # Znajdź wszystkie wystąpienia w tekście z agresywną normalizacją
                                        matches = find_text_in_document(
                                            {'text': full_text, 'document_id': file_path.name, 'title': file_name},
                                            search_term,
                                            pre_context_size=500,
                                            post_context_size=1500,
                                            fuzzy_match=True,
                                            aggressive_normalization=True
                                        )
                                        
                                        # Wyświetl informacje o dokumencie
                                        console.print(Panel(
                                            f"Dokument: {file_name}\nŚcieżka: {file_path}\nDługość tekstu: {len(full_text)} znaków\nZnaleziono {len(matches)} wystąpień frazy '{search_term}'",
                                            title=f"Informacje o dokumencie"
                                        ))
                                        
                                        # Wyświetl znalezione fragmenty
                                        if matches:
                                            matches_panel = ""
                                            for i, match in enumerate(matches, 1):
                                                context = match['context']
                                                match_text = match['match_text']
                                                position = match['position']
                                                
                                                # Podziel tekst, aby podświetlić znalezioną frazę
                                                context_parts = context.split(match_text, 1)
                                                if len(context_parts) > 1:
                                                    highlighted_context = f"{context_parts[0]}[bold on yellow]{match_text}[/bold on yellow]{context_parts[1]}"
                                                else:
                                                    highlighted_context = context
                                                
                                                matches_panel += f"\n[bold cyan]Wystąpienie {i} (pozycja: {position}):[/]\n{highlighted_context}\n---\n"
                                            
                                            console.print(Panel(matches_panel, title=f"Znalezione wystąpienia '{search_term}'"))
                                        else:
                                            console.print(f"[yellow]Nie znaleziono wystąpień frazy '{search_term}' w tekście dokumencie.[/]")
                                            
                                        # Wyświetl początek dokumentu
                                        console.print(Panel(
                                            f"{full_text[:3000]}...",
                                            title=f"Początek dokumentu"
                                        ))
                                else:
                                    console.print("[red]Błąd: Nieprawidłowy numer pliku[/]")
                            
                        except Exception as e:
                            logger.error(f"Błąd podczas przeszukiwania plików: {str(e)}", exc_info=True)
                            console.print(f"[red]Wystąpił błąd podczas przeszukiwania: {str(e)}[/]")
                            
                    elif action.lower().startswith('show '):
                        try:
                            doc_idx = int(action.split()[1]) - 1
                            if 0 <= doc_idx < len(results):
                                selected_doc = results[doc_idx]
                                file_name = selected_doc['file_name']
                                logger.info(f"Pokazanie dokumentu {doc_idx+1}: {file_name}")
                                
                                # Spróbuj odnaleźć oryginalny plik na dysku
                                try:
                                    # Najpierw sprawdź w katalogu processed
                                    processed_dir = Path("data/processed")
                                    pending_dir = Path("data/pending")
                                    
                                    possible_paths = list(processed_dir.glob(f"*{file_name}*")) + list(pending_dir.glob(f"*{file_name}*"))
                                    
                                    if possible_paths:
                                        file_path = possible_paths[0]
                                        logger.info(f"Znaleziono oryginalny plik: {file_path}")
                                        
                                        # Wczytaj cały dokument używając biblioteki docling
                                        from docling.document_converter import DocumentConverter
                                        
                                        with console.status("[bold green]Wczytywanie pełnego dokumentu...[/]"):
                                            converter = DocumentConverter()
                                            doc_result = converter.convert(str(file_path))
                                            
                                            # Pobierz pełny tekst dokumentu - używamy doc_result.document
                                            full_text = doc_result.document.export_to_text()
                                            
                                            # Analizuj pełny dokument z OpenAI
                                            console.print(f"[yellow]Wyszukiwanie konkretnych fragmentów w dokumencie przy użyciu OpenAI. Ta operacja zużywa tokeny.[/]")
                                            
                                            # Logowanie pełnej treści dokumentu
                                            logger.debug(f"Pełna treść dokumentu '{file_name}' (pierwsze 2000 znaków):\n{full_text[:2000]}")
                                            logger.info(f"Dokument '{file_name}' został wczytany, długość tekstu: {len(full_text)} znaków")
                                            
                                            # Pobierz ostatnie zapytanie użytkownika
                                            last_query = query_history[-1] if query_history else "informacje o dokumencie"
                                            logger.info(f"Ostatnie zapytanie użytkownika: '{last_query}'")
                                            
                                            proceed = console.input("[bold yellow]Czy kontynuować z analizą OpenAI? (y/n):[/] ").lower()
                                            
                                            if proceed in ('y', 'yes', 't', 'tak'):
                                                from openai import AsyncOpenAI
                                                client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                                                
                                                # Prompt do dogłębnej analizy dokumentów o żywności
                                                system_prompt = """Jesteś precyzyjnym narzędziem do wyszukiwania i analizy informacji w dokumentach prawnych i naukowych dotyczących żywności.
                                                
                                                Przeprowadź analizę dokumentu i udziel szczegółowej odpowiedzi w następujących sekcjach:
                                                
                                                # 1. ODPOWIEDŹ NA ZAPYTANIE
                                                Znajdź 2-3 DOKŁADNE CYTATY z dokumentu, które najlepiej odpowiadają na zapytanie użytkownika, wraz z kontekstem.
                                                
                                                ## Fragment 1
                                                ```
                                                [dokładny cytat z dokumentu]
                                                ```
                                                
                                                ## Fragment 2
                                                ```
                                                [dokładny cytat z dokumentu]
                                                ```
                                                
                                                # 2. SKŁADNIKI ŻYWIENIOWE I SUBSTANCJE WYMIENIONE W DOKUMENCIE
                                                Wymień wszystkie zidentyfikowane składniki żywieniowe, dodatki do żywności, substancje chemiczne, suplementy diety, witaminy, minerały, 
                                                ekstrakty roślinne i inne składniki aktywne biologicznie znalezione w dokumencie. Gdzie to możliwe, podaj:
                                                - Nazwę składnika (zarówno polską jak i łacińską/chemiczną jeśli występuje)
                                                - Numery E (jeśli są to dodatki do żywności)
                                                - Kontekst, w jakim składnik jest wymieniony
                                                - Dozwolone ilości lub ograniczenia, jeśli są podane
                                                
                                                Dla każdego znalezionego składnika podaj krótki cytat z dokumentu jako dowód.
                                                
                                                # 3. KLUCZOWE INFORMACJE REGULACYJNE
                                                Wymień wszystkie znalezione informacje o:
                                                - Przepisach prawnych i rozporządzeniach
                                                - Ograniczeniach lub dozwolonych zastosowaniach
                                                - Zaleceniach organów regulacyjnych (np. EFSA, GIS)
                                                - Opiniach ekspertów
                                                
                                                Jeśli nie znajdziesz żadnych fragmentów dla którejś z sekcji, napisz: "Nie znaleziono informacji na ten temat."
                                                """
                                                
                                                # Przesyłamy cały tekst dokumentu do analizy API
                                                user_prompt = f"Zapytanie: '{last_query}'\n\nDokument '{file_name}':\n\n{full_text}"
                                                
                                                logger.info(f"Wysyłanie zapytania do OpenAI dla dokumentu '{file_name}'")
                                                logger.debug(f"System prompt:\n{system_prompt}")
                                                logger.debug(f"User prompt (początek):\n{user_prompt[:500]}...")
                                                
                                                # Analizuj dokument, aby znaleźć konkretne fragmenty
                                                with console.status("[bold green]Wyszukiwanie fragmentów w dokumencie...[/]"):
                                                    response = await client.chat.completions.create(
                                                        model="gpt-4o",
                                                        messages=[
                                                            {"role": "system", "content": system_prompt},
                                                            {"role": "user", "content": user_prompt}
                                                        ],
                                                        temperature=0.3
                                                    )
                                                    fragments = response.choices[0].message.content
                                                
                                                # Logowanie odpowiedzi
                                                logger.info(f"Otrzymano odpowiedź od OpenAI dla dokumentu '{file_name}'")
                                                logger.debug(f"Pełna odpowiedź OpenAI:\n{fragments}")
                                                
                                                # Pokaż znalezione fragmenty
                                                console.print(Panel(Markdown(fragments), title=f"Znalezione fragmenty w dokumencie: {file_name}"))
                                            
                                            # Znajdź wszystkie wystąpienia w tekście z agresywną normalizacją
                                            if last_query != "informacje o dokumencie":
                                                matches = find_text_in_document(
                                                    {'text': full_text, 'document_id': file_path.name, 'title': file_name},
                                                    last_query,
                                                    pre_context_size=500,
                                                    post_context_size=1500,
                                                    fuzzy_match=True,
                                                    aggressive_normalization=True
                                                )
                                                
                                                # Wyświetl informacje o dokumencie i znalezionych fragmentach
                                                console.print(Panel(
                                                    f"Dokument: {file_name}\nŚcieżka: {file_path}\nDługość tekstu: {len(full_text)} znaków\nZnaleziono {len(matches)} wystąpień frazy '{last_query}'",
                                                    title=f"Informacje o dokumencie {doc_idx+1}"
                                                ))
                                                
                                                # Wyświetl znalezione fragmenty
                                                if matches:
                                                    matches_panel = ""
                                                    for i, match in enumerate(matches, 1):
                                                        context = match['context']
                                                        match_text = match['match_text']
                                                        position = match['position']
                                                        
                                                        # Podziel tekst, aby podświetlić znalezioną frazę
                                                        context_parts = context.split(match_text, 1)
                                                        if len(context_parts) > 1:
                                                            highlighted_context = f"{context_parts[0]}[bold on yellow]{match_text}[/bold on yellow]{context_parts[1]}"
                                                        else:
                                                            highlighted_context = context
                                                        
                                                        matches_panel += f"\n[bold cyan]Wystąpienie {i} (pozycja: {position}):[/]\n{highlighted_context}\n---\n"
                                                    
                                                    console.print(Panel(matches_panel, title=f"Znalezione wystąpienia '{last_query}'"))
                                            
                                            # Pokaż początek tekstu dla lepszej czytelności
                                            console.print(Panel(
                                                f"{full_text[:3000]}...",
                                                title=f"Początek dokumentu {doc_idx+1}"
                                            ))
                                            
                                            # Log ostateczny rezultat całego procesu
                                            logger.info(f"Zakończono przetwarzanie dokumentu '{file_name}' dla zapytania '{last_query}'")
                                            logger.debug(f"Ostateczny rezultat dla użytkownika: Wyświetlono fragmenty tekstu i pierwsze 3000 znaków")
                                    else:
                                        logger.warning(f"Nie znaleziono oryginalnego pliku dla dokumentu: {file_name}")
                                        console.print(Panel(
                                            f"Dokument: {file_name}\n\n{selected_doc['text']}",
                                            title=f"Treść dokumentu {doc_idx+1} (z chunków)"
                                        ))
                                except Exception as e:
                                    logger.error(f"Błąd podczas odczytu oryginalnego pliku: {str(e)}", exc_info=True)
                                    console.print(f"[red]Wystąpił błąd podczas odczytu oryginalnego pliku: {str(e)}[/]")
                                    console.print(Panel(
                                        f"Dokument: {file_name}\n\n{selected_doc['text']}",
                                        title=f"Treść dokumentu {doc_idx+1} (z chunków)"
                                    ))
                            else:
                                console.print("[red]Błąd: Nieprawidłowy numer dokumentu[/]")
                        except (IndexError, ValueError):
                            console.print("[red]Błąd: Podaj prawidłowy numer dokumentu[/]")
                    
                    elif action.lower().startswith('find '):
                        try:
                            search_term = action[5:].strip()
                            if not search_term:
                                console.print("[red]Błąd: Podaj tekst do wyszukania[/]")
                                continue
                                
                            logger.info(f"Wyszukiwanie tekstu '{search_term}' w znalezionych dokumentach")
                            console.print(f"[bold]Wyszukiwanie '{search_term}' w dokumentach...[/]")
                            
                            # Domyślnie używamy asymetrycznego kontekstu: 500 znaków przed i 1500 znaków po
                            found_chunks = await find_in_documents(
                                searcher, 
                                results, 
                                search_term,
                                pre_context_size=500,
                                post_context_size=1500
                            )
                            display_found_chunks(console, found_chunks)
                        except Exception as e:
                            logger.error(f"Błąd podczas wyszukiwania tekstu: {str(e)}", exc_info=True)
                            console.print(f"[red]Wystąpił błąd podczas wyszukiwania: {str(e)}[/]")
                            
                    elif action.lower().startswith('analyze ') and action.lower() != 'analyze all':
                        try:
                            doc_idx = int(action.split()[1]) - 1
                            if 0 <= doc_idx < len(results):
                                # Generowanie odpowiedzi tylko dla wybranego dokumentu
                                selected_doc = [results[doc_idx]]
                                logger.info(f"Analiza dokumentu {doc_idx+1}: {results[doc_idx]['file_name']}")
                                
                                console.print("[yellow]Uwaga: Ta operacja wymaga użycia API OpenAI i zużywa tokeny.[/]")
                                proceed = console.input("[bold yellow]Czy kontynuować? (y/n):[/] ").lower()
                                
                                if proceed in ('y', 'yes', 't', 'tak'):
                                    console.print("[bold]Generowanie odpowiedzi...[/]")
                                    response = await generate_response(query, selected_doc, context)
                                    console.print(Panel(Markdown(response), title=f"Analiza dokumentu {doc_idx+1}: {results[doc_idx]['file_name']}"))
                                else:
                                    logger.info("Analiza pominięta")
                                    console.print("[blue]Analiza pominięta.[/]")
                            else:
                                console.print("[red]Błąd: Nieprawidłowy numer dokumentu[/]")
                        except (IndexError, ValueError):
                            console.print("[red]Błąd: Podaj prawidłowy numer dokumentu[/]")
                    
                    elif action.lower() == 'analyze all':
                        logger.info("Analiza wszystkich dokumentów")
                        console.print("[yellow]Uwaga: Ta operacja wymaga użycia API OpenAI i zużywa tokeny.[/]")
                        proceed = console.input("[bold yellow]Czy kontynuować? (y/n):[/] ").lower()
                        
                        if proceed in ('y', 'yes', 't', 'tak'):
                            console.print("[bold]Generowanie odpowiedzi dla wszystkich dokumentów...[/]")
                            response = await generate_response(query, results, context)
                            console.print(Panel(Markdown(response), title="Analiza wszystkich dokumentów"))
                        else:
                            logger.info("Analiza pominięta")
                            console.print("[blue]Analiza pominięta.[/]")
                    
                    else:
                        console.print("[yellow]Nieznane polecenie. Wpisz 'help' aby zobaczyć dostępne opcje.[/]")
            
            except Exception as e:
                logger.error(f"Błąd podczas przetwarzania zapytania: {str(e)}", exc_info=True)
                console.print(f"[red]Wystąpił błąd podczas przetwarzania zapytania: {str(e)}[/]")
    
    finally:
        # Zamknij połączenia
        if hasattr(searcher, 'close'):
            await searcher.close()
        logger.info("Zakończenie programu")
        console.print("[bold green]Do widzenia![/]")

def parse_args():
    """Parsuje argumenty wiersza poleceń."""
    parser = argparse.ArgumentParser(description='Document Search CLI')
    parser.add_argument('--query', type=str, help='Zapytanie do wyszukania')
    parser.add_argument('--find', type=str, help='Fraza do wyszukania w dokumentach')
    parser.add_argument('--results-file', type=str, help='Ścieżka do pliku JSON z wynikami wyszukiwania')
    parser.add_argument('--generate', action='store_true', help='Generuj odpowiedź na podstawie wyników')
    parser.add_argument('--json', action='store_true', help='Zwróć wyniki w formacie JSON')
    
    return parser.parse_args()

async def run_cli_mode():
    """Uruchamia aplikację w trybie interaktywnym CLI."""
    await main()

async def run_api_mode(args):
    """Uruchamia aplikację w trybie API."""
    searcher = Searcher()
    
    try:
        # Jeśli podano zapytanie, wykonaj wyszukiwanie
        if args.query:
            results = await searcher.search(args.query, limit=5)
            
            if args.json:
                return json.dumps(results)
            else:
                # Wyświetl wyniki w konsoli
                console = Console()
                table = create_results_table(results)
                console.print(table)
                return None
        
        # Jeśli podano frazę do wyszukania i plik z wynikami
        if args.find and args.results_file:
            # Wczytaj wyniki z pliku
            with open(args.results_file, 'r') as f:
                results = json.load(f)
            
            # Wyszukaj frazę w dokumentach
            found_chunks = await find_in_documents(searcher, results, args.find)
            
            if args.json:
                return json.dumps(found_chunks)
            else:
                # Wyświetl znalezione fragmenty w konsoli
                console = Console()
                display_found_chunks(console, found_chunks)
                return None
        
        # Jeśli podano flagę generate, zapytanie i plik z wynikami
        if args.generate and args.query and args.results_file:
            # Wczytaj wyniki z pliku
            with open(args.results_file, 'r') as f:
                results = json.load(f)
            
            # Generuj odpowiedź
            response = await generate_response(args.query, results)
            
            if args.json:
                return json.dumps({"response": response})
            else:
                # Wyświetl odpowiedź w konsoli
                console = Console()
                console.print(Panel(Markdown(response), title="Wygenerowana odpowiedź"))
                return None
    
    finally:
        # Zamknij połączenia
        if hasattr(searcher, 'close'):
            await searcher.close()

if __name__ == "__main__":
    # Konfiguracja głównego loggera
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parsuj argumenty wiersza poleceń
    args = parse_args()
    
    # Sprawdź, czy uruchomiono w trybie API czy interaktywnym
    if args.query or args.find or args.generate:
        # Tryb API
        result = asyncio.run(run_api_mode(args))
        if result:
            print(result)
    else:
        # Tryb interaktywny
        asyncio.run(run_cli_mode())
