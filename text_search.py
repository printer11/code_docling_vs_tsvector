# text_search.py
import re
from typing import Dict, List, Any, Optional
import logging
from unidecode import unidecode
import os
import logging

logger = logging.getLogger(__name__)

def normalize_text(text: str, aggressive: bool = False) -> str:
    """
    Normalizuje tekst do formy podstawowej do wyszukiwania.
    
    Args:
        text: Tekst do normalizacji
        aggressive: Czy stosować agresywną normalizację (usuwa znaki diakrytyczne)
    
    Returns:
        Znormalizowany tekst
    """
    if text is None:
        return ""
        
    # Utwórz kopię, żeby nie modyfikować oryginału
    normalized = text
    
    # Zamiana na małe litery
    normalized = normalized.lower()
    
    if aggressive:
        # Usuwanie znaków diakrytycznych (tylko dla agresywnej normalizacji)
        normalized = unidecode(normalized)
        # Usuwanie interpunkcji i znaków specjalnych
        normalized = re.sub(r'[^\w\s]', '', normalized)
    else:
        # Usuwanie tylko interpunkcji, ale zachowanie znaków diakrytycznych
        normalized = re.sub(r'[^\w\sąęćńóśźżłĄĘĆŃÓŚŹŻŁ]', '', normalized)
    
    return normalized

def find_text_in_document(
    document: Dict[str, Any], 
    search_text: str, 
    context_size: int = 100,
    pre_context_size: Optional[int] = None,
    post_context_size: Optional[int] = None,
    fuzzy_match: bool = False,
    aggressive_normalization: bool = False
) -> List[Dict[str, Any]]:
    """
    Wyszukuje fragmenty tekstu w dokumencie z kontekstem.
    
    Args:
        document: Słownik zawierający dokument (z kluczem 'text')
        search_text: Tekst do wyszukania
        context_size: Domyślny symetryczny rozmiar kontekstu (liczba znaków przed i po)
        pre_context_size: Liczba znaków kontekstu przed dopasowaniem (nadpisuje context_size)
        post_context_size: Liczba znaków kontekstu po dopasowaniu (nadpisuje context_size)
        fuzzy_match: Czy stosować przybliżone dopasowanie (ignoruje wielkość liter i znaki diakrytyczne)
        aggressive_normalization: Czy stosować agresywną normalizację (usuwa znaki diakrytyczne)
        
    Returns:
        Lista słowników z dopasowaniami i kontekstem
    """
    matches = []
    doc_text = document.get('text', '')
    
    if not doc_text or not search_text:
        return matches
    
    # Ustal rozmiary kontekstu
    before_size = pre_context_size if pre_context_size is not None else context_size
    after_size = post_context_size if post_context_size is not None else context_size
    
    # Normalizacja tekstu i frazy wyszukiwania
    if fuzzy_match:
        # Normalizacja dokumentu i zapytania
        doc_text_norm = normalize_text(doc_text, aggressive=aggressive_normalization)
        search_text_norm = normalize_text(search_text, aggressive=aggressive_normalization)
        
        # Znajdź wszystkie wystąpienia
        start_pos = 0
        while True:
            pos = doc_text_norm.find(search_text_norm, start_pos)
            if pos == -1:
                break
                
            # Znajdź oryginalny fragment (może być innej długości niż znormalizowany)
            # Oblicz przybliżoną pozycję w oryginalnym tekście
            orig_pos = pos
            if aggressive_normalization:
                # W przypadku agresywnej normalizacji musimy szukać ręcznie
                # To jest przybliżenie, które może nie być idealne
                char_count = 0
                for i, char in enumerate(doc_text):
                    if char_count >= pos:
                        orig_pos = i
                        break
                    char_norm = normalize_text(char, aggressive=aggressive_normalization)
                    if char_norm:  # jeśli po normalizacji pozostał jakiś znak
                        char_count += 1
            
            # Znajdź przybliżony oryginalny tekst dopasowania
            approx_match_text = doc_text[orig_pos:orig_pos + len(search_text) + 5]
            
            # Ustal kontekst wokół dopasowania
            context_start = max(0, orig_pos - before_size)
            context_end = min(len(doc_text), orig_pos + len(search_text) + after_size)
            context = doc_text[context_start:context_end]
            
            matches.append({
                'position': orig_pos,
                'match_text': approx_match_text[:len(search_text)],
                'context': context,
                'is_fuzzy': True,
                'is_exact_match': False,
                'document_id': document.get('document_id', document.get('id', '')),
                'document_title': document.get('title', document.get('file_name', '')),
                'match_quality': 0.8  # Niższa jakość dla dopasowań przybliżonych
            })
            
            start_pos = pos + len(search_text_norm)
    else:
        # Dokładne dopasowanie - szukamy dokładnie tej frazy
        start_pos = 0
        while True:
            pos = doc_text.lower().find(search_text.lower(), start_pos)
            if pos == -1:
                break
                
            # Ustal kontekst wokół dopasowania
            context_start = max(0, pos - before_size)
            context_end = min(len(doc_text), pos + len(search_text) + after_size)
            context = doc_text[context_start:context_end]
            
            matches.append({
                'position': pos,
                'match_text': doc_text[pos:pos+len(search_text)],
                'context': context,
                'is_fuzzy': False,
                'is_exact_match': True,
                'document_id': document.get('document_id', document.get('id', '')),
                'document_title': document.get('title', document.get('file_name', '')),
                'match_quality': 1.0  # Najwyższa jakość dla dokładnych dopasowań
            })
            
            start_pos = pos + len(search_text)
    
    # Dodatkowe informacje o dokumencie do każdego dopasowania
    for match in matches:
        match['document_metadata'] = document.get('doc_metadata', document.get('metadata', {}))
    
    logger.debug(f"Found {len(matches)} matches for '{search_text}' in document {document.get('title', document.get('file_name', ''))}")
    return matches

def find_text_in_file(
    file_path: str,
    search_text: str,
    pre_context_size: int = 500,
    post_context_size: int = 1500,
    fuzzy_match: bool = True,
    aggressive_normalization: bool = False
) -> List[Dict[str, Any]]:
    """
    Wyszukuje tekst w pliku na dysku.
    
    Args:
        file_path: Ścieżka do pliku
        search_text: Tekst do wyszukania
        pre_context_size: Liczba znaków kontekstu przed dopasowaniem
        post_context_size: Liczba znaków kontekstu po dopasowaniu
        fuzzy_match: Czy stosować przybliżone dopasowanie
        aggressive_normalization: Czy stosować agresywną normalizację
        
    Returns:
        Lista słowników z dopasowaniami i kontekstem
    """
    try:
        # Sprawdź, czy plik istnieje
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []
            
        # Wczytaj plik z różnymi kodowaniami
        text_content = None
        encodings = ['utf-8', 'latin2', 'windows-1250', 'iso-8859-2', 'cp1250']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    text_content = f.read()
                logger.debug(f"Successfully read file {file_path} with encoding {encoding}")
                break
            except UnicodeDecodeError:
                continue
        
        if text_content is None:
            # Spróbuj wczytać jako PDF
            try:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    text_content = ""
                    for page in pdf_reader.pages:
                        text_content += page.extract_text() + "\n\n"
                logger.debug(f"Successfully read PDF file: {file_path}")
            except Exception as e:
                logger.error(f"Failed to read file as PDF: {e}")
                return []
        
        # Wyszukaj tekst w zawartości pliku
        doc = {
            'id': os.path.basename(file_path),
            'title': os.path.basename(file_path),
            'text': text_content
        }
        
        return find_text_in_document(
            doc, 
            search_text, 
            pre_context_size=pre_context_size,
            post_context_size=post_context_size,
            fuzzy_match=fuzzy_match,
            aggressive_normalization=aggressive_normalization
        )
        
    except Exception as e:
        logger.error(f"Error searching in file {file_path}: {e}", exc_info=True)
        return []

def search_across_files(
    search_text: str,
    file_paths: List[str],
    pre_context_size: int = 500,
    post_context_size: int = 1500,
    fuzzy_match: bool = True,
    aggressive_normalization: bool = False
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Wyszukuje tekst w wielu plikach jednocześnie.
    
    Args:
        search_text: Tekst do wyszukania
        file_paths: Lista ścieżek do plików
        pre_context_size: Liczba znaków kontekstu przed dopasowaniem
        post_context_size: Liczba znaków kontekstu po dopasowaniu
        fuzzy_match: Czy stosować przybliżone dopasowanie
        aggressive_normalization: Czy stosować agresywną normalizację
        
    Returns:
        Słownik z wynikami wyszukiwania dla każdego pliku
    """
    results = {}
    
    for file_path in file_paths:
        matches = find_text_in_file(
            file_path,
            search_text,
            pre_context_size=pre_context_size,
            post_context_size=post_context_size,
            fuzzy_match=fuzzy_match,
            aggressive_normalization=aggressive_normalization
        )
        
        if matches:
            results[file_path] = matches
            
    return results