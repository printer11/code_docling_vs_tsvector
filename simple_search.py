import re
import os
from unidecode import unidecode
import PyPDF2  # Do obsługi plików PDF

def normalize_text(text):
    """
    Normalizuje tekst do formy podstawowej:
    - usuwa znaki diakrytyczne
    - zamienia na małe litery
    - usuwa interpunkcję
    """
    # Usuwanie znaków diakrytycznych
    text = unidecode(text)
    # Zamiana na małe litery
    text = text.lower()
    # Usuwanie interpunkcji i znaków specjalnych
    text = re.sub(r'[^\w\s]', '', text)
    return text

def read_file_content(file_path):
    """
    Odczytuje zawartość pliku w zależności od jego typu (tekstowy lub PDF).
    
    Args:
        file_path (str): Ścieżka do pliku
    
    Returns:
        list: Lista linii tekstu lub None w przypadku błędu
    """
    # Sprawdzenie czy plik istnieje
    if not os.path.exists(file_path):
        print(f"Błąd: Plik {file_path} nie istnieje.")
        return None
    
    # Sprawdzenie rozszerzenia pliku
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()
    
    # Obsługa plików PDF
    if file_extension == '.pdf':
        try:
            print("Wykryto plik PDF. Rozpoczynam ekstrakcję tekstu...")
            content = []
            page_markers = {}  # Słownik do przechowywania informacji o numerach stron
            line_counter = 0
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # Liczba stron w PDF
                num_pages = len(pdf_reader.pages)
                print(f"Liczba stron w PDF: {num_pages}")
                
                # Ekstrakcja tekstu ze wszystkich stron
                for page_num in range(num_pages):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    
                    if page_text:
                        # Dodanie markera strony
                        page_marker = f"[Strona {page_num + 1}]"
                        content.append(page_marker)
                        page_markers[line_counter] = page_num + 1
                        line_counter += 1
                        
                        # Podziel tekst na linie
                        lines = page_text.split('\n')
                        for line in lines:
                            content.append(line)
                            line_counter += 1
            
            return content, page_markers
            
        except Exception as e:
            print(f"Błąd podczas odczytu pliku PDF: {e}")
            return None, None
    
    # Obsługa plików tekstowych
    else:
        try:
            # Próba wczytania pliku z różnymi kodowaniami
            encodings = ['utf-8', 'latin2', 'windows-1250', 'iso-8859-2', 'cp1250']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        content = file.readlines()
                    print(f"Sukces! Plik wczytany z kodowaniem: {encoding}")
                    return [line.strip() for line in content], {}
                except UnicodeDecodeError:
                    continue
            
            print("Błąd: Nie udało się odczytać pliku. Nieobsługiwane kodowanie.")
            return None, None
                
        except Exception as e:
            print(f"Błąd podczas odczytu pliku: {e}")
            return None, None

def find_phrase_occurrences(file_path, phrase_to_find, context_chars=50):
    """
    Wyszukuje wszystkie wystąpienia frazy w pliku,
    ignorując odmianę przez przypadki.
    
    Args:
        file_path (str): Ścieżka do pliku
        phrase_to_find (str): Fraza do znalezienia
        context_chars (int): Liczba znaków kontekstu przed i po znalezionej frazie
    
    Returns:
        list: Lista znalezionych fragmentów tekstu z kontekstem
    """
    # Wczytanie zawartości pliku
    content_lines, page_markers = read_file_content(file_path)
    
    if content_lines is None:
        return "Nie udało się wczytać pliku."
    
    # Normalizacja frazy do wyszukania
    normalized_phrase = normalize_text(phrase_to_find)
    print(f"Znormalizowana fraza do wyszukania: '{normalized_phrase}'")
    
    # Lista na wyniki
    results = []
    
    # Śledzenie bieżącej strony
    current_page = None
    
    # Przeszukiwanie każdej linii
    for i, line in enumerate(content_lines):
        # Sprawdź, czy linia zawiera marker strony
        if i in page_markers:
            current_page = page_markers[i]
            continue
        
        # Normalizacja linii
        normalized_line = normalize_text(line)
        
        # Wyszukiwanie frazy w znormalizowanym tekście
        if normalized_phrase in normalized_line:
            # Znaleziono dopasowanie, dodaj do wyników
            # Zbierz kontekst z kilku linii (przed i po)
            context_before = ""
            context_after = ""
            
            # Zbierz kontekst przed - maksymalnie 5 linii w górę
            for j in range(i-1, max(0, i-5), -1):
                if j not in page_markers:  # Pomijamy markery stron
                    context_before = content_lines[j] + "\n" + context_before
            
            # Zbierz kontekst po - maksymalnie 10 linii w dół
            for j in range(i+1, min(len(content_lines), i+10)):
                if j not in page_markers:  # Pomijamy markery stron
                    context_after += "\n" + content_lines[j]
            
            result = {
                'line_number': i + 1,
                'page': current_page,
                'original_line': line,
                'normalized_line': normalized_line,
                'position': normalized_line.find(normalized_phrase),
                'context_before': context_before,
                'context_after': context_after
            }
            results.append(result)
    
    return results

def display_results(results, phrase, context_before=200, context_after=500):
    """
    Wyświetla wyniki wyszukiwania w czytelnej formie z określoną ilością 
    znaków kontekstu przed i po znalezionej frazie.
    """
    if isinstance(results, str):
        # Obsługa błędów
        print(results)
        return
    
    if not results:
        print(f"Nie znaleziono frazy '{phrase}' w tekście.")
        return
    
    print(f"Znaleziono {len(results)} wystąpień frazy '{phrase}':")
    for i, result in enumerate(results, 1):
        # Wyświetl informację o znalezisku - numer i stronę
        page_info = f" (Strona {result['page']})" if 'page' in result and result['page'] else ""
        print(f"\n{i}. Linia {result['line_number']}{page_info}:")
        
        original_line = result['original_line']
        position = result['position']
        normalized_phrase = normalize_text(phrase)
        phrase_length = len(phrase)  # Używamy długości oryginalnej frazy
        
        # Zbierz pełny kontekst
        full_context = result['context_before'] + "\n" + original_line + "\n" + result['context_after']
        
        # Znajdź frazę w oryginalnym kontekście (może być z inną wielkością liter)
        phrase_pos_in_original = original_line.lower().find(phrase.lower())
        if phrase_pos_in_original >= 0:
            # Rzeczywista fraza z oryginalnego tekstu
            actual_phrase = original_line[phrase_pos_in_original:phrase_pos_in_original + phrase_length]
        else:
            # Fallback jeśli nie można znaleźć
            actual_phrase = phrase
            
        # Wyświetl kontekst z separatorem
        print(f"--- Kontekst przed ---")
        print(result['context_before'])
        print(f"--- Znaleziona fraza: {actual_phrase} ---")
        print(original_line)
        print(result['context_after'])

def main():
    # Ścieżka do pliku (zhardcodowana)
    file_path = "/Users/michalk/code/code_DSPY/code_dockling_foodie/data/processed/Opinia_VERSHOLD - 15.07.pdf"
    print(f"Używam pliku: {file_path}")
    
    # Fraza do wyszukania
    search_phrase = input("Podaj frazę do wyszukania: ")
    
    # Liczba znaków kontekstu przed i po frazie
    context_before = 200  # 200 znaków przed
    context_after = 500   # 500 znaków po
    
    # Wyszukiwanie frazy
    results = find_phrase_occurrences(file_path, search_phrase, max(context_before, context_after))
    
    # Wyświetlanie wyników
    display_results(results, search_phrase, context_before, context_after)

if __name__ == "__main__":
    main()