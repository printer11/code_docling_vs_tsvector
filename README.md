# code_docling_vs_tsvector
porownanie systemow przetwarzania dokumentów i wyszukiwania semantycznego
=======
# Dockling - Document Processing and Search System

Dockling to system przetwarzania dokumentów i wyszukiwania informacji, który łączy semantyczne wyszukiwanie wektorowe z mechanizmem analizy tekstu. System przetwarza dokumenty (PDF, DOCX, TXT) i umożliwia zaawansowane przeszukiwanie ich zawartości.

## Funkcjonalności

- Przetwarzanie dokumentów PDF, DOCX i TXT
- Inteligentne dzielenie dokumentów na fragmenty (chunking)
- Wektorowe wyszukiwanie semantyczne
- Rozpoznawanie typów dokumentów
- Zaawansowana deduplikacja wyników
- Wielojęzyczne wyszukiwanie
- Interaktywny interfejs CLI
- Analiza i eksploracja dokumentów

## Wymagania

- Python 3.9+
- OpenAI API Key
- Opcjonalnie dostęp do bazy PostgreSQL z pgvector (jeśli nie używasz trybu dockling)

## Instalacja

1. Sklonuj repozytorium:
   ```bash
   git clone https://github.com/username/code_dockling.git
   cd code_dockling
   ```

2. Utwórz i aktywuj środowisko wirtualne:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Na Windows: .venv\Scripts\activate
   ```

3. Zainstaluj zależności:
   ```bash
   pip install -r requirements.txt  # pełny zestaw zależności
   # LUB
   pip install -r requirements_minimal.txt  # minimalne zależności dla podstawowej funkcjonalności
   ```
   
   > **Uwaga**: Plik `requirements_minimal.txt` zawiera tylko niezbędne pakiety do uruchomienia podstawowych funkcji. Dla pełnej funkcjonalności (w tym zaawansowanego przetwarzania PDF i interaktywnego interfejsu CLI) używaj `requirements.txt`.

4. Skonfiguruj zmienne środowiskowe:
   - Utwórz plik `.env` bazując na dostarczonym szablonie
   - Dodaj swój klucz API OpenAI
   - Skonfiguruj połączenie z bazą danych (jeśli używasz pgvector)

## Struktura katalogów

- `data/pending/` - katalog dla dokumentów oczekujących na przetworzenie
- `data/processed/` - katalog dla przetworzonych dokumentów
- `logs/` - katalog z plikami dziennika
- `config/` - pliki konfiguracyjne
- `utils/` - narzędzia pomocnicze

## Użycie

### Przetwarzanie dokumentów

1. Umieść dokumenty w katalogu `data/pending/`
2. Uruchom skrypt przetwarzania:
   ```bash
   python dockling_main.py
   ```

### Wyszukiwanie

Podstawowe wyszukiwanie:
```bash
python main.py --search "Twoje zapytanie" --limit 5
```

Wyszukiwanie z filtrem typu dokumentu:
```bash
python main.py --search "Twoje zapytanie" --type-filter "OPINIA"
```

### Interaktywny interfejs CLI

Uruchom interaktywny interfejs CLI:
```bash
python interactive_cli.py
```

W interfejsie CLI dostępne są następujące polecenia:
- `search tekst` - zaawansowane wyszukiwanie z obsługą wariantów i tłumaczeń
- `find tekst` - wyszukuje konkretny tekst w znalezionych dokumentach
- `search-across tekst` - przeszukuje wszystkie dokumenty dla dokładnego ciągu znaków
- `search-all tekst` - przeszukuje aktualne wyniki wyszukiwania
- `search-every tekst` - przeszukuje wszystkie pliki w katalogach
- `show N` - pokazuje pełną treść dokumentu o numerze N
- `analyze N` - analizuje dokument o numerze N
- `analyze all` - analizuje wszystkie dokumenty
- `limit N` - zmienia limit wyników wyszukiwania
- `context` - ustawia stały kontekst dla wszystkich zapytań
- `history` - pokazuje historię zapytań
- `help` - wyświetla pomoc
- `exit` / `quit` - kończy program

### Używanie ulepszonego wyszukiwania

Aby skorzystać z ulepszonego wyszukiwania z deduplikacją:
```bash
python main.py --search "Twoje zapytanie" --use-enhanced
```

Lub z CLI:
```bash
python interactive_cli.py --use-enhanced
```

## Dostępne moduły

1. **dockling_loader.py** - podstawowa klasa do ładowania i przetwarzania dokumentów
2. **dockling_loader_enhanced.py** - rozszerzona wersja z deduplikacją wyników
3. **dockling_main.py** - skrypt do wsadowego przetwarzania dokumentów
4. **interactive_cli.py** - interaktywny interfejs wiersza poleceń
5. **main.py** - główny skrypt do wyszukiwania z argumentami

## Pliki instalacyjne

- **requirements.txt** - pełny zestaw zależności dla wszystkich funkcjonalności
- **requirements_minimal.txt** - minimalny zestaw zależności dla podstawowej funkcjonalności
- **.env.example** - przykładowy plik konfiguracyjny

## Dostosowywanie

System można dostosować poprzez:
1. Edycję plików konfiguracyjnych w katalogu `config/`
2. Modyfikację parametrów w pliku `.env`
3. Dostosowanie prompta dla modelu OpenAI w plikach źródłowych

## Rozwiązywanie problemów

W przypadku problemów:
1. Sprawdź pliki dziennika w katalogu `logs/`
2. Upewnij się, że klucz API OpenAI jest poprawny
3. Sprawdź dostęp do bazy danych (w trybie pgvector)
4. Upewnij się, że masz zainstalowane wszystkie wymagane zależności

### Problemy z instalacją zależności

- Jeśli masz problemy z instalacją `docling`, spróbuj najpierw zainstalować minimalne zależności: `pip install -r requirements_minimal.txt`
- W przypadku problemów z `psycopg2-binary`, możesz zainstalować `psycopg2` (wersja bez binarnej kompilacji): `pip install psycopg2`
- Jeśli używasz Windows, niektóre pakiety mogą wymagać dodatkowych bibliotek lub kompilatorów C++. W takim przypadku użyj wheelów z repozytorium: https://www.lfd.uci.edu/~gohlke/pythonlibs/

## Licencja

Ten projekt jest objęty licencją MIT - zobacz plik LICENSE dla szczegółów.

