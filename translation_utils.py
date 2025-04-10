from typing import List, Dict, Any

# Słownik tłumaczeń dla najczęstszych terminów
TRANSLATION_DICT = {
    # Polski -> Angielski
    "sorbinian potasu": "potassium sorbate",
    "kwas askorbinowy": "ascorbic acid",
    "witamina c": "vitamin c",
    "konserwant": "preservative",
    "dodatek do żywności": "food additive",
    "składniki": "ingredients",
    # Angielski -> Polski
    "potassium sorbate": "sorbinian potasu",
    "ascorbic acid": "kwas askorbinowy",
    "vitamin c": "witamina c",
    "preservative": "konserwant",
    "food additive": "dodatek do żywności",
    "ingredients": "składniki",
}

# Słownik specjalistycznych terminów z synonimami
SPECIALIZED_TERMS = {
    "sorbinian potasu": ["potassium sorbate", "E202", "sorbinian", "konserwant sorbinian"],
    "kwas askorbinowy": ["ascorbic acid", "vitamin c", "E300", "kwas L-askorbinowy"],
    "konserwant": ["preservative", "środek konserwujący"],
    "dodatek do żywności": ["food additive", "E-dodatek"],
}

class SimpleTranslator:
    """Prosta klasa tłumacza oparta na słowniku."""
    
    def __init__(self, translation_dict: Dict[str, str] = None, specialized_terms: Dict[str, List[str]] = None):
        """Inicjalizuje tłumacza z opcjonalnym słownikiem tłumaczeń."""
        self.translation_dict = translation_dict or TRANSLATION_DICT
        self.specialized_terms = specialized_terms or SPECIALIZED_TERMS
        
    def translate(self, text: str, source_lang: str = "PL", target_lang: str = "EN") -> str:
        """Tłumaczy tekst używając słownika."""
        text_lower = text.lower()
        
        # Sprawdź, czy tekst jest w słowniku tłumaczeń
        if text_lower in self.translation_dict:
            return self.translation_dict[text_lower]
        
        # Jeśli nie ma bezpośredniego tłumaczenia, zwróć oryginalny tekst
        return text
        
    def get_multilingual_variants(self, query: str) -> List[str]:
        """Zwraca warianty zapytania w różnych językach."""
        variants = [query]  # Zawsze zawiera oryginalne zapytanie
        
        # Spróbuj przetłumaczyć na angielski
        translated = self.translate(query, "PL", "EN")
        if translated != query and translated.lower() != query.lower():
            variants.append(translated)
            
        # Dodaj warianty ze słownika terminów specjalistycznych
        query_lower = query.lower()
        if query_lower in self.specialized_terms:
            variants.extend(self.specialized_terms[query_lower])
            
        return list(set(variants))  # Usuń duplikaty