from typing import List, Dict, Any, Callable
import logging
import re
from datetime import datetime
from text_search import find_text_in_document
from ranking import rank_search_results
from translation_utils import SimpleTranslator

# Konfiguracja loggera
logger = logging.getLogger(__name__)

# Słownik dodatków do żywności (E-numerów)
E_NUMBERS = {
    "E1412": ["fosforan diskrobiowy", "distarch phosphate", "modified starch"],
    "E202": ["sorbinian potasu", "potassium sorbate"],
    "E300": ["kwas askorbinowy", "ascorbic acid", "vitamin c"],
    "E330": ["kwas cytrynowy", "citric acid"],
    # Dodaj więcej E-numerów według potrzeb
}

class EnhancedTwoLayerSearch:
    """
    Usprawniona implementacja dwuwarstwowego wyszukiwania.
    """
    
    def __init__(self, vector_search_func, text_search_func=find_text_in_document):
        """
        Inicjalizuje wyszukiwanie dwuwarstwowe.
        
        Args:
            vector_search_func: Funkcja do wyszukiwania semantycznego
            text_search_func: Funkcja do wyszukiwania tekstowego
        """
        self.vector_search = vector_search_func
        self.text_search = text_search_func
        self.translator = SimpleTranslator()
        self.e_numbers = E_NUMBERS
    
    async def search(self, query: str, max_vector_results: int = 10, 
               context_size: int = 150, pre_context_size: int = 500, 
               post_context_size: int = 1500, fuzzy_match: bool = True) -> Dict[str, Any]:
        """
        Wykonuje wyszukiwanie dwuwarstwowe z rozszerzoną analizą zapytania.
        
        Args:
            query: Zapytanie do wyszukania
            max_vector_results: Maksymalna liczba wyników z wyszukiwania wektorowego
            context_size: Domyślna symetryczna wielkość kontekstu (jeśli nie używane są pre/post_context_size)
            pre_context_size: Liczba znaków kontekstu przed wystąpieniem (domyślnie 500)
            post_context_size: Liczba znaków kontekstu po wystąpieniu (domyślnie 1500)
            fuzzy_match: Czy używać dopasowania przybliżonego
        """
        start_time = datetime.now()
        query = query.strip()
        
        results = {
            "query": query,
            "vector_results": [],
            "text_matches": [],
            "ranked_matches": [],
            "multilingual_variants": [],
            "e_number_variants": [],
            "metadata": {
                "search_time": None,
                "num_documents": 0,
                "num_matches": 0,
                "query_analysis": {}
            }
        }
        
        try:
            # 0. Analiza zapytania
            query_analysis = self._analyze_query(query)
            results["metadata"]["query_analysis"] = query_analysis
            
            # 1. Generuj warianty zapytania
            query_variants = self._get_expanded_variants(query, query_analysis)
            results["multilingual_variants"] = query_variants
            
            # 1.1 Dodaj warianty z E-numerów
            e_number_variants = self._get_e_number_variants(query, query_analysis)
            results["e_number_variants"] = e_number_variants
            query_variants.extend(e_number_variants)
            
            # Usunięcie duplikatów z wariantów zapytania
            query_variants = list(set(query_variants))
            logger.info(f"Expanded query variants: {query_variants}")
            
            # 2. Wyszukiwanie semantyczne
            all_vector_results = []
            for variant in query_variants:
                vector_docs = await self.vector_search(variant, limit=max_vector_results)
                all_vector_results.extend(vector_docs)
            
            # Deduplikacja wyników
            seen_ids = set()
            deduplicated_results = []
            for doc in all_vector_results:
                doc_id = doc.get("id", doc.get("document_id", ""))
                if doc_id and doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    deduplicated_results.append(doc)
            
            results["vector_results"] = deduplicated_results[:max_vector_results]
            results["metadata"]["num_documents"] = len(results["vector_results"])
            
            # 3. Wyszukiwanie tekstowe w znalezionych dokumentach
            all_text_matches = []
            for doc in results["vector_results"]:
                for variant in query_variants:
                    matches = self.text_search(
                        doc, 
                        variant, 
                        context_size=context_size,
                        pre_context_size=pre_context_size,
                        post_context_size=post_context_size,
                        fuzzy_match=fuzzy_match,
                        aggressive_normalization=True
                    )
                    for match in matches:
                        match["query_variant"] = variant
                    all_text_matches.extend(matches)
            
            results["text_matches"] = all_text_matches
            results["metadata"]["num_matches"] = len(all_text_matches)
            
            # 4. Ranking wyników z uwzględnieniem dodatkowych informacji
            if all_text_matches:
                results["ranked_matches"] = self._enhanced_ranking(all_text_matches, query, query_analysis)
            
        except Exception as e:
            logger.exception(f"Błąd podczas wyszukiwania: {e}")
            results["error"] = str(e)
        
        # Czas trwania wyszukiwania
        results["metadata"]["search_time"] = (datetime.now() - start_time).total_seconds()
        
        return results
    
    def _analyze_query(self, query: str) -> Dict:
        """
        Analizuje zapytanie, aby wykryć potencjalne E-numery, składniki, etc.
        """
        analysis = {
            "e_numbers": [],
            "ingredients": [],
            "keywords": []
        }
        
        # Szukaj E-numerów używając regexów
        e_patterns = [
            r'E[ -]?(\d{3,4}[a-z]?)',  # E100, E-100, E 100
            r'(\d{3,4}[a-z]?)[ -]?E',   # 100E, 100-E, 100 E
        ]
        
        for pattern in e_patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            for match in matches:
                if len(match.groups()) > 0:
                    e_number = f"E{match.group(1)}"
                    analysis["e_numbers"].append(e_number.upper())
        
        # Sprawdź czy zapytanie zawiera nazwy dodatków i dopasuj je do E-numerów
        common_additives = {
            "fosforan diskrobiowy": "E1412",
            "sorbinian potasu": "E202",
            "kwas askorbinowy": "E300",
            "kwas cytrynowy": "E330",
            "distarch phosphate": "E1412",
            "potassium sorbate": "E202",
            "ascorbic acid": "E300",
            "citric acid": "E330",
        }
        
        for name, e_num in common_additives.items():
            if re.search(r'\b' + re.escape(name) + r'\b', query, re.IGNORECASE):
                analysis["ingredients"].append(name)
                if e_num not in analysis["e_numbers"]:
                    analysis["e_numbers"].append(e_num)
        
        # Ekstrahuj potencjalne słowa kluczowe (tymczasowo używamy wszystkich słów w zapytaniu)
        words = query.split()
        for word in words:
            if len(word) > 3 and word.lower() not in ["jest", "oraz", "jako", "przy"]:
                analysis["keywords"].append(word)
        
        return analysis
    
    def _get_expanded_variants(self, query: str, query_analysis: Dict) -> List[str]:
        """
        Generuje rozszerzone warianty zapytania na podstawie analizy.
        """
        # Podstawowe warianty
        base_variants = self.translator.get_multilingual_variants(query)
        
        # Dodaj warianty dla znalezionych składników
        ingredient_variants = []
        for ingredient in query_analysis.get("ingredients", []):
            ingredient_variants.extend(self.translator.get_multilingual_variants(ingredient))
        
        # Kombinacja podstawowych wariantów i wariantów składników
        return list(set(base_variants + ingredient_variants))
    
    def _get_e_number_variants(self, query: str, query_analysis: Dict) -> List[str]:
        """
        Generuje warianty zapytania na podstawie numerów E.
        """
        e_variants = []
        
        for e_num in query_analysis.get("e_numbers", []):
            if e_num in self.e_numbers:
                e_variants.extend(self.e_numbers[e_num])
        
        return e_variants
    
    def _enhanced_ranking(self, matches: List[Dict], query: str, query_analysis: Dict) -> List[Dict]:
        """
        Zaawansowane rankowanie wyników z uwzględnieniem analizy zapytania.
        """
        # Najpierw użyj podstawowego rankingu
        basic_ranked = rank_search_results(matches, query)
        
        # Sortowanie rozszerzone
        def enhanced_score(match):
            base_score = match.get("score", 0)
            boost = 0
            
            # Zwiększ ocenę, jeśli dokument zawiera E-numery z zapytania
            doc_metadata = match.get("document_metadata", {})
            chunk_metadata = match.get("metadata", {})
            
            # Sprawdź e_numbers w metadanych chunka
            chunk_e_numbers = []
            if "e_numbers" in chunk_metadata:
                chunk_e_numbers = [e.get("e_number", "") for e in chunk_metadata.get("e_numbers", [])]
            
            # Sprawdź e_numbers w metadanych dokumentu
            doc_e_numbers = []
            if "document_analysis" in doc_metadata and "e_numbers" in doc_metadata.get("document_analysis", {}):
                doc_e_numbers = [e.get("e_number", "") for e in doc_metadata.get("document_analysis", {}).get("e_numbers", [])]
            
            # Połącz wszystkie e_numbers
            all_e_numbers = set(chunk_e_numbers + doc_e_numbers)
            
            # Zwiększ ocenę dla każdego dopasowania E-numeru
            for e_num in query_analysis.get("e_numbers", []):
                if e_num in all_e_numbers:
                    boost += 0.2
            
            # Sprawdź dopasowanie wariantów
            variant = match.get("query_variant", "")
            if variant != query and variant in query_analysis.get("ingredients", []):
                boost += 0.1
                
            return base_score + boost
        
        # Sortuj wyniki według rozszerzonej oceny
        for match in basic_ranked:
            match["enhanced_score"] = enhanced_score(match)
        
        return sorted(basic_ranked, key=lambda x: x["enhanced_score"], reverse=True)


# Dla zachowania kompatybilności wstecznej
class TwoLayerSearch(EnhancedTwoLayerSearch):
    """
    Kompatybilna implementacja dwuwarstwowego wyszukiwania.
    """
    pass