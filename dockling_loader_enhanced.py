# dockling_loader_enhanced.py

from dockling_loader import DoclingLoader
from typing import List, Dict, Any, Optional
import logging
import asyncio

logger = logging.getLogger(__name__)

class EnhancedDoclingLoader(DoclingLoader):
    """
    Rozszerzona wersja FoodieDoclingLoader z deduplikacją wyników wyszukiwania.
    Zachowuje wszystkie oryginalne funkcjonalności, dodając ulepszone wyszukiwanie.
    """
    
    def __init__(self):
        """Inicjalizacja wyszukiwania z deduplikacją."""
        super().__init__()  # Inicjalizacja klasy bazowej
        logger.info("Initialized EnhancedDoclingLoader with deduplication")
        
    async def search(self, query: str, limit: int = 5, 
                  metadata_filter: Optional[Dict] = None) -> List[Dict]:
        """
        Wyszukuje podobne dokumenty z deduplikacją wyników.
        
        Args:
            query: Zapytanie tekstowe
            limit: Maksymalna liczba wyników
            metadata_filter: Opcjonalny filtr metadanych
        
        Returns:
            Lista unikalnych dokumentów posortowanych według podobieństwa
        """
        logger.info(f"Enhanced search for: '{query}' with limit {limit}")
        
        # Generowanie embeddingu dla zapytania
        query_embedding = await self._generate_embedding(query)
        
        # Wyszukiwanie z większym limitem, aby mieć zapas po deduplikacji
        raw_results = await self.vector_store.search_similar(
            query_embedding, 
            limit=limit*3,  # Zwiększamy limit, aby mieć zapas
            metadata_filter=metadata_filter
        )
        
        logger.info(f"Found {len(raw_results)} raw results before deduplication")
        
        # Grupowanie wyników według nazwy pliku
        grouped_results = {}
        for result in raw_results:
            file_name = result.get('file_name')
            
            # Jeśli ten dokument nie był jeszcze widziany lub ma wyższe podobieństwo
            if file_name not in grouped_results or result['similarity'] > grouped_results[file_name]['similarity']:
                grouped_results[file_name] = result
        
        # Konwersja zgrupowanych wyników do listy
        unique_results = list(grouped_results.values())
        
        # Sortowanie wyników według podobieństwa (od najwyższego)
        unique_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Ograniczenie do żądanego limitu
        final_results = unique_results[:limit]
        
        logger.info(f"Returning {len(final_results)} deduplicated results")
        
        return final_results


# Przykład użycia bezpośrednio z tego skryptu
async def test_search():
    """Prosty test funkcjonalności wyszukiwania."""
    loader = EnhancedDoclingLoader()
    results = await loader.search("Jakie składniki opisaliśmy w dokumencie?", limit=5)
    
    print(f"\nZnaleziono {len(results)} unikalnych dokumentów:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['file_name']} (podobieństwo: {result['similarity']:.2f})")
        print(f"   Typ: {result['doc_metadata'].get('document_type', 'unknown')}")
        print(f"   Fragment: {result['text'][:150]}...\n")

    
if __name__ == "__main__":
    # Konfiguracja logowania
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Prosty test funkcjonalności
    asyncio.run(test_search())