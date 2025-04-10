import asyncio
import logging
from pathlib import Path
from datetime import datetime
from openai import AsyncOpenAI
import os
import asyncpg
from dotenv import load_dotenv
from urllib.parse import urlparse

# Ładowanie zmiennych środowiskowych
load_dotenv()

# Konfiguracja logowania
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "test_postgres_search.txt"

# Konfiguracja OpenAI i połączenia do bazy
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
DATABASE_URL = os.getenv("TIMESCALE_SERVICE_URL")

# Lista 15 pytań z wcześniejszej analizy
QUESTIONS = [
    # Pytania dotyczące glukozaminy i siarczanu chondroityny
    {
        "category": "Momordica charantia i regulacja poziomu glukozy we krwi",
        "questions": [
            "Jakie są główne wnioski EFSA dotyczące skuteczności spożycia składników żywności w kontekście utrzymania prawidłowego poziomu glukozy we krwi?",
	"Dlaczego EFSA uznała, że badania na zwierzętach i badania in vitro nie są wystarczające do wykazania wpływu składników żywności na poziom glukozy we krwi u ludzi?",
	"Jakie ograniczenia metodologiczne zostały wskazane w badaniach oceniających wpływ składników żywności na metabolizm węglowodanów?",
	"Na jakiej podstawie EFSA odrzucała związki przyczynowo-skutkowe pomiędzy spożyciem składników żywności a długoterminowym utrzymaniem prawidłowego poziomu glukozy we krwi?",
	"W jaki sposób EFSA oceniła, czy dany składnik może być stosowany jako wsparcie w utrzymaniu zdrowego poziomu cukru we krwi w populacji ogólnej?"
        ]
    }
    # Pytania dotyczące kwasu hialuronowego
    # {
    #     "category": "Kwas hialuronowy",
    #     "questions": [
    #         "Jakie właściwości fizykochemiczne kwasu hialuronowego zadecydowały o uznaniu go za dobrze scharakteryzowany składnik?",
    #         "W jaki sposób oceniano bezpieczeństwo różnych dawek kwasu hialuronowego w badaniach na zwierzętach?",
    #         "Dlaczego badania kliniczne przeprowadzone na pacjentach z osteoartrozą nie mogły być podstawą do potwierdzenia skuteczności w populacji ogólnej?",
    #         "Jakie były proponowane mechanizmy działania kwasu hialuronowego w kontekście utrzymania prawidłowej funkcji stawów?",
    #         "Dlaczego Panel EFSA odrzucił dowody na skuteczność kwasu hialuronowego w zmniejszaniu stanu zapalnego stawów?"
    #     ]
    # },
    # # Pytania dotyczące L-5-metylotetrahydrofolianu wapnia
    # {
    #     "category": "L-5-metylotetrahydrofolian wapnia",
    #     "questions": [
    #         "Jakie są główne różnice w biodostępności między L-5-MTHF-Ca a kwasem foliowym?",
    #         "W jaki sposób określono maksymalny dopuszczalny poziom spożycia L-5-MTHF-Ca i jakie było jego uzasadnienie?",
    #         "Jakie były główne zanieczyszczenia zidentyfikowane w L-5-MTHF-Ca i jak oceniono ich bezpieczeństwo?",
    #         "Na jakiej podstawie ustalono, że L-5-MTHF-Ca może być stosowany jako alternatywa dla kwasu foliowego?",
    #         "Jakie badania toksykologiczne przeprowadzono, aby potwierdzić bezpieczeństwo L-5-MTHF-Ca?"
    #     ]
    # }
]



if not DATABASE_URL:
    raise ValueError("TIMESCALE_SERVICE_URL not found in environment variables")

class PostgresVectorSearch:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.pool = None
        
    async def initialize(self):
        """Inicjalizacja puli połączeń."""
        self.pool = await asyncpg.create_pool(self.connection_string)
        
    async def close(self):
        """Zamknięcie puli połączeń."""
        if self.pool:
            await self.pool.close()
            
    async def search_similar(self, query_embedding: list, limit: int = 3) -> list:
        """Wyszukiwanie podobnych dokumentów w bazie PostgreSQL z użyciem pgvector."""
        if not self.pool:
            raise RuntimeError("Database connection not initialized")
            
        query = """
        SELECT 
            id,
            title,
            doc_metadata,
            summary,
            1 - (embedding <=> $1::vector) as similarity
        FROM documents
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> $1::vector
        LIMIT $2;
        """
        
        async with self.pool.acquire() as conn:
            # Konwersja listy na format akceptowany przez pgvector
            embedding_array = f'[{",".join(str(x) for x in query_embedding)}]'
            
            try:
                rows = await conn.fetch(query, embedding_array, limit)
            except Exception as e:
                logging.error(f"Database query error: {str(e)}")
                raise
            
            results = []
            for row in rows:
                results.append({
                    'id': row['id'],
                    'title': row['title'],
                    'doc_metadata': row['doc_metadata'],
                    'summary': row['summary'],
                    'similarity': float(row['similarity']) if row['similarity'] is not None else 0.0
                })
                
            return results

    async def test_connection(self):
        """Test połączenia z bazą."""
        if not self.pool:
            raise RuntimeError("Database connection not initialized")
            
        async with self.pool.acquire() as conn:
            version = await conn.fetchval("SELECT version();")
            doc_count = await conn.fetchval("SELECT COUNT(*) FROM documents;")
            # Test czy pgvector jest zainstalowany
            has_vector = await conn.fetchval("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector');")
            return version, doc_count, has_vector

    async def test_connection(self):
        """Test połączenia z bazą."""
        if not self.pool:
            raise RuntimeError("Database connection not initialized")
            
        async with self.pool.acquire() as conn:
            version = await conn.fetchval("SELECT version();")
            doc_count = await conn.fetchval("SELECT COUNT(*) FROM documents;")
            return version, doc_count

async def generate_embedding(text: str) -> list:
    """Generuje embedding dla tekstu używając OpenAI."""
    try:
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            encoding_format="float" 
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Error generating embedding: {str(e)}")
        raise

async def generate_answer(question: str, relevant_fragments: list) -> str:
    """Generuje odpowiedź na podstawie znalezionych fragmentów tekstu."""
    context = "\n\n".join([
        f"Fragment {i+1}:\nTytuł: {result['title']}\n"
        f"Podsumowanie: {result['summary']}\n"
        f"Metadata: {result['doc_metadata']}\n"
        for i, result in enumerate(relevant_fragments)
    ])
    
    prompt = f"""Na podstawie podanych fragmentów dokumentów odpowiedz na pytanie.
    Odpowiedź powinna być szczegółowa i bazować wyłącznie na informacjach z podanych fragmentów.
    Jeśli fragmenty nie zawierają wystarczających informacji, zaznacz to w odpowiedzi.

    Pytanie: {question}

    Kontekst:
    {context}
    """

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "Jesteś ekspertem w analizie dokumentów naukowych i regulacyjnych. Twoje odpowiedzi są precyzyjne i oparte na dostarczonych materiałach."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    
    return response.choices[0].message.content.strip()

async def run_search_test():
    # Inicjalizacja wyszukiwania
    search = PostgresVectorSearch(DATABASE_URL)
    await search.initialize()
    
    try:
        # Test połączenia
        version, doc_count = await search.test_connection()
        
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"Test wyszukiwania w bazie PostgreSQL - {datetime.now()}\n")
            f.write(f"Wersja bazy: {version}\n")
            f.write(f"Liczba dokumentów w bazie: {doc_count}\n")
            f.write("=" * 80 + "\n\n")

            for category in QUESTIONS:
                f.write(f"\nKategoria: {category['category']}\n")
                f.write("-" * 80 + "\n\n")

                for i, question in enumerate(category['questions'], 1):
                    f.write(f"Pytanie {i}: {question}\n")
                    f.write("-" * 40 + "\n")

                    try:
                        # 1. Generowanie embeddingu dla pytania
                        query_embedding = await generate_embedding(question)
                        f.write(f"Wygenerowano embedding o długości: {len(query_embedding)}\n")
                        
                        # 2. Wyszukiwanie podobnych dokumentów
                        results = await search.search_similar(query_embedding, limit=3)
                        
                        if results:
                            f.write("\nZnalezione dokumenty:\n")
                            for j, result in enumerate(results, 1):
                                f.write(f"\nDokument {j}:\n")
                                f.write(f"Tytuł: {result['title']}\n")
                                f.write(f"Podobieństwo: {result['similarity']:.2f}\n")
                                f.write(f"Metadata: {result['doc_metadata']}\n")
                                if result['summary']:
                                    f.write(f"Podsumowanie:\n{result['summary']}\n")
                            
                            # 3. Generowanie odpowiedzi
                            f.write("\nGenerowanie odpowiedzi...\n")
                            answer = await generate_answer(question, results)
                            f.write("\nODPOWIEDŹ:\n")
                            f.write("-" * 40 + "\n")
                            f.write(answer)
                            f.write("\n" + "-" * 40 + "\n")
                        else:
                            f.write("\nNie znaleziono odpowiednich dokumentów.\n")
                    
                    except Exception as e:
                        f.write(f"\nBłąd podczas przetwarzania pytania: {str(e)}\n")
                        logging.error(f"Error processing question: {str(e)}", exc_info=True)
                    
                    f.write("\n" + "=" * 80 + "\n\n")
    
    finally:
        await search.close()

async def main():
    try:
        await run_search_test()
        print(f"Test zakończony. Wyniki zapisano w pliku: {log_file}")
    except Exception as e:
        print(f"Wystąpił błąd podczas testu: {str(e)}")
        logging.error(f"Error during test execution: {str(e)}", exc_info=True)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    asyncio.run(main())