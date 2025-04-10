import asyncio
import logging
from pathlib import Path
from datetime import datetime
from foodie_dockling_loader import FoodieDoclingLoader
from openai import AsyncOpenAI
import os

# Konfiguracja OpenAI
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Konfiguracja logowania
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "test_dockling.txt"

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
    # # Pytania dotyczące kwasu hialuronowego
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


async def generate_answer(question: str, relevant_fragments: list) -> str:
    """Generuje odpowiedź na podstawie znalezionych fragmentów tekstu."""
    
    # Przygotowanie kontekstu z fragmentów
    context = "\n\n".join([f"Fragment {i+1}:\n{result['text']}" 
                          for i, result in enumerate(relevant_fragments)])
    
    prompt = f"""Na podstawie podanych fragmentów tekstów odpowiedz na pytanie.
    Odpowiedź powinna być szczegółowa i bazować wyłącznie na informacjach z podanych fragmentów.
    Jeśli fragmenty nie zawierają wystarczających informacji, zaznacz to w odpowiedzi.

    Pytanie: {question}

    Kontekst:
    {context}
    """

    response = await client.chat.completions.create(
        model="gpt-4o",  # lub inny odpowiedni model
        messages=[
            {"role": "system", "content": "Jesteś ekspertem w analizie dokumentów naukowych i regulacyjnych. Twoje odpowiedzi są precyzyjne i oparte na dostarczonych materiałach."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    
    return response.choices[0].message.content.strip()

async def run_search_test():
    loader = FoodieDoclingLoader()
    
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Test wyszukiwania dokumentów - {datetime.now()}\n")
        f.write("=" * 80 + "\n\n")

        for category in QUESTIONS:
            f.write(f"\nKategoria: {category['category']}\n")
            f.write("-" * 80 + "\n\n")

            for i, question in enumerate(category['questions'], 1):
                f.write(f"Pytanie {i}: {question}\n")
                f.write("-" * 40 + "\n")

                try:
                    # 1. Generowanie embeddingu dla pytania
                    query_embedding = await loader._generate_embedding(question)
                    f.write(f"Wygenerowano embedding o długości: {len(query_embedding)}\n")
                    
                    # 2. Wyszukiwanie podobnych dokumentów
                    results = await loader.vector_store.search_similar(
                        query_embedding,
                        limit=3  # zwiększamy limit dla lepszego kontekstu
                    )
                    
                    if results:
                        f.write("\nZnalezione fragmenty:\n")
                        for j, result in enumerate(results, 1):
                            f.write(f"\nFragment {j}:\n")
                            f.write(f"Dokument: {result['file_name']}\n")
                            f.write(f"Podobieństwo: {result['similarity']:.2f}\n")
                            f.write(f"Typ dokumentu: {result['doc_metadata'].get('document_type', 'unknown')}\n")
                            f.write(f"Tekst:\n{result['text'][:500]}...\n")
                        
                        # 3. Generowanie odpowiedzi
                        f.write("\nGenerowanie odpowiedzi...\n")
                        answer = await generate_answer(question, results)
                        f.write("\nODPOWIEDŹ:\n")
                        f.write("-" * 40 + "\n")
                        f.write(answer)
                        f.write("\n" + "-" * 40 + "\n")
                    else:
                        f.write("\nNie znaleziono odpowiednich fragmentów.\n")
                
                except Exception as e:
                    f.write(f"\nBłąd podczas przetwarzania pytania: {str(e)}\n")
                
                f.write("\n" + "=" * 80 + "\n\n")

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