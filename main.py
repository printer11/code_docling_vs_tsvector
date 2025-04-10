import asyncio
import logging
from pathlib import Path
import argparse
from datetime import datetime
from dotenv import load_dotenv
import os
import sys

# Załaduj zmienne środowiskowe na samym początku
current_dir = Path(__file__).parent
env_path = current_dir / '.env'
load_dotenv(env_path)

# Debug info - tylko do sprawdzenia czy załadowało
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Sprawdź czy używać ulepszonej wersji loadera
USE_ENHANCED = "--use-enhanced" in sys.argv

try:
    if USE_ENHANCED:
        from dockling_loader_enhanced import EnhancedDoclingLoader as DoclingLoader
        print("Używam ulepszonej wersji DoclingLoader")
    else:
        from dockling_loader import DoclingLoader
        print("Używam standardowej wersji DoclingLoader")
except ImportError as e:
    print(f"Nie można zaimportować wybranego loadera: {e}. Używam standardowej wersji.")
    from dockling_loader import DoclingLoader

async def main():
    parser = argparse.ArgumentParser(description='Dockling Processing System')
    parser.add_argument('--process', action='store_true', help='Process pending documents')
    parser.add_argument('--search', type=str, help='Search query')
    parser.add_argument('--limit', type=int, default=5, help='Search results limit')
    parser.add_argument('--type-filter', type=str, help='Filter by document type')
    parser.add_argument('--use-enhanced', action='store_true', help='Use enhanced dockling loader')
    
    args = parser.parse_args()
    
    # Inicjalizacja loadera
    loader = DoclingLoader()
    
    if args.process:
        pending_files = loader.list_pending_files()
        logger.info(f"Found {len(pending_files)} files to process")
        
        for file_path in pending_files:
            logger.info(f"Processing: {file_path}")
            result = await loader.process_file(file_path)
            
            if result['status'] == 'success':
                logger.info(f"Successfully processed: {file_path}")
                logger.info(f"Document ID: {result['document_id']}")
                logger.info(f"Metadata: {result['metadata']}")
            else:
                logger.error(f"Error processing {file_path}: {result['error']}")
    
    if args.search:
        metadata_filter = {"document_type": args.type_filter} if args.type_filter else None
        results = await loader.search(args.search, limit=args.limit, 
                                    metadata_filter=metadata_filter)
        
        print(f"\nSearch results for: {args.search}")
        print("-" * 50)
        
        for idx, result in enumerate(results, 1):
            print(f"\n{idx}. File: {result['file_name']}")
            print(f"Similarity: {result['similarity']:.2f}")
            print(f"Document type: {result['doc_metadata'].get('document_type', 'unknown')}")
            print(f"Relevant text: {result['text'][:200]}...")
            print("-" * 50)

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    asyncio.run(main())