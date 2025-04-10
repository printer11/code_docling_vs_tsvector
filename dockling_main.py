# dockling_main.py

import asyncio
from dockling_loader import DoclingLoader
from typing import Dict, List
import logging
from pathlib import Path

async def main():
    # Inicjalizacja loadera
    loader = DoclingLoader()
    
    # Pobierz listę plików do przetworzenia
    pending_files = loader.list_pending_files()
    logging.info(f"Znaleziono {len(pending_files)} plików do przetworzenia")
    
    # Przetwórz każdy plik
    for file_path in pending_files:
        logging.info(f"Przetwarzanie: {file_path}")
        result = await loader.process_file(file_path)
        
        if result['status'] == 'success':
            # Tu dodamy logikę zapisu do bazy
            logging.info(f"Pomyślnie przetworzono: {file_path}")
        else:
            logging.error(f"Błąd przetwarzania {file_path}: {result['error']}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())