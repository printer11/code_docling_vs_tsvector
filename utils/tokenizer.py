# utils/tokenizer.py
from transformers import PreTrainedTokenizerBase
import tiktoken
from typing import List, Optional, Union, Dict

class OpenAITokenizerWrapper(PreTrainedTokenizerBase):
    def __init__(self):
        super().__init__()
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        # Wymagane atrybuty
        self.vocab_size = self.tokenizer.n_vocab
        self.model_max_length = 8191
        
    def tokenize(self, text: str, **kwargs) -> List[str]:
        """Zaimplementowana metoda tokenize."""
        tokens = self.tokenizer.encode(text)
        return [str(token) for token in tokens]

    def _tokenize(self, text: str) -> List[str]:
        """Zaimplementowana metoda _tokenize."""
        return self.tokenize(text)

    def _convert_token_to_id(self, token: str) -> int:
        """Konwersja tokena na id."""
        return int(token)

    def _convert_id_to_token(self, index: int) -> str:
        """Konwersja id na token."""
        return str(index)

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        """Konwersja listy tokenów na tekst."""
        return self.tokenizer.decode([int(token) for token in tokens])

    def get_vocab(self) -> Dict[str, int]:
        """Zwraca słownik tokenów."""
        return {str(i): i for i in range(self.vocab_size)}

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> tuple:
        """Zapisuje słownik."""
        return ()  # Tiktoken nie wymaga zapisywania słownika

    @property
    def vocab_size(self) -> int:
        """Rozmiar słownika."""
        return self._vocab_size

    @vocab_size.setter
    def vocab_size(self, value: int):
        self._vocab_size = value 