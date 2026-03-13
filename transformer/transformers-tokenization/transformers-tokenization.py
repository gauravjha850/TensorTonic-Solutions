import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        for token in [self.pad_token,self.unk_token,self.bos_token,self.eos_token]:
            self.word_to_id[token]=self.vocab_size
            self.id_to_word[self.vocab_size]=token
            self.vocab_size +=1

        seen=set(self.word_to_id)
        for text in texts:
            for word in text.lower().split():
                if word not in seen:
                    self.word_to_id[word]=self.vocab_size
                    self.id_to_word[self.vocab_size]=word
                    self.vocab_size +=1
                    seen.add(word)
                    
            
        
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        unk_id = self.word_to_id[self.unk_token]
        return [self.word_to_id.get(w, unk_id)
                for w in text.lower().split()]
        
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        return " ".join(
            self.id_to_word.get(i, self.unk_token) for i in ids)
        
        
