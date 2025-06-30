"""In order to use our word embeddings for similiraty search in combination with infinigram, we should tokenize the whole thing∏
"""
from transformers import AutoTokenizer
import time
from tqdm import tqdm
from pathlib import Path
import json
from gensim.test.utils import datapath
from gensim import utils
from gensim.utils import simple_preprocess
from typing import Iterator
from utils import get_token

HF_TOKEN = get_token('HF_TOKEN')
INDEX='v4_olmo-2-0325-32b-instruct_llama'
TOKENIZER_NAME='meta-llama/Llama-2-7b-hf'
TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_NAME, token=HF_TOKEN)


class MyJsonCorpus:
    """
    An iterator over a directory of JSON files.
    For each file, prep_text_file() is called, the result is lowercased,
    tokenized, and yielded as a list of tokens.
    """

    def __init__(self, dir_path):
        self.dir_path = Path(dir_path)

    def __iter__(self):
        files = list(self.dir_path.glob("*.json"))
        for file_path in tqdm(files, desc="Processing files"):
            try:
                text = self.prep_text_file(file_path)
                text = text.lower()
                yield simple_preprocess(text)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    def prep_text_file(self, file: str) -> str:
        with open(file) as f:
            json_file = json.load(f)
        if isinstance(json_file, list): # this is the output that I expect from infinigram
            text_file = ' '.join(element[0] for element in json_file)
        else:
            print(f'no list in {json_file}')
            return ''
        return text_file

    

    def search_string(self, query: str, expand=0) -> list[str] | list[tuple[str, str]]:
        """
        Return a list of filenames where the lowercase text contains the given query.
        If `expand` > 0, return a tuple of (filename, surrounding text) for the first match.
        Also prints the number of matches.
        """
        query = query.lower()
        results = []
    
        for file_path in self.dir_path.glob("*.json"):
            try:
                text = self.prep_text_file(file_path)
                text_lower = text.lower()
                match_index = text_lower.find(query)
    
                if match_index != -1:
                    if expand > 0:
                        start = max(0, match_index - expand)
                        end = min(len(text), match_index + len(query) + expand)
                        snippet = text[start:end].replace('\n', ' ').strip()
                        results.append((file_path.name, snippet))
                    else:
                        results.append(file_path.name)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
        print(f"Found {len(results)} occurrences")
        return results

