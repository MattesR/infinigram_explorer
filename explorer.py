"""Explorer for infinigram API and n-gram related research questions.
For comparison with LLM outputs,  olmo2 1B is used locally
"""


import requests
from transformers import AutoTokenizer
import time
from tqdm import tqdm
import os 
from pathlib import Path
import json
from utils import get_token
HF_TOKEN = get_token('HF_TOKEN')

TEST_OUTPUT = """The Winter Soldier is a fictional character appearing in American comic books published by Marvel Comics. 
Originally introduced as the superhero Bucky Barnes, Captain America's sidekick during World War II, he was believed to have perished. 
However, he was later revived and brainwashed as the Winter Soldier, a Soviet assassin with a cybernetic arm. The character has had a complex and evolving storyline, including stints as Captain America himself. 
He is a central figure in Marvel's comic book universe and has been portrayed by Sebastian Stan in the Marvel Cinematic Universe, notably in films like "Captain America: The Winter Soldier." 
His story is one of redemption, as he struggles with his past and the mind control he was subjected to as the Winter Soldier."""

GPT_FACTS= [
"The Winter Soldier is a fictional character.",
"He appears in American comic books.",
"These comic books are published by Marvel Comics.",
"The character was originally introduced as Bucky Barnes.",
"Bucky Barnes was Captain America's sidekick during World War II.",
"Bucky Barnes was believed to have perished.",
"He was later revived.",
"He was brainwashed and became the Winter Soldier.",
"As the Winter Soldier, he was a Soviet assassin.",
"He has a cybernetic arm.",
"The character has had a complex and evolving storyline.",
"The Winter Soldier has taken on the role of Captain America at some point.",
"He is a central figure in Marvel’s comic book universe.",
"The character is portrayed by Sebastian Stan in the Marvel Cinematic Universe (MCU).",
'He appears in MCU films, including "Captain America: The Winter Soldier."',
"His story includes a theme of redemption.",
"He struggles with his past.",
"He struggles with the mind control he was subjected to as the Winter Soldier."
]
INDEX='v4_olmo-2-0325-32b-instruct_llama'
TOKENIZER_NAME='meta-llama/Llama-2-7b-hf'
TOKENIZER = AutoTokenizer.from_pretrained(TOKENIZER_NAME, token=HF_TOKEN)


def query_infinigram(query_type='count', string_query=False, query='', added_payload={}, index=INDEX):
    """Counts the number of occurences of 'query' in 'index'.
    :param query_type: which api endpoint to request
    :param string_query: True if you query from string, false if you query from IDS
    :param query: string or ID query
    :return:
    """
    payload = {
        'index': f'{index}',
        'query_type': query_type,
    }
    if string_query:
        payload['query'] = query
    else:
        payload['query_ids'] = query
    if added_payload:
        for k, v in added_payload.items():
            payload[k] = v
    response = requests.post('https://api.infini-gram.io/', json=payload).json()
    if 'error' in response:
        raise Exception(response['error'])
    else:
        return response


def get_all_ngrams(query=TEST_OUTPUT, tokenizer=TOKENIZER, min_len=2, max_len=None, olmotrace_criterion=False):
    ## first, tokenize with the llama tokenizer
    tokens = tokenizer(query, add_special_tokens=False)['input_ids']
    ## ChatGPT-Code
    n = len(tokens)
    if not max_len:
        max_len = n
    max_len+=1
    if max_len < min_len:
        raise Exception('max_len must be larger than min_len')
    result = []
    for length in range(min_len, max_len):
        for start in range(n - length + 1):
            sublist = tokens[start:start + length]
            if olmotrace_criterion:
                tokens=tokenizer.decode(sublist)
                
            result.append(sublist)
    return result


def get_all_docs(query, string_query=True, index=INDEX, delay=0.2, max_docs=1000000, output_dir=None, overwrite=True, dry=False):
    find = query_infinigram(query_type='find', query=query, string_query=string_query)
    if find['cnt'] > max_docs:
        print(f'there are {find["cnt"]} documents for that query, will only get {max_docs}')
    if dry:
        print(f'there are {find["cnt"]} documents for the query:\n{query}')
        return find
    responses = []
    if output_dir:
        output_dir = Path(output_dir)
        if output_dir.exists():
            if overwrite:
                print('output_dir already exists, overwriting files')
            else: 
                print('output dir exists, resuming download')
        output_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = output_dir / 'metadata'
        text_path = output_dir / 'text'
        metadata_path.mkdir(exist_ok=True)
        text_path.mkdir(exist_ok=True)

    payload = {
    'index': f'{index}',
    'query_type': 'get_doc_by_rank',
    'max_disp_len': 10000
    }
    for shard_id, shard_span in enumerate(find['segment_by_shard']):
        if len(responses) == max_docs:
            break
        for doc in tqdm(range(shard_span[0], shard_span[1]),desc='getting docs'):
            metadata_file = metadata_path / f'{shard_id}_{doc}.json'
            text_file = text_path / f'{shard_id}_{doc}.json'
            if not overwrite:
                if metadata_file.exists() and text_file.exists():
                    print(f'file {shard_id}_{doc} already exists, skipping')
                    continue
            payload['s'] = shard_id
            payload['rank'] = doc
            if string_query:
                payload['query'] = query
            else:
                payload['query_ids'] = query
            response = requests.post('https://api.infini-gram.io/', json=payload).json()
            responses.append(response)
            if output_dir:
                if 'metadata' in response.keys():
                    with metadata_file.open('w', encoding='utf-8') as f:
                        json.dump(response['metadata'], f, ensure_ascii=False, indent=2)
                else:
                    print(f'document {shard_id}_{doc} does not have metadata')
                # Save text
                if 'spans' in response.keys():
                    with text_file.open('w', encoding='utf-8') as f:
                        json.dump(response['spans'], f, ensure_ascii=False, indent=2)
                elif 'text' in response.keys():
                    print(f'document {shard_id}_{doc} has text not spans')
                    with text_file.open('w', encoding='utf-8') as f:
                        json.dump(response['text'], f, ensure_ascii=False, indent=2)
                else: 
                    print(f'document {shard_id}_{doc} does not have spans or text')
            if len(responses) == max_docs:
                return responses
    return responses
    
            


def batch_query(query_list,  string_query=False, query_type='count', added_payload={} , index=INDEX, delay = 0.1):
    responses = []
    failures = {'ids':[], 'errors': []}
    for idx, query in tqdm(enumerate(query_list)):
        response = query_infinigram(
            query_type=query_type, 
            string_query=string_query, 
            query=query, 
            added_payload=added_payload, 
            index=index)
        if 'error' in response:
            print(response['error'])
            failures['ids'].append(idx)
            failures['errors'].append(response['error'])
        else:
            responses.append(response)
        time.sleep(delay)
    return responses, failures