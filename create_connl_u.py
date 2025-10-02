import pandas as pd 
from spacy.lang.en import English
import lrec_paper
import pickle
from tqdm import tqdm
import re
import os
DATASET_NAME = 'WebOrganizer200B'

nlp = English()
nlp.add_pipe("sentencizer")

def split_sentences(text):
    sentences = []
    paragraphs = text.split('\n') # split newline first then add all sentences in the line
    for paragraph in paragraphs:
        doc = nlp(paragraph)
        sentences.extend([sent.text.strip() for sent in doc.sents if sent.text.strip()])
    return sentences

## load all data
def get_all_data(result_file= 'WebOrganizer_sample_info.pkl', skip_existing=True):
    with open(result_file,'rb') as f:
        result = pickle.load(f)

    docs = {}
    for file in result['downloaded_paths']['documents']:
        name = file.split('/')[-1]
        name = name.split('.')[0]
        if os.path.isfile(f'{name}.connlu') and skip_existing:
            print(f'skipping {name}, exists')
            continue
        print(f'load data from {file}')
        docs[name] = lrec_paper.read_zst_jsonl(file)
    return docs

def preprocess_document(doc):
    return doc ## if I can think of any preprocessing, it'll go here. For now, splitting is done in the sentence splitter


def doc_to_connlu(doc, metadata):
    outstring = '\n'
    outstring += f'# newdoc id = {metadata["doc_id"]}\n' ## should be shardnumber_docIndex
    for metadata_key, metadata_vaule in metadata.items():
       outstring += f'# meta::{metadata_key} = {metadata_vaule}\n'
        
    sentences = split_sentences(preprocess_document(doc)) ## note preprocess_document here!
    for sentence_id, sentence in enumerate(sentences):
        outstring += f'# sent_id = {metadata["doc_id"]}_{sentence_id}\n'
        outstring += f'# text = {sentence}\n'
        outstring += '\t'.join('_' for _ in range(10)) + '\n\n'
    return outstring


def create_connlu_files(docs):
    for filename, doc_list in docs.items():
        print(f'processing {filename}')
        if os.path.isfile(f'{filename}.connlu'):
            print(f'skipping {filename}, exists')
            continue
        outstring = ''
        for document_id, doc_dict in tqdm(enumerate(doc_list), desc=f'processing {filename}', total=len(doc_list)):
            metadata = {}
            if 'WARC-Target-URI' in doc_dict['metadata']:
                metadata['WARC-Target-URI'] = doc_dict['metadata']['WARC-Target-URI']
            if 'Content-Length' in doc_dict['metadata']:
                metadata['Content-Length'] = doc_dict['metadata']['Content-Length']
            metadata['doc_id'] = f'{filename}_{document_id}'
            outstring += doc_to_connlu(doc_dict['text'], metadata=metadata)
        with open(f'{filename}.connlu', 'w') as f:
            f.write(outstring)


if __name__ == "__main__":
    docs = get_all_data()
    create_connlu_files(docs)


def docs_to_sentence_files(docs):
    for filename, doc_list in docs.items():
        file_docs = [doc['text'] for doc in tqdm(doc_list, desc=f'procesing {filename}', total=len(doc_list))]
        sentences = [{'sentenceID': f'{filename}_{id_doc}_{idx}', 'text': sentence} for id_doc, doc in tqdm(enumerate(file_docs), desc=f'go through file_docs of {filename}') for idx, sentence in enumerate(split_sentences(doc))]
        print('create df...')
        s_df = pd.DataFrame(sentences)
        print('complete!')
        s_df.to_csv(f'{filename}_sentences.csv',index=False)
        with open(f'{filename}_sentences.pickle','wb') as f:
            sentence_list = s_df.text.to_list()
            pickle.dump(sentence_list,f)
