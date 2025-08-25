import pandas as pd
from explorer import TOKENIZER as tokenizer
from math import prod

DEFAULT_UNI_PROB= 'token_unigram_probabilities_olmo-mix.csv'

def get_span_probability_from_span(span, unigram_probability=None, tokenizer=tokenizer, ids=False):
    if unigram_probability is None:
        print(f'reading default unigram_prob file {DEFAULT_UNI_PROB}')
        unigram_probability = pd.read_csv(DEFAULT_UNI_PROB, index_col=0)
    if not ids:
        ids = tokenizer.encode(span, add_special_tokens=False)
    else: # means that the span is already ids, maybe that code is weird..
        ids = span
    return prod([unigram_probability[token] for token in ids])
    

def create_token_prob_table(query, probabilities, tokenizer=tokenizer, min_len=2, max_len=None, olmotrace_criterion=True):
    sublists = get_all_ngrams(query, tokenizer,min_len,max_len,olmotrace_criterion)
    ##lists_and_probs = [{'sublist': sublist, 'prob', get

    