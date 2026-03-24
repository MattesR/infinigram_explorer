 ## all stuff for the ipython shell I use
print('doing the imports')
import beam_search, bs_utils
from similarity_search import similarity_search
from sentence_transformers import SparseEncoder, SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from transformers import AutoTokenizer, AutoModel
from bs_utils import get_tokens_from_splade
import pandas as pd
import numpy as np
import tup
from tup import word_tup
from wordpiece_cluster import cluster_tokens, clusters_to_or_clauses
from query_pipeline import QueryPipeline
from resolve_documents import resolve_all_queries
print('populating the variables')
tokenizer,engine = beam_search.load_default_engine(with_embedding_model=False)
splade = SparseEncoder("naver/splade-cocondenser-ensembledistil")
first_query = "what makes up a community, including its definitions in various contexts like science and what it means to be a 'civilized community."
input_ids = tokenizer.encode('community')
q_query = splade.encode(first_query)
tokenizer_splade = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
probs = np.load('tup.probs.npy')
counts = np.load('tup.counts.npy')
rel_tokens = get_tokens_from_splade(q_query, tokenizer_splade)

# Initialize once
pipeline = QueryPipeline(
    splade_model=splade,
    infini_tokenizer=tokenizer,
    probs=probs,
)