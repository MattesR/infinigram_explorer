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
from recall_ceiling import compare_recall_ceiling
from query_construction import build_cnf_queries, run_queries
from llm_keyword_filter import load_faceted_keywords, STOPWORDS, load_all_expansions
from progressive_queries import build_pieces, peek_and_grab, build_combination_queries
from adaptive_queries import run_adaptive
from trec_output import load_qrels

print('populating the variables')
tokenizer,engine = beam_search.load_default_engine(with_embedding_model=False)
# splade = SparseEncoder("naver/splade-cocondenser-ensembledistil")
first_query = "what makes up a community, including its definitions in various contexts like science and what it means to be a 'civilized community."
input_ids = tokenizer.encode('community')
# q_query = splade.encode(first_query)
# tokenizer_splade = AutoTokenizer.from_pretrained("naver/splade-cocondenser-ensembledistil")
# probs = np.load('tup.probs.npy')
# counts = np.load('tup.counts.npy')
# min_splade_score = 0.3
# rel_tokens = get_tokens_from_splade(q_query, tokenizer_splade)
# Initialize once
# pipeline = QueryPipeline(
#    splade_model=splade,
#     infini_tokenizer=tokenizer,
#     probs=probs,
# )
# top_tokens = pipeline.score_tokens(rel_tokens)