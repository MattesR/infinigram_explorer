import json
from tqdm import tqdm


def find_doc_ids(engine, input_ids):
    """
    Find all document IDs that contain the given input_ids.

    Args:
        engine: An infini-gram engine instance.
        input_ids: Token IDs to search for.

    Returns:
        A list of unique docid strings extracted from document metadata.
    """
    result = engine.find(input_ids=input_ids)
    total = sum(end - start for start, end in result['segment_by_shard'])
    doc_ids = set()

    with tqdm(total=total, desc="Fetching doc IDs") as pbar:
        for s, (start, end) in enumerate(result['segment_by_shard']):
            for rank in range(start, end):
                doc = engine.get_doc_by_rank(s=s, rank=rank, max_disp_len=100)
                metadata = json.loads(doc['metadata'])
                doc_ids.add(metadata['metadata']['docid'])
                pbar.update(1)

    return list(doc_ids)


def find_doc_ids_cnf(engine, cnf):
    """
    Find all document IDs matching a CNF query.

    Args:
        engine: An infini-gram engine instance.
        cnf: A CNF query, e.g. [[input_ids_a], [input_ids_b]] for A AND B.

    Returns:
        A list of unique docid strings extracted from document metadata.
    """
    result = engine.find_cnf(cnf=cnf)
    total = sum(len(ptrs) for ptrs in result['ptrs_by_shard'])
    doc_ids = set()

    with tqdm(total=total, desc="Fetching doc IDs (CNF)") as pbar:
        for s, ptrs in enumerate(result['ptrs_by_shard']):
            for ptr in ptrs:
                doc = engine.get_doc_by_ptr(s=s, ptr=ptr, max_disp_len=100)
                metadata = json.loads(doc['metadata'])
                doc_ids.add(metadata['metadata']['docid'])
                pbar.update(1)

    return list(doc_ids)