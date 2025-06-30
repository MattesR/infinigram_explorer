import pandas as pd
import ast






def olmotrace_criteria(n_gram_df, token_column='tokens'):
    """First criterium is already satisfied just form infinigram.
    2. Self Contained: period or newline only at the end of span, doesn't begin or end with incomplete words.
    3. Maximality: No subspan of another span that meets the above criteria
    """
    n_gram_df = n_gram_df.copy()
    if isinstance(n_gram_df.loc[0][token_column], str):
        print('converting token column to list')
        n_gram_df[token_column] = n_gram_df[token_column].apply(ast.literal_eval)
    token_lists = n_gram_df[token_column].to_list()
    is_legit = []
    for idx, token_list in enumerate(token_lists):
        if not token_list[0].startswith('▁') or len(token_list[0]) == 1: #len 1 means just ▁
            is_legit.append(False)
            continue
        if idx < len(token_lists) and not token_list[idx + 1][-1].startswith('▁'):
            is_legit.append(False)
            continue
        if '.' in token_list[:-1] or '<0x0A>' in token_list[:-1]:
            is_legit.append(False)
            continue
        is_legit.append(True)
    n_gram_df['is_legit'] = is_legit
    return n_gram_df
      

def is_sublist(sublist, main_list):
    """Check if sublist is a contiguous sublist of main_list.
    The sublist must be smaller than the main list, so if
    both lists are equal, sublist is not a sublist of main_list
    """
    n = len(sublist)
    if len(sublist) >= len(main_list):
        return False
    for i in range(len(main_list) - n + 1):
        if main_list[i:i+n] == sublist:
            return True
    return False

def is_sublist_of_any(sublist, list_of_lists):
    """Check if sublist exists in any of the lists in list_of_lists."""
    for lst in list_of_lists:
        if is_sublist(sublist, lst):
            return True
    return False


def get_all_sublists(df, max_list):
    