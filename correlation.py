import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from itertools import combinations
import pickle
import gc
from tqdm import tqdm
from pathlib import Path


def calculate_matrix_correlation(vec1, vec2, method='pearson'):
    """
    Compute correlation and p-value between two distance matrices.
    Uses only the upper triangle (excluding diagonal) and replaces NaNs with 1.0.
    """

    # Replace NaNs with 1.0 (max distance)
    vec1 = np.nan_to_num(vec1, nan=1.0)
    vec2 = np.nan_to_num(vec2, nan=1.0)

    if method == 'pearson':
        corr, pval = pearsonr(vec1, vec2)
    elif method == 'spearman':
        corr, pval = spearmanr(vec1, vec2)
    else:
        raise ValueError(f"Unknown method: {method}")

    return corr, pval


def extend_matrix_to_keys(matrix, all_keys):
    """
    Extend a distance matrix to include all keys, filling missing values with 1.0.
    Ensures indices are sorted.
    
    Parameters:
    matrix : pandas DataFrame
        Distance matrix to extend
    all_keys : list
        All keys that should be present
    
    Returns:
    pandas DataFrame with all keys, sorted indices
    """
    # Reindex both rows and columns to include all keys, fill with 1.0
    matrix_extended = matrix.reindex(index=all_keys, columns=all_keys, fill_value=1.0)
    
    # Sort both index and columns
    matrix_extended = matrix_extended.sort_index(axis=0).sort_index(axis=1)
    
    return matrix_extended


def compare_all_matrices_streaming(matrix_dict, method='both', partial_save_path=None, save_every=5):
    """
    Compare all pairs of distance matrices using streaming to save memory.

    Parameters
    ----------
    matrix_dict : dict
        {feature_name: file_path} pointing to pickled distance matrices
    method : str
        'pearson', 'spearman', or 'both'
    partial_save_path : str or Path, optional
        Folder to save partial results during computation
    save_every : int
        Save results after every `save_every` matrices are processed

    Returns
    -------
    dict of pandas DataFrames with correlation and p-value results
    """
    feature_names = list(matrix_dict.keys())
    n_features = len(feature_names)
    
    # Preallocate results
    results = {}
    if method in ['pearson', 'both']:
        pearson_corr = np.zeros((n_features, n_features))
        pearson_pval = np.zeros((n_features, n_features))
    if method in ['spearman', 'both']:
        spearman_corr = np.zeros((n_features, n_features))
        spearman_pval = np.zeros((n_features, n_features))
    
    # Streaming pairwise comparison
    for i, feat1 in enumerate(feature_names):
        print(f"\nLoading {feat1} ({i+1}/{n_features}) from {matrix_dict[feat1]}")
        m1 = np.load(matrix_dict[feat1])

        for j in tqdm(range(i, n_features), total=len(range(1,n_features)), desc=f'doing the thing for {feat1}'):
            feat2 = feature_names[j]
            m2 = np.load(matrix_dict[feat2])

            if method in ['pearson', 'both']:
                corr, pval = calculate_matrix_correlation(m1, m2, method='pearson')
                pearson_corr[i, j] = pearson_corr[j, i] = corr
                pearson_pval[i, j] = pearson_pval[j, i] = pval

            if method in ['spearman', 'both']:
                corr, pval = calculate_matrix_correlation(m1, m2, method='spearman')
                spearman_corr[i, j] = spearman_corr[j, i] = corr
                spearman_pval[i, j] = spearman_pval[j, i] = pval

                # free memory
                del m2
                gc.collect()
        
        # free memory
        del m1
        gc.collect()

        # optional partial saving
        if partial_save_path is not None and (i+1) % save_every == 0:
            Path(partial_save_path).mkdir(parents=True, exist_ok=True)
            if method in ['pearson', 'both']:
                pd.DataFrame(pearson_corr, index=feature_names, columns=feature_names).to_csv(
                    Path(partial_save_path)/f"pearson_corr_partial_{i+1}.csv"
                )
            if method in ['spearman', 'both']:
                pd.DataFrame(spearman_corr, index=feature_names, columns=feature_names).to_csv(
                    Path(partial_save_path)/f"spearman_corr_partial_{i+1}.csv"
                )
            print(f"  → Saved partial results after processing {i+1} matrices")

    # final results
    if method in ['pearson', 'both']:
        results['pearson_correlation'] = pd.DataFrame(pearson_corr, index=feature_names, columns=feature_names)
        results['pearson_pvalue'] = pd.DataFrame(pearson_pval, index=feature_names, columns=feature_names)
    if method in ['spearman', 'both']:
        results['spearman_correlation'] = pd.DataFrame(spearman_corr, index=feature_names, columns=feature_names)
        results['spearman_pvalue'] = pd.DataFrame(spearman_pval, index=feature_names, columns=feature_names)

    return results



def compare_all_matrices(matrices, method='both', extend=True):
    """
    Compare all pairs of distance matrices.
    
    Parameters:
    matrix_files : dict
        {feature_name: file_path}
    method : str
        'pearson', 'spearman', or 'both'
    extend : bool
        If True, extend all matrices to have matching indices
    
    Returns:
    dict of DataFrames with correlation results
    """


    
    feature_names = list(matrices.keys())
    n_features = len(feature_names)
    
    results = {}
    
    if method in ['pearson', 'both']:
        pearson_corr = np.zeros((n_features, n_features))
        pearson_pval = np.zeros((n_features, n_features))
        
        for i, feat1 in enumerate(feature_names):
            for j, feat2 in enumerate(feature_names):
                if i == j:
                    pearson_corr[i, j] = 1.0
                    pearson_pval[i, j] = 0.0
                elif i < j:
                    corr, pval = calculate_matrix_correlation(
                        matrices[feat1], matrices[feat2], method='pearson'
                    )
                    pearson_corr[i, j] = pearson_corr[j, i] = corr
                    pearson_pval[i, j] = pearson_pval[j, i] = pval
        
        results['pearson_correlation'] = pd.DataFrame(
            pearson_corr, index=feature_names, columns=feature_names
        )
        results['pearson_pvalue'] = pd.DataFrame(
            pearson_pval, index=feature_names, columns=feature_names
        )
    
    if method in ['spearman', 'both']:
        spearman_corr = np.zeros((n_features, n_features))
        spearman_pval = np.zeros((n_features, n_features))
        
        for i, feat1 in enumerate(feature_names):
            for j, feat2 in enumerate(feature_names):
                if i == j:
                    spearman_corr[i, j] = 1.0
                    spearman_pval[i, j] = 0.0
                elif i < j:
                    corr, pval = calculate_matrix_correlation(
                        matrices[feat1], matrices[feat2], method='spearman'
                    )
                    spearman_corr[i, j] = spearman_corr[j, i] = corr
                    spearman_pval[i, j] = spearman_pval[j, i] = pval
        
        results['spearman_correlation'] = pd.DataFrame(
            spearman_corr, index=feature_names, columns=feature_names
        )
        results['spearman_pvalue'] = pd.DataFrame(
            spearman_pval, index=feature_names, columns=feature_names
        )
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Calculate correlation between distance matrices'
    )
    parser.add_argument(
        'matrix_files',
        nargs='+',
        help='Paths to distance matrix pickle files (format: feature_name:path)'
    )
    parser.add_argument(
        '--method',
        choices=['pearson', 'spearman', 'both'],
        default='both',
        help='Correlation method (default: both)'
    )
    parser.add_argument(
        '--output',
        '-o',
        help='Output pickle file for results (optional)'
    )
    parser.add_argument(
        '--no-extend',
        action='store_false',
        dest='extend',
        help='Do not extend matrices to matching indices (default: extend)'
    )
    parser.add_argument(
        '--no-stream',
        action='store_false',
        dest='stream',
        help='Do not stream, but load all matrices indices (default: stream)'
    )

    args = parser.parse_args()
    
    # Parse matrix files
    print('doing the thing')
    matrix_dict = {}
    for item in args.matrix_files:
        path_obj = Path(item)
        name = path_obj.stem
        matrix_dict[name] = item

    print(f"Comparing {len(matrix_dict)} distance matrices...\n")

    # Optional extension
    if args.extend:
        with open('target_structures.pkl', 'rb') as f:
            keys = pickle.load(f)
            all_keys = sorted(keys.keys())

        print(f"\nExtending all matrices to {len(all_keys)} keys...")

        for name, filepath in matrix_dict.items():
            print(f"Loading {name} from {filepath}...")
            df = pd.read_pickle(filepath)
            orig_size = len(df)
            if isinstance(filepath, str):
                filepath = Path(filepath)
            has_all_keys = set(df.index) == set(all_keys) and set(df.columns) == set(all_keys)
            is_sorted = list(df.index) == all_keys and list(df.columns) == all_keys
            if has_all_keys and is_sorted:
                print(f"  ✅ Already contains all {len(all_keys)} keys in correct order — skipping extension and overwrite.")
                if not (filepath.parent / f'{name}.npy').exists():
                    print(f'creating np array for {name}')
                    arr = df.to_numpy()
                    upper_tri = arr[np.triu_indices_from(arr, k=1)]
                    np.save(filepath.parent / f'{name}.npy', upper_tri, allow_pickle=False)
                    print(f"  💾 Saved {filepath.parent / f'{name}.npy'} ({upper_tri.size} elements)")
                matrix_dict[name] = filepath.parent / f'{name}.npy'
                continue
            else:
                df_extended = extend_matrix_to_keys(df, all_keys)
                # Overwrite the pickle file
                df_extended.to_pickle(filepath)
                print(f"  → Overwrote {filepath}")
                if not (filepath.parent / f'{name}.npy').exists():
                    print(f'creating np array for {name}')
                    arr = df.to_numpy()
                    upper_tri = arr[np.triu_indices_from(arr, k=1)]
                    np.save(filepath.parent / f'{name}.npy', upper_tri, allow_pickle=False)
                    print(f"  💾 Saved {filepath.parent / f'{name}.npy'} ({upper_tri.size} elements)")
                matrix_dict[name] = filepath.parent / f'{name}.npy'
    ## if there's no extension, the script will assume that matrix_dict is a dict containing names and paths to NPY arrays

    # Calculate correlations
    if not args.stream:
        results = compare_all_matrices(matrix_dict, method=args.method)
    else:
        results = compare_all_matrices_streaming(matrix_dict, method=args.method)
        
    # Display results
    print("\n" + "="*60)
    for result_type, df in results.items():
        print(f"\n{result_type.upper().replace('_', ' ')}:")
        print(df.round(4))
        print()
    
    # Save if requested
    if args.output:
        output_path = Path(args.output)
        pd.to_pickle(results, output_path)
        print(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()