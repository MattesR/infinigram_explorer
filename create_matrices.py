from pathlib import Path
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from huggingface_hub import login
import glob




def cooccurrence_to_distance_chunked(df, chunk_size=1000):
    """
    Memory-efficient version that processes rows in chunks.
    
    Parameters:
    df : pandas DataFrame
        A d×d co-occurrence matrix
    chunk_size : int
        Number of rows to process at once (tune based on your RAM)
    
    Returns:
    pandas DataFrame
        A distance matrix
    """
    n = len(df)
    mat = df.values
    diag = np.diag(mat)
    
    # Pre-allocate output
    distance_mat = np.zeros((n, n), dtype=np.float32)
    
    # Process in chunks
    for i in range(0, n, chunk_size):
        end_i = min(i + chunk_size, n)
        
        # Get chunk of rows
        chunk = mat[i:end_i, :]
        chunk_diag = diag[i:end_i]
        
        # Calculate for this chunk
        # numerator: c_ij + c_ji for rows i:end_i
        numerator = chunk + mat.T[:, i:end_i].T
        
        # denominator: c_ii + c_jj
        denominator = chunk_diag[:, None] + diag[None, :]
        
        # Store result
        distance_mat[i:end_i, :] = 1 - (numerator / denominator)
    
    return pd.DataFrame(distance_mat, index=df.index, columns=df.columns)


def process_file(input_path, output_dir, chunk_size=1000):
    """Process a single pickle file."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    
    print(f"Processing {input_path}...")
    
    # Load co-occurrence matrix
    df = pd.DataFrame(pd.read_pickle(input_path))
    
    # Convert to distance matrix
    df_distance = cooccurrence_to_distance_chunked(df, chunk_size=chunk_size)
    
    # Create output path with same filename
    output_path = output_dir / input_path.name
    
    # Save distance matrix
    df_distance.to_pickle(output_path)
    print(f"  → Saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(
        description='Convert co-occurrence matrices to distance matrices'
    )
    parser.add_argument(
        'input_pattern',
        help='Glob pattern for input pickle files (e.g., ./pickles/**/*.pkl)'
    )
    parser.add_argument(
        'output_dir',
        help='Output directory for distance matrices'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=1000,
        help='Chunk size for processing (default: 1000)'
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all matching files using glob
    input_files = glob.glob(args.input_pattern, recursive=True)
    
    if not input_files:
        print(f"No files found matching pattern: {args.input_pattern}")
        return
    
    print(f"Found {len(input_files)} files to process\n")
    
    # Process each file
    for input_file in tqdm(input_files,desc=f'Processing {len(input_files)} files',total=len(input_files)):
        try:
            process_file(input_file, output_dir, chunk_size=args.chunk_size)
        except Exception as e:
            print(f"ERROR processing {input_file}: {e}")
    
    print(f"\nDone! Processed {len(input_files)} files")


if __name__ == '__main__':
    main()