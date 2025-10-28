import argparse
from hf_corpus import HFCorpusBuffered
import corpus_preprocessing

def main():
    parser = argparse.ArgumentParser(description="Preprocess Hugging Face corpus into shards.")
    parser.add_argument(
        "--subset",
        type=str,
        nargs="+",
        required=True,
        help="Subset name(s) — can be a single string or a list of subsets."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory for shards. Defaults to '<subset>_shards' or joined name if multiple subsets."
    )
    parser.add_argument(
        "--split_processes",
        type=int,
        default=None,
        help="Number of processes to use for sentence splitting (None = auto / disable parallelization)."
    )
    parser.add_argument(
        "--yield_batch_size",
        type=int,
        default=1000,
        help="Batch size for dataset iteration."
    )
    parser.add_argument(
        "--max_files_per_stream",
        type=int,
        default=10,
        help="Maximum number of files to stream simultaneously per dataset batch (default: 10)."
    )
    parser.add_argument(
        "--check_dir",
        type=str,
        default='.',
        help="Where to check for existing shards (default: '.')"
    )
    parser.add_argument(
        "--batch_offset",
        type=int,
        default=0,
        help="offset to start batching from (default: 0)"
    )
    parser.add_argument(
        "--disable_caching",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable Hugging Face dataset caching (default: True). Use --no-disable_caching to enable caching."
    )

    args = parser.parse_args()

    # Derive output directory name if not provided
    subset = args.subset if len(args.subset) > 1 else args.subset[0]
    if args.out_dir is None:
        if isinstance(subset, str):
            args.out_dir = f"{subset}_shards"
        else:
            args.out_dir = f"{'_'.join(subset)}_shards"

    print(f"  Configuration:")
    print(f"  Subset(s):         {args.subset}")
    print(f"  Output directory:  {args.out_dir}")
    print(f"  Split processes:   {args.split_processes}")
    print(f"  Yield batch size:  {args.yield_batch_size}")
    print(f"  Max files per stream: {args.max_files_per_stream}")
    print(f"  Disable caching:     {args.disable_caching}")

    # Initialize the corpus
    corpus = HFCorpusBuffered(
        subset=args.subset if len(args.subset) > 1 else args.subset[0],
        yield_style="raw",
        yield_batch_size=args.yield_batch_size,
        max_files_per_stream=args.max_files_per_stream,
        disable_caching=args.disable_caching, 
        batch_offest=args.batch_offset
    )
    chunk_size = max(1000, args.yield_batch_size // args.split_processes)

    # Run the preprocessing
    corpus_preprocessing.shards_from_corpus(
        corpus,
        out_dir=args.out_dir,
        split_processes=args.split_processes,
        chunk_size=chunk_size,
        check_dir = args.check_dir,
    )

if __name__ == "__main__":
    main()