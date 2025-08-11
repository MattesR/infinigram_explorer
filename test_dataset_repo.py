from datasets import load_dataset, Features, Value
import click
import traceback

FEATURES = Features({
    "text": Value("string"),
    "added": Value("string"),
    "created": Value("string"),
    "attributes": Value("string"),
    "doc": Value("string"),
    "id": Value("string"),
    "metadata": Value("string"),
    "source": Value("string"),
    "version": Value("string"),
    "bff_contained_ngram_count_before_dedupe": Value("int64"),
    "previous_word_count": Value("int64"),
    "url": Value("string"),
    "warcinfo": Value("string"),
    "fasttext_openhermes_reddit_eli5_vs_rw_v2_bigram_200k_train_prob": Value("float64"),
    "language_id_whole_page_fasttext": {
        "en": Value("float64")
    },
})


def test_dataset_subsets(dataset_name="allenai/olmo-mix-1124", with_features=False, timeout=30):
    """
    Test all valid subsets of a dataset to check if they're streamable.
    
    Args:
        dataset_name (str): Name of the dataset to test
        features: Optional features specification for casting
        timeout (int): Timeout in seconds for each subset test
    
    Returns:
        dict: Results containing successful and failed subsets
    """
    results = {
        'successful': [],
        'failed': [],
        'total_tested': 0
    }
    
    try:
        # First, get the list of available configurations/subsets
        print(f"Getting configurations for {dataset_name}...")
        
        # Try to load dataset info to get available configs
        try:
            from datasets import get_dataset_config_names
            config_names = get_dataset_config_names(dataset_name)
        except Exception as e:
            print(f"Could not get config names automatically: {e}")
            print("You may need to specify config names manually or check the dataset page")
            return results
        
        print(f"Found {len(config_names)} configurations: {config_names}")
        
        # Test each subset/configuration
        for config_name in config_names:
            results['total_tested'] += 1
            print(f"\nTesting subset: '{config_name}'")
            
            try:
                # Load the dataset with streaming=True
                ds = load_dataset(
                    dataset_name,
                    name=config_name,
                    split="train",
                    streaming=True,
                    features=FEATURES if with_features else None
                )
                
                # Try to get the first item
                first_item = next(iter(ds))
                
                print(f"✅ SUCCESS: '{config_name}' - streamable")
                results['successful'].append({
                    'name': config_name,
                    'sample_keys': list(first_item.keys()) if first_item else None
                })
                
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                print(f"❌ FAILED: '{config_name}' - {error_type}: {error_msg}")
                
                results['failed'].append({
                    'name': config_name,
                    'error_type': error_type,
                    'error_message': error_msg,
                    'traceback': traceback.format_exc()
                })
    
    except Exception as e:
        print(f"Critical error during testing: {e}")
        traceback.print_exc()
    
    return results

def print_test_summary(results):
    """Print a summary of the test results."""
    print("\n" + "="*60)
    print("DATASET STREAMING TEST SUMMARY")
    print("="*60)
    
    total = results['total_tested']
    successful = len(results['successful'])
    failed = len(results['failed'])
    
    print(f"Total subsets tested: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/total*100:.1f}%" if total > 0 else "No tests run")
    
    if results['successful']:
        print(f"\n✅ SUCCESSFUL SUBSETS ({successful}):")
        for item in results['successful']:
            print(f"  - {item['name']}")
    
    if results['failed']:
        print(f"\n❌ FAILED SUBSETS ({failed}):")
        for item in results['failed']:
            print(f"  - {item['name']}: {item['error_type']}")
    
    print("="*60)

@click.command()
@click.option('--dataset-name', default="allenai/olmo-mix-1124", 
              help='Name of the dataset to test')
@click.option('--with-features', is_flag=True, default=False,
              help='Use FEATURES for casting (default: use dataset default schema)')
@click.option('--timeout', default=30, type=int,
              help='Timeout in seconds for each subset test')
@click.option('--save-results', is_flag=True, default=False,
              help='Save detailed results to JSON file')
def main(dataset_name, with_features, timeout, save_results):
    """Test all subsets of a dataset to check if they're streamable."""
    
    print(f"Testing dataset: {dataset_name}")
    print(f"Using features: {'Yes (FEATURES)' if with_features else 'No (default schema)'}")
    print(f"Timeout: {timeout}s")
    print("-" * 50)
    
    # Run the test
    results = test_dataset_subsets(
        dataset_name=dataset_name,
        with_features=with_features,
        timeout=timeout
    )
    
    # Print summary
    print_test_summary(results)
    
    # Save results if requested
    if save_results:
        import json
        filename = f'dataset_test_results_{dataset_name.replace("/", "_")}.json'
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {filename}")

if __name__ == "__main__":
    main()