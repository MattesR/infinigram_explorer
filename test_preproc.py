from word2vec import HFCorpusBuffered
import corpus_preprocessing

def test():
    corpus = HFCorpusBuffered(subset='wiki', yield_style='raw', yield_batch_size=1000)
    corpus_preprocessing.shards_from_corpus(corpus,out_dir='wiki_shard', split_processes=3)


if __name__ == "__main__":
    test()