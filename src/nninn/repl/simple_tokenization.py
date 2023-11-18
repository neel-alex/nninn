from sklearn.cluster import KMeans


class SimpleTokenizer:
    def __init__(self, chunks, vocab_size=1024):
        self.chunks = chunks
        self.clusters = KMeans(n_clusters=vocab_size)
        self.clusters.fit(chunks)

    def encode(self, sample):
        indices = self.clusters.predict(sample)
        return indices

    def decode(self, tokens):
        return self.clusters.cluster_centers_[tokens]
