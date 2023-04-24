import numpy as np
import faiss

class HNSW_GPU:
    def __init__(self, data, M=16, ef_construction=200, ef_search=50, use_float16=False):
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.n_trees = 1
        self.use_float16 = use_float16
        self.res = faiss.StandardGpuResources()

        if self.use_float16:
            self.index = faiss.IndexHNSWFlat(len(data[0]), self.M, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = self.ef_construction
            self.index.hnsw.efSearch = self.ef_search
            self.index = faiss.index_cpu_to_all_gpus(self.index, self.res)
            data = data.astype(np.float16)
        else:
            self.index = faiss.GpuIndexFlatL2(self.res, len(data[0]))
            data = data.astype(np.float32)

        self.index.add(data)

    def query(self, query_point, k=1):
        if self.use_float16:
            query_point = query_point.astype(np.float16)
        else:
            query_point = query_point.astype(np.float32)

        distances, neighbors = self.index.search(query_point, k)
        return neighbors, distances

# example usage:
import time

data = np.random.randn(530000, 768)
hnsw_gpu = HNSW_GPU(data, use_float16=False)
import time
seconds = time.time() 
query_point = np.random.randn(100000, 768).astype(np.float32)
neighbors, distances = hnsw_gpu.query(query_point, k=10)
print(neighbors, distances)
print((time.time()-seconds))


