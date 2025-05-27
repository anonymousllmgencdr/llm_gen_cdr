
import numpy as np
import faiss
from FlagEmbedding import BGEM3FlagModel

class BGE_M3(object):
    def __init__(self, BGE_PATH, device="cuda:0"):
        self.model = BGEM3FlagModel(BGE_PATH, use_fp16=True, device=device)

    def get_embeddings(self, texts, batch_size=128, max_length=512):
        embeddings = self.model.encode(texts, batch_size=batch_size, max_length=max_length, )['dense_vecs']
        return embeddings

    def get_embedding(self, text, max_length=512):
        return self.get_embeddings([text], max_length=max_length)


class RecallUtilBatch(object):
    def __init__(self, texts, bge_model, max_length=1024, batch_size=128, flat_fun="IP"):

        self.max_length = max_length
        self.embedding_func = bge_model
        embeddings = self.embedding_func.get_embeddings(texts, max_length=self.max_length, batch_size=batch_size)
        embeddings = np.array(embeddings, dtype=np.float32)
        self.embeddings = embeddings
        print(embeddings.shape)
        if flat_fun == "IP":
            emb_index = faiss.IndexFlatIP(embeddings.shape[1])
        elif flat_fun == "L2":
            emb_index = faiss.IndexFlatL2(embeddings.shape[1])

        self.emb_index = emb_index
        self.emb_index.add(embeddings)

    def search_topn(self, texts=None, embedding=None, top=10):
        if texts is not None:
            if isinstance(texts, str):
                texts = [texts]
            text_embedding = self.embedding_func.get_embeddings(texts, max_length=self.max_length)
            text_embedding = text_embedding.astype(np.float32)
        else:
            text_embedding = embedding

        base_distances, base_indices = self.emb_index.search(text_embedding, top)

        retrun_info = {
            "base_distances": base_distances,
            "base_index": base_indices,
        }

        return retrun_info
