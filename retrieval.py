import os
import jieba
import numpy as np
from rank_bm25 import BM25Okapi
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import ZhipuAIEmbeddings
import faiss
from sentence_transformers import CrossEncoder


class HybridRetriever:
    def __init__(self, doc_dir="data/documents"):
        self.doc_dir = doc_dir
        self.chunks = []
        self.all_texts = []
        self.embeddings = ZhipuAIEmbeddings(
            model="embedding-2",
            api_key=os.getenv("ZHIPUAI_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/")
        )
        self.faiss_index = None
        self.bm25 = None
        # 使用轻量级 CrossEncoder 模型（已下载到本地缓存）
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', max_length=512)
        self._load_and_index()

    def _load_and_index(self):
        """加载文档、分块、建立 FAISS 和 BM25 索引"""
        if not os.path.exists(self.doc_dir):
            os.makedirs(self.doc_dir)
            print(f"请将文档放入 {self.doc_dir} 目录")
            return

        all_texts = []
        for file in os.listdir(self.doc_dir):
            path = os.path.join(self.doc_dir, file)
            if file.endswith('.pdf'):
                loader = PyPDFLoader(path)
            elif file.endswith('.txt'):
                # 尝试多种编码
                encodings = ['utf-8', 'gbk', 'gb2312', 'latin-1']
                loaded = False
                for enc in encodings:
                    try:
                        loader = TextLoader(path, encoding=enc)
                        documents = loader.load()
                        loaded = True
                        break
                    except UnicodeDecodeError:
                        continue
                if not loaded:
                    print(f"无法读取文件 {path}，跳过")
                    continue
            else:
                continue

            if file.endswith('.txt') and loaded:
                # 已经加载过
                pass
            elif not file.endswith('.pdf'):
                # 非 PDF 非 TXT 跳过
                continue
            else:
                documents = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_documents(documents)
            self.chunks.extend(chunks)
            chunk_texts = [chunk.page_content for chunk in chunks]
            all_texts.extend(chunk_texts)

        if not all_texts:
            print("未找到可读文档，请将 .txt 或 .pdf 文件放入 data/documents/ 目录")
            return

        self.all_texts = all_texts

        # FAISS 向量索引
        vectors = self.embeddings.embed_documents(all_texts)
        vectors = np.array(vectors).astype('float32')
        self.faiss_index = faiss.IndexFlatL2(vectors.shape[1])
        self.faiss_index.add(vectors)

        # BM25 索引（需要分词）
        tokenized_chunks = [list(jieba.cut(text)) for text in all_texts]
        self.bm25 = BM25Okapi(tokenized_chunks)

    def hybrid_search(self, query: str, top_k: int = 10, alpha: float = 0.5):
        """混合检索：向量检索 + BM25，返回 (index, score) 列表"""
        if self.faiss_index is None or self.bm25 is None:
            return []

        # 向量检索
        query_vec = self.embeddings.embed_query(query)
        query_vec = np.array(query_vec).astype('float32').reshape(1, -1)
        distances, indices = self.faiss_index.search(query_vec, top_k)
        # 距离转相似度（距离越小越相似）
        vector_scores = 1 / (1 + distances[0])

        # BM25 检索
        tokenized_query = list(jieba.cut(query))
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # 取 top_k 个 BM25 结果
        top_bm25_indices = np.argsort(bm25_scores)[-top_k:][::-1]

        # 合并结果（取并集）
        all_indices = set(indices[0]) | set(top_bm25_indices)
        final_scores = {}
        for idx in all_indices:
            # 向量分数
            if idx in indices[0]:
                vec_pos = list(indices[0]).index(idx)
                vec_score = vector_scores[vec_pos]
            else:
                vec_score = 0
            # BM25 分数
            bm_score = bm25_scores[idx] if idx in top_bm25_indices else 0
            final_scores[idx] = alpha * vec_score + (1 - alpha) * bm_score

        sorted_indices = sorted(final_scores, key=final_scores.get, reverse=True)[:top_k]
        return [(idx, final_scores[idx]) for idx in sorted_indices]

    def rerank(self, query: str, passages: list, top_k: int = 3):
        """使用 CrossEncoder 重排序"""
        if not passages:
            return []
        pairs = [[query, p] for p in passages]
        scores = self.reranker.predict(pairs)
        sorted_indices = np.argsort(scores)[::-1][:top_k]
        return [passages[i] for i in sorted_indices]

    def retrieve_with_rerank(self, query: str, top_k_initial: int = 20, top_k_final: int = 3):
        """完整的检索流程：混合检索 + 重排序"""
        candidates = self.hybrid_search(query, top_k=top_k_initial)
        if not candidates:
            return []
        passages = [self.all_texts[idx] for idx, _ in candidates]
        final_passages = self.rerank(query, passages, top_k=top_k_final)
        return final_passages