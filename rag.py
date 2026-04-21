import os
from typing import List, Tuple
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import ZhipuAIEmbeddings
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
import jieba
from FlagEmbedding import FlagReranker
from loguru import logger


class RAGSystem:
    def __init__(self, docs_dir: str = "data/documents"):
        self.docs_dir = docs_dir
        self.chunks = []  # 文本块列表
        self.embeddings = ZhipuAIEmbeddings(
            model="embedding-2",
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("LLM_BASE_URL")
        )
        self.faiss_index = None
        self.bm25 = None
        self.reranker = FlagReranker('BAAI/bge-reranker-base', use_fp16=True)
        self._load_or_build()

    def _load_documents(self):
        """加载文档并分块"""
        documents = []
        for file in os.listdir(self.docs_dir):
            path = os.path.join(self.docs_dir, file)
            if file.endswith('.pdf'):
                loader = PyPDFLoader(path)
            else:
                loader = TextLoader(path, encoding='utf-8')
            documents.extend(loader.load())
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.chunks = splitter.split_documents(documents)
        # 转换为纯文本列表
        self.texts = [chunk.page_content for chunk in self.chunks]

    def _build_faiss(self):
        """构建FAISS向量索引"""
        vectors = self.embeddings.embed_documents(self.texts)
        dim = len(vectors[0])
        self.faiss_index = faiss.IndexFlatL2(dim)
        self.faiss_index.add(np.array(vectors).astype('float32'))

    def _build_bm25(self):
        """构建BM25索引"""
        tokenized = [list(jieba.cut(text)) for text in self.texts]
        self.bm25 = BM25Okapi(tokenized)

    def _load_or_build(self):
        if not os.path.exists(self.docs_dir):
            os.makedirs(self.docs_dir, exist_ok=True)
            logger.warning(f"文档目录 {self.docs_dir} 为空，请添加文档")
            return
        self._load_documents()
        if not self.texts:
            return
        self._build_faiss()
        self._build_bm25()
        logger.info(f"RAG初始化完成，共{len(self.texts)}个文档块")

    def hybrid_search(self, query: str, top_k: int = 10, alpha: float = 0.5) -> List[Tuple[int, float]]:
        """混合检索：FAISS向量 + BM25关键词"""
        # 向量检索
        query_vec = self.embeddings.embed_query(query)
        distances, indices = self.faiss_index.search(np.array([query_vec]).astype('float32'), top_k)
        vector_scores = {idx: 1 / (dist + 1e-6) for idx, dist in zip(indices[0], distances[0])}

        # BM25检索
        tokenized_query = list(jieba.cut(query))
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_top = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
        bm25_scores_dict = {i: bm25_scores[i] for i in bm25_top}

        # 合并分数（归一化）
        all_ids = set(vector_scores.keys()) | set(bm25_scores_dict.keys())
        final_scores = {}
        for idx in all_ids:
            vec_score = vector_scores.get(idx, 0)
            bm_score = bm25_scores_dict.get(idx, 0)
            final_scores[idx] = alpha * vec_score + (1 - alpha) * bm_score
        sorted_ids = sorted(final_scores, key=final_scores.get, reverse=True)[:top_k]
        return [(idx, final_scores[idx]) for idx in sorted_ids]

    def rerank(self, query: str, passages: List[str]) -> List[str]:
        """重排序"""
        pairs = [[query, p] for p in passages]
        scores = self.reranker.compute_score(pairs, normalize=True)
        sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [passages[i] for i in sorted_idx]

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """完整检索流程：混合检索 + 重排序"""
        if not self.texts:
            return ["暂无文档，请联系人工客服。"]
        hybrid_results = self.hybrid_search(query, top_k=10)
        passages = [self.texts[idx] for idx, _ in hybrid_results]
        reranked = self.rerank(query, passages)
        return reranked[:top_k]