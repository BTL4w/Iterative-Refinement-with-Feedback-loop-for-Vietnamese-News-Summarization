"""
Extractive Summarization Model
Sử dụng PhoBERT để ranking câu quan trọng trong document
"""

import numpy as np
import torch
from typing import List, Dict, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

# Handle both relative and absolute imports
try:
    from .preprocessor import VietnamesePreprocessor
except ImportError:
    from preprocessor import VietnamesePreprocessor


class ExtractiveModel:
    """
    Extractive Summarization Model sử dụng PhoBERT
    
    Cơ chế hoạt động:
    1. Tách document thành sentences
    2. Encode mỗi sentence bằng PhoBERT → embeddings
    3. Encode document = mean pooling của tất cả sentence embeddings
    4. Tính cosine similarity giữa mỗi sentence_emb và doc_emb
    5. Rank theo similarity score
       - Hỗ trợ 2 strategies: 'similarity' và 'mmr' (Maximal Marginal Relevance)
    6. Trả về top-k sentences theo thứ tự xuất hiện trong văn bản gốc
    """
    
    def __init__(self, 
                 model_name: str = "vinai/phobert-base",
                 device: Optional[str] = None,
                 max_length: int = 256):
        """
        Args:
            model_name: Model PhoBERT từ Hugging Face
            device: 'cuda' hoặc 'cpu'. Nếu None thì tự động detect
            max_length: Max length cho tokenizer (câu quá dài sẽ truncate)
        """
        self.model_name = model_name
        self.max_length = max_length
        
        # Auto detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Loading {model_name} on {self.device}...")
        
        # Load PhoBERT tokenizer và model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Load preprocessor
        self.preprocessor = VietnamesePreprocessor(tokenizer="underthesea")
        
        print(f"✓ Model loaded successfully on {self.device}")
    
    def encode_sentence(self, sentence: str) -> np.ndarray:
        """
        Encode một câu thành embedding vector
        
        Args:
            sentence: Câu đầu vào (đã được word tokenized với underscore)
            
        Returns:
            Embedding vector (768-dim cho PhoBERT-base)
        """
        # Tokenize
        inputs = self.tokenizer(
            sentence,
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding=True
        ).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Mean pooling: lấy trung bình của last hidden state
        # Shape: (batch_size, seq_len, hidden_size) -> (batch_size, hidden_size)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        
        # Convert to numpy
        embedding = embeddings.cpu().numpy()[0]
        
        return embedding
    
    def encode_sentences(self, sentences: List[str]) -> np.ndarray:
        """
        Encode nhiều câu cùng lúc (batch processing)
        
        Args:
            sentences: List các câu (đã word tokenized)
            
        Returns:
            Matrix of embeddings (n_sentences, embedding_dim)
        """
        if len(sentences) == 0:
            return np.array([])
        
        embeddings = []
        
        # Process từng câu (có thể optimize thành batch sau)
        for sent in sentences:
            emb = self.encode_sentence(sent)
            embeddings.append(emb)
        
        return np.array(embeddings)
    
    def compute_document_embedding(self, sentence_embeddings: np.ndarray) -> np.ndarray:
        """
        Tính document embedding bằng MEAN POOLING của tất cả sentence embeddings
        
        Thay vì encode toàn bộ document (có thể quá dài), ta:
        1. Encode từng câu riêng biệt
        2. Lấy trung bình cộng (mean pooling) của tất cả sentence embeddings
        
        Args:
            sentence_embeddings: Matrix (n_sentences, embedding_dim)
            
        Returns:
            Document embedding vector (embedding_dim,)
        """
        # Mean pooling
        doc_embedding = np.mean(sentence_embeddings, axis=0)
        return doc_embedding
    
    def compute_position_weights(self, n_sentences: int, 
                                 strategy: str = 'inverse_pyramid') -> np.ndarray:
        """
        Tính position weights để giảm bias cho câu đầu/cuối
        
        Strategies:
        - 'inverse_pyramid': Câu giữa có weight cao nhất
        - 'linear_decay': Giảm dần từ đầu đến cuối
        - 'uniform': Không có position bias (weight = 1)
        
        Args:
            n_sentences: Số câu trong document
            strategy: Loại position weighting
            
        Returns:
            Array of weights (n_sentences,)
        """
        if strategy == 'uniform':
            return np.ones(n_sentences)
        
        elif strategy == 'inverse_pyramid':
            # Inverted pyramid: câu đầu có weight thấp, giữa cao, cuối trung bình
            # Pattern: [0.6, 0.8, 1.0, 1.0, 1.0, 0.9, 0.8]
            weights = np.ones(n_sentences)
            
            # Reduce weight for first sentence (lead bias)
            if n_sentences >= 1:
                weights[0] = 0.6  # First sentence penalty
            
            # Boost middle sentences
            if n_sentences >= 3:
                start_boost = max(1, n_sentences // 4)
                end_boost = min(n_sentences - 1, 3 * n_sentences // 4)
                weights[start_boost:end_boost] = 1.0
            
            # Slight reduction for last sentences
            if n_sentences >= 2:
                weights[-1] = 0.8
            if n_sentences >= 3:
                weights[-2] = 0.9
                
            return weights
        
        elif strategy == 'linear_decay':
            # Linear decay: giảm dần từ đầu đến cuối
            # Pattern: [1.0, 0.9, 0.8, 0.7, ...]
            weights = np.linspace(1.0, 0.5, n_sentences)
            return weights
        
        else:
            raise ValueError(f"Unknown position strategy: {strategy}")
    
    def rank_sentences_textrank(self, 
                                sentence_embeddings: np.ndarray,
                                damping: float = 0.85,
                                max_iter: int = 100,
                                tol: float = 1e-6) -> np.ndarray:
        """
        TextRank algorithm - Graph-based ranking
        
        Ý tưởng: Câu quan trọng = câu có nhiều liên kết với câu khác (centrality)
        
        Algorithm:
        1. Build similarity graph giữa các câu
        2. Run PageRank algorithm
        3. Sentences với PageRank score cao = important
        
        Khác với similarity-based:
        - Similarity: So sánh với document embedding
        - TextRank: Xét mối quan hệ giữa các câu với nhau
        
        Args:
            sentence_embeddings: Matrix (n_sentences, embedding_dim)
            damping: Damping factor cho PageRank (default 0.85)
            max_iter: Max iterations
            tol: Convergence tolerance
            
        Returns:
            TextRank scores (n_sentences,)
        """
        n_sentences = len(sentence_embeddings)
        
        if n_sentences == 0:
            return np.array([])
        
        if n_sentences == 1:
            return np.ones(1)
        
        # Step 1: Compute similarity matrix
        sim_matrix = cosine_similarity(sentence_embeddings)
        
        # Remove self-loops (diagonal = 0)
        np.fill_diagonal(sim_matrix, 0)
        
        # Step 2: Normalize to create transition matrix
        # Avoid division by zero
        row_sums = sim_matrix.sum(axis=1)
        row_sums[row_sums == 0] = 1  # Prevent division by zero
        
        # Normalize each row to sum to 1 (stochastic matrix)
        transition_matrix = sim_matrix / row_sums[:, np.newaxis]
        
        # Step 3: PageRank iteration
        scores = np.ones(n_sentences) / n_sentences  # Initialize uniformly
        
        for iteration in range(max_iter):
            prev_scores = scores.copy()
            
            # PageRank formula: PR(i) = (1-d)/N + d * sum(PR(j)/L(j))
            scores = (1 - damping) / n_sentences + damping * transition_matrix.T @ scores
            
            # Check convergence
            if np.abs(scores - prev_scores).sum() < tol:
                break
        
        return scores
    
    def rank_sentences_by_similarity(self,
                                     sentence_embeddings: np.ndarray,
                                     document_embedding: np.ndarray) -> np.ndarray:
        """
        Rank các câu theo cosine similarity với document embedding
        
        Args:
            sentence_embeddings: Matrix (n_sentences, embedding_dim)
            document_embedding: Vector (embedding_dim,)
            
        Returns:
            Scores array (n_sentences,) - score càng cao càng quan trọng
        """
        # Reshape document embedding cho cosine_similarity
        doc_emb_reshaped = document_embedding.reshape(1, -1)
        
        # Compute cosine similarity
        scores = cosine_similarity(sentence_embeddings, doc_emb_reshaped)
        
        # Flatten to 1D array
        scores = scores.flatten()
        
        return scores
    
    def rank_sentences_by_mmr(self,
                              sentence_embeddings: np.ndarray,
                              document_embedding: np.ndarray,
                              k: int,
                              lambda_param: float = 0.7) -> Tuple[List[int], np.ndarray]:
        """
        Maximal Marginal Relevance (MMR) - chọn câu vừa relevant vừa diverse
        
        Công thức MMR:
        MMR = arg max [λ * Sim(Si, D) - (1-λ) * max Sim(Si, Sj)]
              Si∈R\S                           Sj∈S
        
        Trong đó:
        - Si: Câu candidate
        - D: Document
        - S: Tập câu đã chọn
        - λ: Trade-off giữa relevance và diversity (0.7 = 70% relevance, 30% diversity)
        
        Args:
            sentence_embeddings: Matrix (n_sentences, embedding_dim)
            document_embedding: Vector (embedding_dim,)
            k: Số câu cần chọn
            lambda_param: Trade-off parameter (0.5 = balanced, >0.5 = prefer relevance)
            
        Returns:
            selected_indices: List indices của câu được chọn (theo thứ tự chọn)
            mmr_scores: Scores cuối cùng của các câu được chọn
        """
        n_sentences = len(sentence_embeddings)
        
        if k >= n_sentences:
            # Nếu k >= số câu, chọn tất cả
            indices = list(range(n_sentences))
            scores = self.rank_sentences_by_similarity(sentence_embeddings, document_embedding)
            return indices, scores
        
        # Step 1: Tính similarity với document cho tất cả câu
        doc_emb_reshaped = document_embedding.reshape(1, -1)
        sim_with_doc = cosine_similarity(sentence_embeddings, doc_emb_reshaped).flatten()
        
        # Step 2: Tính similarity matrix giữa các câu
        sim_matrix = cosine_similarity(sentence_embeddings, sentence_embeddings)
        
        # Initialize
        selected_indices = []
        remaining_indices = list(range(n_sentences))
        mmr_scores = []
        
        # Step 3: Chọn câu đầu tiên = câu có similarity với document cao nhất
        first_idx = np.argmax(sim_with_doc)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)
        mmr_scores.append(sim_with_doc[first_idx])
        
        # Step 4: Iteratively chọn k-1 câu còn lại
        for _ in range(k - 1):
            if len(remaining_indices) == 0:
                break
            
            best_mmr_score = -1
            best_idx = None
            
            # Với mỗi câu candidate còn lại
            for idx in remaining_indices:
                # Relevance: similarity với document
                relevance = sim_with_doc[idx]
                
                # Redundancy: max similarity với các câu đã chọn
                redundancy = max([sim_matrix[idx][selected_idx] 
                                 for selected_idx in selected_indices])
                
                # MMR score
                mmr_score = lambda_param * relevance - (1 - lambda_param) * redundancy
                
                # Update best
                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_idx = idx
            
            # Chọn câu có MMR score cao nhất
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
            mmr_scores.append(best_mmr_score)
        
        return selected_indices, np.array(mmr_scores)
    
    def rank_sentences(self, 
                      document: str, 
                      k: int = 3,
                      strategy: str = 'mmr',
                      lambda_param: float = 0.7,
                      use_position_weights: bool = True,
                      position_strategy: str = 'inverse_pyramid') -> Dict:
        """
        Rank các câu trong document và trả về top-k
        
        Args:
            document: Document gốc (chưa tokenized)
            k: Số câu muốn chọn
            strategy: 'similarity', 'mmr', hoặc 'textrank'
            lambda_param: Parameter cho MMR (nếu dùng MMR strategy)
            use_position_weights: Có áp dụng position weighting không (giảm lead bias)
            position_strategy: 'inverse_pyramid', 'linear_decay', hoặc 'uniform'
            
        Returns:
            Dict chứa:
            - 'sentences': Top-k câu (theo thứ tự xuất hiện trong document)
            - 'sentences_original': Câu gốc (không có underscore)
            - 'scores': Scores tương ứng
            - 'indices': Indices trong document gốc
            - 'strategy': Strategy được sử dụng
            - 'all_sentences': Tất cả câu (for debugging)
            - 'all_scores': Scores của tất cả câu
        """
        # Step 1: Preprocess document
        processed = self.preprocessor.preprocess_document(document)
        sentences = processed['sentences']  # Câu gốc
        sentences_tokenized = processed['sentences_tokenized']  # Câu tokenized
        
        if len(sentences) == 0:
            return {
                'sentences': [],
                'sentences_original': [],
                'scores': np.array([]),
                'indices': [],
                'strategy': strategy,
                'all_sentences': [],
                'all_scores': np.array([])
            }
        
        # Step 2: Encode tất cả sentences
        print(f"Encoding {len(sentences)} sentences...")
        sentence_embeddings = self.encode_sentences(sentences_tokenized)
        
        # Step 3: Compute document embedding = mean pooling
        document_embedding = self.compute_document_embedding(sentence_embeddings)
        
        # Step 4: Rank sentences theo strategy
        if strategy == 'similarity':
            # Simple similarity-based ranking
            scores = self.rank_sentences_by_similarity(sentence_embeddings, document_embedding)
            
            # Apply position weights if enabled
            if use_position_weights:
                position_weights = self.compute_position_weights(len(sentences), position_strategy)
                scores = scores * position_weights
            
            # Sort và lấy top-k
            ranked_indices = np.argsort(scores)[::-1]  # Descending order
            top_k_indices = ranked_indices[:k]
            top_k_scores = scores[top_k_indices]
            
        elif strategy == 'mmr':
            # MMR-based ranking
            top_k_indices, top_k_scores = self.rank_sentences_by_mmr(
                sentence_embeddings, 
                document_embedding, 
                k,
                lambda_param
            )
            
            # Compute all scores for reference
            scores = self.rank_sentences_by_similarity(sentence_embeddings, document_embedding)
            
            # Apply position weights to all scores (for visualization)
            if use_position_weights:
                position_weights = self.compute_position_weights(len(sentences), position_strategy)
                scores = scores * position_weights
            
        elif strategy == 'textrank':
            # TextRank-based ranking (graph-based)
            scores = self.rank_sentences_textrank(sentence_embeddings)
            
            # Apply position weights if enabled
            if use_position_weights:
                position_weights = self.compute_position_weights(len(sentences), position_strategy)
                scores = scores * position_weights
            
            # Sort và lấy top-k
            ranked_indices = np.argsort(scores)[::-1]  # Descending order
            top_k_indices = ranked_indices[:k]
            top_k_scores = scores[top_k_indices]
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'similarity', 'mmr', or 'textrank'")
        
        # Step 5: Sắp xếp lại theo thứ tự xuất hiện trong document gốc
        # Điều này quan trọng để ViT5 hiểu được flow của document
        sorted_pairs = sorted(zip(top_k_indices, top_k_scores), key=lambda x: x[0])
        sorted_indices = [idx for idx, _ in sorted_pairs]
        sorted_scores = np.array([score for _, score in sorted_pairs])
        
        # Get sentences
        top_sentences_tokenized = [sentences_tokenized[i] for i in sorted_indices]
        top_sentences_original = [sentences[i] for i in sorted_indices]
        
        return {
            'sentences': top_sentences_tokenized,  # Tokenized (có underscore)
            'sentences_original': top_sentences_original,  # Original (không underscore)
            'scores': sorted_scores,
            'indices': sorted_indices,
            'strategy': strategy,
            'all_sentences': sentences,
            'all_scores': scores
        }
    
    def extract_summary(self, 
                       document: str, 
                       k: int = 3,
                       strategy: str = 'mmr',
                       lambda_param: float = 0.7,
                       use_position_weights: bool = False,  # MMR works best without
                       position_strategy: str = 'inverse_pyramid',
                       return_original: bool = True) -> str:
        """
        Trả về extractive summary dạng concatenation của top-k câu
        
        Args:
            document: Document gốc
            k: Số câu cần extract
            strategy: 'similarity', 'mmr', hoặc 'textrank'
            lambda_param: Parameter cho MMR
            use_position_weights: Có áp dụng position weighting không
            position_strategy: 'inverse_pyramid', 'linear_decay', hoặc 'uniform'
            return_original: True = trả về câu gốc (không underscore),
                           False = trả về câu tokenized (có underscore)
            
        Returns:
            Summary string (concatenation của top-k câu theo thứ tự gốc)
        """
        result = self.rank_sentences(
            document, k, strategy, lambda_param,
            use_position_weights, position_strategy
        )
        
        if return_original:
            # Trả về câu gốc (không có underscore) - tốt hơn cho abstractive model
            sentences = result['sentences_original']
        else:
            # Trả về câu tokenized (có underscore)
            sentences = result['sentences']
        
        # Concatenate các câu
        summary = ' '.join(sentences)
        
        return summary
    
    def visualize_scores(self, document: str, k: int = 3, strategy: str = 'similarity'):
        """
        Visualize scores của tất cả câu (for debugging/analysis)
        
        Args:
            document: Document gốc
            k: Số câu cần chọn
            strategy: Ranking strategy
        """
        result = self.rank_sentences(document, k, strategy)
        
        print(f"\n{'='*80}")
        print(f"EXTRACTIVE RANKING VISUALIZATION")
        print(f"Strategy: {strategy.upper()}")
        print(f"Document: {len(result['all_sentences'])} sentences")
        print(f"Selected: {k} sentences")
        print(f"{'='*80}\n")
        
        selected_indices = set(result['indices'])
        
        for i, (sent, score) in enumerate(zip(result['all_sentences'], result['all_scores'])):
            marker = "★" if i in selected_indices else " "
            print(f"{marker} [{i:2d}] Score: {score:.4f} | {sent[:80]}...")
        
        print(f"\n{'='*80}")
        print("SELECTED SENTENCES (in document order):")
        print(f"{'='*80}\n")
        
        for i, (sent, score) in enumerate(zip(result['sentences_original'], result['scores'])):
            print(f"{i+1}. [Score: {score:.4f}]")
            print(f"   {sent}\n")


# Test function
if __name__ == "__main__":
    print("\n" + "="*80)
    print("TESTING EXTRACTIVE MODEL")
    print("="*80 + "\n")
    
    # Sample document (Vietnamese)
    document = """
    Ngày 27/3, Cơ_quan Cảnh_sát điều_tra Công_an TP. Hưng_Yên, tỉnh Hưng_Yên cho biết, đơn_vị vừa ra quyết_định khởi_tố vụ án, khởi_tố bị_can đối_với đối_tượng Mai_Văn_Thương (SN 1989, trú tại đội 11, thôn An_Chiểu 1, xã Liên_Phương, TP. Hưng_Yên) để điều_tra về hành_vi trộm_cắp tài_sản.
    Theo tài_liệu điều_tra của cơ_quan công_an, vào_khoảng 7h30 ngày 13/3, lợi_dụng gia_đình ông Mai_Văn_Thịnh (chú ruột đối_tượng Thương) ở cạnh nhà đi vắng, đối_tượng này đã đạp gãy chấn_song cửa_sổ, đột_nhập vào nhà ông Thịnh trộm_cắp 121kg thóc mang bán cho người cùng thôn lấy 700.000 đ.
    Không dừng lại, sau đó đối_tượng tiếp_tục quay lại lục_soát tủ nhà ông Thịnh trộm_cắp 8.500.000 đ tiền_mặt (ông Thịnh để dưới đáy tủ), rồi dùng số tiền trên để đi mua ma_tuý về sử_dụng và tiêu_xài hết 6.080.000 đ.
    Đến ngày 15/3, đối_tượng Thương đã đến Cơ_quan điều_tra Công_an TP. Hưng_Yên tự_thú và khai nhận toàn_bộ hành_vi phạm_tội của mình, đồng_thời giao_nộp cho cơ_quan công_an 3.120.000 đ.
    Hiện Công_an TP. Hưng_Yên đã thu_giữ toàn_bộ 121kg thóc đối_tượng đã trộm_cắp để trao_trả cho gia_đình ông Thịnh.
    Được biết Thương là đối_tượng nghiện ma_tuý từ nhiều năm nay, đã có 1 tiền_án về tội Tàng_trữ trái_phép chất ma_tuý bị TAND tỉnh Hưng_Yên xử_phạt 2 năm 3 tháng tù_giam.
    Ra tù năm 2016, đối_tượng này tiếp_tục có hành_vi cố_ý gây thương_tích, bị Công_an TP. Hưng_Yên ra quyết_định xử_phạt 2,5 triệu đồng.
    Vụ án đang được Công_an TP. Hưng_Yên hoàn_thiện hồ_sơ để xử_lý Mai_Văn_Thương theo quy_định của pháp_luật.
    """
    
    # Initialize model
    model = ExtractiveModel(model_name="vinai/phobert-base")
    
    # Test 1: Similarity-based ranking
    print("\n" + "="*80)
    print("TEST 1: SIMILARITY-BASED RANKING")
    print("="*80)
    
    result_sim = model.rank_sentences(document, k=3, strategy='similarity')
    print(f"\nTop-3 sentences (similarity):")
    for i, (sent, score) in enumerate(zip(result_sim['sentences_original'], result_sim['scores']), 1):
        print(f"\n{i}. [Score: {score:.4f}]")
        print(f"   {sent}")
    
    # Test 2: Extract summary
    print("\n" + "="*80)
    print("TEST 2: EXTRACT SUMMARY")
    print("="*80)
    
    summary = model.extract_summary(document, k=3, strategy='similarity')
    print(f"\nExtractive Summary:\n{summary}")
    
    # Test 3: MMR-based ranking
    print("\n" + "="*80)
    print("TEST 3: MMR-BASED RANKING")
    print("="*80)
    
    result_mmr = model.rank_sentences(document, k=3, strategy='mmr', lambda_param=0.7)
    print(f"\nTop-3 sentences (MMR with λ=0.7):")
    for i, (sent, score) in enumerate(zip(result_mmr['sentences_original'], result_mmr['scores']), 1):
        print(f"\n{i}. [Score: {score:.4f}]")
        print(f"   {sent}")
    
    # Test 4: TextRank-based ranking (NEW)
    print("\n" + "="*80)
    print("TEST 4: TEXTRANK-BASED RANKING (NEW)")
    print("="*80)
    
    result_textrank = model.rank_sentences(document, k=3, strategy='textrank')
    print(f"\nTop-3 sentences (TextRank):")
    for i, (sent, score) in enumerate(zip(result_textrank['sentences_original'], result_textrank['scores']), 1):
        print(f"\n{i}. [Score: {score:.4f}]")
        print(f"   {sent}")
    
    # Test 5: Compare with/without position weights
    print("\n" + "="*80)
    print("TEST 5: POSITION WEIGHTING COMPARISON")
    print("="*80)
    
    result_no_pos = model.rank_sentences(document, k=3, strategy='similarity', use_position_weights=False)
    result_with_pos = model.rank_sentences(document, k=3, strategy='similarity', use_position_weights=True)
    
    print("\nWithout position weights:")
    print(f"Selected indices: {result_no_pos['indices']}")
    print(f"First sentence selected: {0 in result_no_pos['indices']}")
    
    print("\nWith position weights (inverse_pyramid):")
    print(f"Selected indices: {result_with_pos['indices']}")
    print(f"First sentence selected: {0 in result_with_pos['indices']}")
    
    # Test 6: Visualize scores
    print("\n" + "="*80)
    print("TEST 6: VISUALIZE ALL SCORES")
    print("="*80)
    
    model.visualize_scores(document, k=3, strategy='similarity')
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80 + "\n")

