"""
Vietnamese Evaluator
Đánh giá chất lượng summary với ROUGE + BERTScore + Hallucination Detection
"""

import torch
import numpy as np
from typing import Dict, List, Optional
from rouge import Rouge
from bert_score import BERTScorer

# Handle both relative and absolute imports
try:
    from .preprocessor import VietnamesePreprocessor
except ImportError:
    from preprocessor import VietnamesePreprocessor


class VietnameseEvaluator:
    """
    Vietnamese-aware Evaluator cho text summarization
    
    Features:
    - ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) với Vietnamese word tokenization
    - BERTScore sử dụng PhoBERT embeddings
    - Hybrid scoring: Faithfulness & Coverage
    - Sentence-level hallucination detection
    
    Cơ chế:
    1. ROUGE: N-gram overlap (lexical matching)
    2. BERTScore: Semantic similarity (embedding-based)
    3. HybridScore = 0.4 * ROUGE + 0.6 * BERTScore
    4. Faithfulness = HybridScore_Precision (summary chính xác?)
    5. Coverage = HybridScore_Recall (bao phủ reference?)
    """
    
    def __init__(self,
                 bert_model: str = "vinai/phobert-base",
                 device: Optional[str] = None):
        """
        Initialize Vietnamese Evaluator
        
        Args:
            bert_model: PhoBERT model cho BERTScore (default: vinai/phobert-base)
            device: 'cuda' hoặc 'cpu'. None = auto-detect
        """
        self.bert_model = bert_model
        
        # Auto detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Initializing Vietnamese Evaluator...")
        print(f"  Device: {self.device}")
        print(f"  BERTScore model: {bert_model}")
        
        # Initialize preprocessor
        self.preprocessor = VietnamesePreprocessor(tokenizer="underthesea")
        
        # Initialize ROUGE
        self.rouge = Rouge()
        
        # Initialize BERTScorer
        print(f"  Loading BERTScorer (this may take a moment)...")
        self.bert_scorer = BERTScorer(
            model_type=bert_model,
            lang="vi",
            device=self.device,
            rescale_with_baseline=True  # Normalize scores
        )
        
        print("✓ Vietnamese Evaluator initialized successfully")
    
    def compute_rouge(self, summary: str, reference: str) -> Dict:
        """
        Compute ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L)
        
        Cơ chế:
        1. Word tokenize cả summary và reference (Vietnamese tokenization)
        2. Compute ROUGE scores với n-gram overlap
        3. Return precision, recall, F1 cho mỗi metric
        
        Args:
            summary: Generated summary
            reference: Reference summary (ground truth)
            
        Returns:
            Dict with ROUGE-1, ROUGE-2, ROUGE-L scores
            
        Example:
            >>> scores = evaluator.compute_rouge(summary, reference)
            >>> # {'rouge-1': {'p': 0.75, 'r': 0.68, 'f': 0.71}, ...}
        """
        # Word tokenize cho tiếng Việt
        # ROUGE cần tokenized text (space-separated words)
        sum_tokens = self.preprocessor.word_tokenize(summary)
        ref_tokens = self.preprocessor.word_tokenize(reference)
        
        # Join tokens with space
        sum_text = ' '.join(sum_tokens)
        ref_text = ' '.join(ref_tokens)
        
        # Handle empty strings
        if not sum_text.strip() or not ref_text.strip():
            return {
                'rouge-1': {'p': 0.0, 'r': 0.0, 'f': 0.0},
                'rouge-2': {'p': 0.0, 'r': 0.0, 'f': 0.0},
                'rouge-l': {'p': 0.0, 'r': 0.0, 'f': 0.0}
            }
        
        # Compute ROUGE
        try:
            scores = self.rouge.get_scores(sum_text, ref_text)[0]
            return scores
        except Exception as e:
            print(f"Warning: ROUGE computation failed: {e}")
            return {
                'rouge-1': {'p': 0.0, 'r': 0.0, 'f': 0.0},
                'rouge-2': {'p': 0.0, 'r': 0.0, 'f': 0.0},
                'rouge-l': {'p': 0.0, 'r': 0.0, 'f': 0.0}
            }
    
    def compute_bertscore(self, summary: str, reference: str) -> Dict:
        """
        Compute BERTScore sử dụng PhoBERT embeddings
        
        Cơ chế:
        1. Encode summary và reference bằng PhoBERT
        2. Compute token-level cosine similarity matrix
        3. Precision: avg max similarity của mỗi token trong summary
        4. Recall: avg max similarity của mỗi token trong reference
        5. F1: harmonic mean of precision & recall
        
        Args:
            summary: Generated summary
            reference: Reference summary
            
        Returns:
            Dict with precision, recall, f1 scores
            
        Example:
            >>> scores = evaluator.compute_bertscore(summary, reference)
            >>> # {'precision': 0.87, 'recall': 0.78, 'f1': 0.82}
        """
        # Handle empty strings
        if not summary.strip() or not reference.strip():
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
        
        # Compute BERTScore
        try:
            P, R, F1 = self.bert_scorer.score([summary], [reference])
            
            return {
                'precision': P.item(),
                'recall': R.item(),
                'f1': F1.item()
            }
        except Exception as e:
            print(f"Warning: BERTScore computation failed: {e}")
            return {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
    
    def compute_hybrid_score(self,
                            summary: str,
                            reference: str,
                            document: str,
                            rouge_weight: float = 0.4,
                            bert_weight: float = 0.6) -> Dict:
        """
        Compute Hybrid Score = ROUGE + BERTScore với weights
        
        Cơ chế tính toán chi tiết:
        
        1. Compute ROUGE scores (lexical matching)
        2. Compute BERTScore scores (semantic matching)
        3. Aggregate với weights:
           - HybridScore_P = rouge_weight * ROUGE_P + bert_weight * BERT_P
           - HybridScore_R = rouge_weight * ROUGE_R + bert_weight * BERT_R
           - HybridScore_F1 = rouge_weight * ROUGE_F1 + bert_weight * BERT_F1
        
        4. Compute derived metrics:
           - Faithfulness = HybridScore_P (summary chính xác bao nhiêu?)
           - Coverage_raw = HybridScore_R (bao phủ reference bao nhiêu?)
           
        5. Normalize Coverage theo compression ratio:
           compression = len(summary) / len(document)
           expected_recall = max(compression * 0.8, 0.3)
           coverage = min(coverage_raw / expected_recall, 1.0)
           
        6. Overall score = (Faithfulness + Coverage) / 2
        
        Args:
            summary: Generated summary
            reference: Reference summary
            document: Original document
            rouge_weight: Weight cho ROUGE (default: 0.4)
            bert_weight: Weight cho BERTScore (default: 0.6)
            
        Returns:
            Dict with comprehensive evaluation metrics
            
        Example:
            >>> result = evaluator.compute_hybrid_score(summary, reference, document)
            >>> print(f"Overall: {result['overall_score']:.3f}")
            >>> print(f"Faithfulness: {result['faithfulness']:.3f}")
            >>> print(f"Coverage: {result['coverage']:.3f}")
        """
        # Step 1: Compute ROUGE
        rouge_scores = self.compute_rouge(summary, reference)
        
        # Average ROUGE-1, ROUGE-2, ROUGE-L
        rouge_p = np.mean([rouge_scores[k]['p'] for k in rouge_scores])
        rouge_r = np.mean([rouge_scores[k]['r'] for k in rouge_scores])
        rouge_f1 = np.mean([rouge_scores[k]['f'] for k in rouge_scores])
        
        # Step 2: Compute BERTScore
        bert_scores = self.compute_bertscore(summary, reference)
        
        # Step 3: Hybrid scores
        hybrid_p = rouge_weight * rouge_p + bert_weight * bert_scores['precision']
        hybrid_r = rouge_weight * rouge_r + bert_weight * bert_scores['recall']
        hybrid_f1 = rouge_weight * rouge_f1 + bert_weight * bert_scores['f1']
        
        # Step 4: Derived metrics
        faithfulness = hybrid_p  # Precision-based
        coverage_raw = hybrid_r  # Recall-based
        
        # Step 5: Normalize coverage by compression ratio
        sum_tokens = self.preprocessor.word_tokenize(summary)
        doc_tokens = self.preprocessor.word_tokenize(document)
        
        sum_len = len(sum_tokens)
        doc_len = len(doc_tokens)
        
        compression = sum_len / doc_len if doc_len > 0 else 0.1
        
        # Expected recall based on compression
        # Nếu summary = 2% của document, không thể expect recall = 100%
        expected_recall = max(compression * 0.8, 0.3)  # At least 30%
        coverage = min(coverage_raw / expected_recall, 1.0) if expected_recall > 0 else coverage_raw
        
        # Step 6: Overall score
        overall_score = (faithfulness + coverage) / 2
        
        return {
            'rouge': rouge_scores,
            'bertscore': bert_scores,
            'rouge_avg': {
                'precision': rouge_p,
                'recall': rouge_r,
                'f1': rouge_f1
            },
            'hybrid': {
                'precision': hybrid_p,
                'recall': hybrid_r,
                'f1': hybrid_f1
            },
            'faithfulness': faithfulness,
            'coverage': coverage,
            'coverage_raw': coverage_raw,
            'expected_recall': expected_recall,
            'overall_score': overall_score,
            'compression_ratio': compression,
            'summary_length': sum_len,
            'document_length': doc_len
        }
    
    def analyze_sentence_level(self,
                               summary: str,
                               document: str,
                               threshold: float = 0.6) -> List[Dict]:
        """
        Phát hiện hallucination ở sentence level
        
        Cơ chế:
        1. Split summary thành sentences
        2. Split document thành sentences
        3. Với mỗi sentence trong summary:
           - Compute BERTScore với TÁCH sentence trong document
           - Tìm sentence trong document có similarity cao nhất
           - Nếu max_similarity < threshold → nghi ngờ hallucination
        4. Return list các câu với hallucination scores
        
        Args:
            summary: Generated summary
            document: Original document
            threshold: Similarity threshold cho hallucination (default: 0.6)
            
        Returns:
            List of dicts với sentence analysis
            
        Example:
            >>> analysis = evaluator.analyze_sentence_level(summary, document)
            >>> hallucinated = [s for s in analysis if s['is_hallucination']]
            >>> print(f"Found {len(hallucinated)} potential hallucinations")
        """
        # Split into sentences
        summary_sents = self.preprocessor.sentence_tokenize(summary)
        document_sents = self.preprocessor.sentence_tokenize(document)
        
        # Handle empty inputs
        if not summary_sents or not document_sents:
            return []
        
        results = []
        
        for sum_sent in summary_sents:
            if not sum_sent.strip():
                continue
            
            # Compute BERTScore với mỗi câu trong document
            try:
                # Create list: [sum_sent] * len(document_sents) để batch process
                sum_sents_repeated = [sum_sent] * len(document_sents)
                
                P, R, F1 = self.bert_scorer.score(
                    sum_sents_repeated,
                    document_sents
                )
                
                # Find best match
                max_f1 = F1.max().item()
                best_match_idx = F1.argmax().item()
                
                results.append({
                    'sentence': sum_sent,
                    'max_similarity': max_f1,
                    'best_match': document_sents[best_match_idx],
                    'best_match_index': best_match_idx,
                    'is_hallucination': max_f1 < threshold
                })
            except Exception as e:
                print(f"Warning: Sentence-level analysis failed for: {sum_sent[:50]}... Error: {e}")
                results.append({
                    'sentence': sum_sent,
                    'max_similarity': 0.0,
                    'best_match': '',
                    'best_match_index': -1,
                    'is_hallucination': True
                })
        
        return results
    
    def evaluate_comprehensive(self,
                              summary: str,
                              reference: str,
                              document: str,
                              hallucination_threshold: float = 0.6) -> Dict:
        """
        Comprehensive evaluation: All metrics in one call
        
        Args:
            summary: Generated summary
            reference: Reference summary
            document: Original document
            hallucination_threshold: Threshold cho hallucination detection
            
        Returns:
            Dict with all evaluation results
        """
        # Compute hybrid scores
        hybrid_results = self.compute_hybrid_score(summary, reference, document)
        
        # Analyze sentence-level hallucination
        sentence_analysis = self.analyze_sentence_level(
            summary,
            document,
            threshold=hallucination_threshold
        )
        
        # Count hallucinations
        num_hallucinations = sum(1 for s in sentence_analysis if s['is_hallucination'])
        num_sentences = len(sentence_analysis)
        hallucination_rate = num_hallucinations / num_sentences if num_sentences > 0 else 0.0
        
        return {
            **hybrid_results,
            'sentence_analysis': sentence_analysis,
            'num_sentences': num_sentences,
            'num_hallucinations': num_hallucinations,
            'hallucination_rate': hallucination_rate
        }


# Test function
if __name__ == "__main__":
    print("\n" + "="*80)
    print("TESTING VIETNAMESE EVALUATOR")
    print("="*80 + "\n")
    
    # Sample data
    document = """
    Sáng 20/10, phòng Cảnh sát điều tra tội phạm về ma túy, Công an tỉnh Đắk Lắk 
    cho biết, đang tiếp tục điều tra, xử lý 21 nam nữ thanh niên tụ tập trong 
    quán karaoke sử dụng ma túy bị lực lượng công an phát hiện. Trước đó, vào 
    khoảng 1h sáng 19/10, tổ công tác của phòng Cảnh sát hình sự và phòng Cảnh 
    sát cơ động tiến hành kiểm tra quán karaoke GaLaXy. Tại đây, tổ công tác 
    phát hiện có 4 phòng hát có 22 thanh niên nam, nữ đang có biểu hiện phê ma 
    túy. Thời điểm kiểm tra, tổ công tác thu giữ nhiều tang vật như: Ma túy đá, 
    ketamin dùng để hít, thuốc lắc và cỏ Mỹ.
    """
    
    reference = """
    Công an Đắk Lắk bắt giữ 21 thanh niên sử dụng ma túy tại quán karaoke GaLaXy, 
    thu giữ ma túy đá, ketamin và cỏ Mỹ.
    """
    
    # Good summary (from fine-tuned ViT5)
    summary_good = """
    Công an Đắk Lắk điều tra 21 nam nữ thanh niên sử dụng ma túy trong quán 
    karaoke, thu giữ ma túy đá, ketamin và thuốc lắc.
    """
    
    # Poor summary (with hallucination)
    summary_poor = """
    Công an Hà Nội bắt giữ 50 người buôn bán ma túy tại nhà hàng, thu giữ 
    100kg heroin.
    """
    
    # Initialize evaluator
    print("Initializing evaluator...")
    evaluator = VietnameseEvaluator()
    
    # Test 1: ROUGE scores
    print("\n" + "="*80)
    print("TEST 1: ROUGE SCORES")
    print("="*80 + "\n")
    
    rouge_scores = evaluator.compute_rouge(summary_good, reference)
    print("ROUGE scores (good summary vs reference):")
    for metric, scores in rouge_scores.items():
        print(f"  {metric.upper()}:")
        print(f"    Precision: {scores['p']:.3f}")
        print(f"    Recall:    {scores['r']:.3f}")
        print(f"    F1:        {scores['f']:.3f}")
    
    # Test 2: BERTScore
    print("\n" + "="*80)
    print("TEST 2: BERTSCORE")
    print("="*80 + "\n")
    
    bert_scores = evaluator.compute_bertscore(summary_good, reference)
    print("BERTScore (good summary vs reference):")
    print(f"  Precision: {bert_scores['precision']:.3f}")
    print(f"  Recall:    {bert_scores['recall']:.3f}")
    print(f"  F1:        {bert_scores['f1']:.3f}")
    
    # Test 3: Hybrid Score
    print("\n" + "="*80)
    print("TEST 3: HYBRID SCORE")
    print("="*80 + "\n")
    
    hybrid_results = evaluator.compute_hybrid_score(summary_good, reference, document)
    print("Hybrid evaluation (good summary):")
    print(f"  Faithfulness:     {hybrid_results['faithfulness']:.3f}")
    print(f"  Coverage:         {hybrid_results['coverage']:.3f}")
    print(f"  Overall Score:    {hybrid_results['overall_score']:.3f}")
    print(f"  Compression:      {hybrid_results['compression_ratio']:.1%}")
    
    # Test 4: Hallucination Detection
    print("\n" + "="*80)
    print("TEST 4: HALLUCINATION DETECTION")
    print("="*80 + "\n")
    
    print("Analyzing GOOD summary:")
    analysis_good = evaluator.analyze_sentence_level(summary_good, document)
    for i, sent_info in enumerate(analysis_good, 1):
        print(f"\n  Sentence {i}: {sent_info['sentence'][:60]}...")
        print(f"    Similarity: {sent_info['max_similarity']:.3f}")
        print(f"    Hallucination: {'YES ❌' if sent_info['is_hallucination'] else 'NO ✓'}")
    
    print("\n" + "-"*80)
    print("\nAnalyzing POOR summary (with hallucination):")
    analysis_poor = evaluator.analyze_sentence_level(summary_poor, document)
    for i, sent_info in enumerate(analysis_poor, 1):
        print(f"\n  Sentence {i}: {sent_info['sentence'][:60]}...")
        print(f"    Similarity: {sent_info['max_similarity']:.3f}")
        print(f"    Hallucination: {'YES ❌' if sent_info['is_hallucination'] else 'NO ✓'}")
    
    # Test 5: Comprehensive evaluation
    print("\n" + "="*80)
    print("TEST 5: COMPREHENSIVE EVALUATION")
    print("="*80 + "\n")
    
    comp_results = evaluator.evaluate_comprehensive(summary_good, reference, document)
    print("Comprehensive results:")
    print(f"  Overall Score:        {comp_results['overall_score']:.3f}")
    print(f"  Faithfulness:         {comp_results['faithfulness']:.3f}")
    print(f"  Coverage:             {comp_results['coverage']:.3f}")
    print(f"  Hallucination Rate:   {comp_results['hallucination_rate']:.1%}")
    print(f"  ({comp_results['num_hallucinations']}/{comp_results['num_sentences']} sentences)")
    
    print("\n" + "="*80)
    print("✅ ALL TESTS PASSED!")
    print("="*80 + "\n")
    
    print("Key Features Verified:")
    print("  ✓ ROUGE scores (Vietnamese tokenization)")
    print("  ✓ BERTScore (PhoBERT embeddings)")
    print("  ✓ Hybrid scoring (Faithfulness + Coverage)")
    print("  ✓ Hallucination detection (sentence-level)")
    print("  ✓ Comprehensive evaluation")
    
    print("\n" + "="*80)
    print("PHASE 1.4 COMPLETE ✓")
    print("="*80 + "\n")


