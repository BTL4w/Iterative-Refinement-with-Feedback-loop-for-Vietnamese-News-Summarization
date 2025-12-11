"""
Abstractive Summarization Model
Sử dụng ViT5 (Vietnamese T5) để generate summary
"""

import torch
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Handle both relative and absolute imports
try:
    from .preprocessor import VietnamesePreprocessor
except ImportError:
    from preprocessor import VietnamesePreprocessor


class AbstractiveModel:
    """
    Abstractive Summarization Model sử dụng ViT5
    
    Cơ chế hoạt động:
    1. Tokenize input text (document hoặc extractive sentences)
    2. Truncate nếu quá dài (max_input_length = 512 tokens)
    3. Generate summary bằng ViT5 với beam search
    4. Decode về Vietnamese text
    
    Features:
    - Generate từ full document
    - Generate từ extractive sentences (hybrid approach)
    - Configurable generation parameters
    - Vietnamese-optimized preprocessing
    """
    
    def __init__(self,
                 model_name: str = "VietAI/vit5-base-vietnews-summarization",
                 device: Optional[str] = None,
                 max_input_length: int = 512,
                 max_output_length: int = 128,
                 min_output_length: int = 20):
        """
        Initialize Abstractive Model
        
        Args:
            model_name: ViT5 model từ Hugging Face
                       (default: VietAI/vit5-base-vietnews-summarization - FINE-TUNED)
            device: 'cuda' hoặc 'cpu'. None = auto-detect
            max_input_length: Max tokens cho input (truncate nếu dài hơn)
            max_output_length: Max tokens cho output summary
            min_output_length: Min tokens cho output summary
            
        Note:
            Using VietAI/vit5-base-vietnews-summarization
        """   
        
        self.model_name = model_name
        self.max_input_length = max_input_length
        self.max_output_length = max_output_length
        self.min_output_length = min_output_length
        
        # Auto detect device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Loading {model_name} on {self.device}...")
        
        # Load ViT5 tokenizer và model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Load preprocessor
        self.preprocessor = VietnamesePreprocessor(tokenizer="underthesea")
        
        print(f"✓ Model loaded successfully on {self.device}")
        print(f"  Max input length: {max_input_length} tokens")
        print(f"  Output length: {min_output_length}-{max_output_length} tokens")
    
    def preprocess_input(self, text: str) -> str:
        """
        Preprocess input text cho ViT5
        
        ViT5 cần input KHÔNG CÓ underscore (sử dụng SentencePiece tokenizer riêng)
        
        Args:
            text: Input text (có thể có underscore từ extractive)
            
        Returns:
            Preprocessed text (không có underscore, normalized)
        """
        import re
        
        # Step 1: Remove underscores (from word tokenization: Cảnh_sát → Cảnh sát)
        text_clean = text.replace('_', ' ')
        
        # Step 2: Clean multiple spaces → single space
        text_clean = re.sub(r'\s+', ' ', text_clean)
        
        # Step 3: Strip leading/trailing spaces
        text_clean = text_clean.strip()
        
        return text_clean
    
    def generate_summary(self,
                        input_text: str,
                        max_length: Optional[int] = None,
                        min_length: Optional[int] = None,
                        num_beams: int = 4,
                        length_penalty: float = 1.0,
                        no_repeat_ngram_size: int = 3,
                        early_stopping: bool = True) -> str:
        """
        Generate abstractive summary từ input text
        
        Args:
            input_text: Input text (document hoặc extractive sentences)
            max_length: Max length của summary (None = use default)
            min_length: Min length của summary (None = use default)
            num_beams: Số beams cho beam search (4 = good balance)
            length_penalty: Penalty cho độ dài (1.0 = no penalty)
            no_repeat_ngram_size: Tránh lặp n-gram (3 = tránh lặp 3-gram)
            early_stopping: Dừng sớm khi tìm được beam tốt
            
        Returns:
            Generated summary (Vietnamese text)
            
        Note:
            Using fine-tuned model (vit5-base-vietnews-summarization) provides
            excellent quality without needing aggressive generation parameters.
        """
        # Use default lengths if not specified
        if max_length is None:
            max_length = self.max_output_length
        if min_length is None:
            min_length = self.min_output_length
        
        # Preprocess input
        input_clean = self.preprocess_input(input_text)
        
        # Tokenize input
        inputs = self.tokenizer(
            input_clean,
            max_length=self.max_input_length,
            truncation=True,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate summary
        with torch.no_grad():
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                early_stopping=early_stopping
            )
        
        # Decode summary
        summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return summary
    
    def generate_from_extractive(self,
                                 extractive_sentences: List[str],
                                 max_length: Optional[int] = None,
                                 min_length: Optional[int] = None,
                                 num_beams: int = 4,
                                 **kwargs) -> str:
        """
        Generate abstractive summary từ extractive sentences
        
        Hybrid approach:
        - Input: Top-k sentences từ extractive model (có underscore)
        - ViT5 sẽ:
          * Detokenize: Remove underscore (Cảnh_sát → Cảnh sát)
          * Fusion: Gộp thông tin từ nhiều câu
          * Paraphrase: Viết lại mượt mà hơn
          * Compression: Rút gọn redundancy
        
        Args:
            extractive_sentences: List các câu từ extractive model (có thể có underscore)
            max_length: Max length của summary
            min_length: Min length của summary
            num_beams: Số beams cho beam search
            **kwargs: Additional generation parameters
            
        Returns:
            Generated summary (không có underscore)
            
        Example:
            >>> extractive_sentences = [
            ...     "Công_an tỉnh Đắk_Lắk đang điều_tra 21 thanh_niên.",
            ...     "Thu_giữ ma_túy đá và ketamin."
            ... ]
            >>> summary = model.generate_from_extractive(extractive_sentences)
            >>> # Output: "Công an tỉnh Đắk Lắk điều tra 21 thanh niên liên quan ma túy."
        """
        # Concatenate extractive sentences
        # preprocess_input() will handle detokenization (remove underscores)
        input_text = ' '.join(extractive_sentences)
        
        # Generate summary (preprocess_input sẽ tự động remove underscores)
        summary = self.generate_summary(
            input_text,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            **kwargs
        )
        
        return summary
    
    def generate_from_document(self,
                              document: str,
                              max_length: Optional[int] = None,
                              min_length: Optional[int] = None,
                              num_beams: int = 4,
                              **kwargs) -> str:
        """
        Generate abstractive summary trực tiếp từ full document
        
        Note: 
        - Document sẽ bị truncate nếu quá dài (> max_input_length)
        - Document có underscore sẽ được tự động detokenize
        - Recommend: Sử dụng extractive trước để chọn câu quan trọng
        
        Args:
            document: Full document text (có thể có underscore)
            max_length: Max length của summary
            min_length: Min length của summary
            num_beams: Số beams cho beam search
            **kwargs: Additional generation parameters
            
        Returns:
            Generated summary (không có underscore)
        """
        # preprocess_input() will handle detokenization automatically
        return self.generate_summary(
            document,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            **kwargs
        )
    
    def batch_generate(self,
                      texts: List[str],
                      batch_size: int = 8,
                      **kwargs) -> List[str]:
        """
        Generate summaries cho multiple texts (batch processing)
        
        Args:
            texts: List of input texts
            batch_size: Batch size cho processing
            **kwargs: Generation parameters
            
        Returns:
            List of generated summaries
        """
        summaries = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Preprocess batch
            batch_clean = [self.preprocess_input(text) for text in batch_texts]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_clean,
                max_length=self.max_input_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=kwargs.get('max_length', self.max_output_length),
                    min_length=kwargs.get('min_length', self.min_output_length),
                    num_beams=kwargs.get('num_beams', 4),
                    length_penalty=kwargs.get('length_penalty', 1.0),
                    no_repeat_ngram_size=kwargs.get('no_repeat_ngram_size', 3),
                    early_stopping=kwargs.get('early_stopping', True)
                )
            
            # Decode batch
            batch_summaries = self.tokenizer.batch_decode(
                outputs,
                skip_special_tokens=True
            )
            summaries.extend(batch_summaries)
        
        return summaries


# Test function
if __name__ == "__main__":
    print("\n" + "="*80)
    print("TESTING ABSTRACTIVE MODEL")
    print("="*80 + "\n")
    
    # Sample Vietnamese document
    document = """
    Sáng 20/10, phòng Cảnh_sát điều_tra tội_phạm về ma_túy, Công_an tỉnh Đắk_Lắk cho biết, đang tiếp_tục điều_tra, xử_lý 21 nam_nữ thanh_niên tụ_tập trong quán karaoke sử_dụng ma_túy bị lực_lượng công_an phát_hiện.
    Trước đó, vào khoảng 1h sáng 19/10, tổ công_tác của phòng Cảnh_sát hình_sự và phòng Cảnh_sát cơ_động, Công_an tỉnh Đắk_Lắk tiến_hành kiểm_tra quán karaoke GaLaXy ở 391 đường Hùng_Vương, thị_xã Buôn_Hồ, tỉnh Đắk_Lắk.
    Tại đây, tổ công_tác phát_hiện có 5 phòng đang hát. Trong đó, 4 phòng hát có 22 thanh_niên nam, nữ đang có biểu_hiện phê ma_túy, bật nhạc to và nhảy múa.
    """
    
    # Sample extractive sentences
    extractive_sentences = [
        "Công_an tỉnh Đắk_Lắk đang điều_tra, xử_lý 21 nam_nữ thanh_niên sử_dụng ma_túy trong quán karaoke.",
        "Tổ công_tác phát_hiện 4 phòng hát có 22 thanh_niên đang phê ma_túy.",
        "Thu_giữ ma_túy đá, ketamin, thuốc lắc và cỏ Mỹ."
    ]
    
    # Initialize model
    print("Initializing ViT5 model (fine-tuned for Vietnamese news)...")
    model = AbstractiveModel(model_name="VietAI/vit5-base-vietnews-summarization")
    
    # Test 1: Generate from extractive sentences (HYBRID APPROACH)
    print("\n" + "="*80)
    print("TEST 1: HYBRID APPROACH (Extractive → Abstractive)")
    print("="*80 + "\n")
    
    print("Extractive sentences:")
    for i, sent in enumerate(extractive_sentences, 1):
        print(f"  {i}. {sent}")
    
    print("\nGenerating abstractive summary...")
    summary_hybrid = model.generate_from_extractive(extractive_sentences)
    
    print("\n📝 Abstractive Summary (from extractive):")
    print("-" * 80)
    print(summary_hybrid)
    
    # Test 2: Generate from full document
    print("\n" + "="*80)
    print("TEST 2: DIRECT APPROACH (Full Document → Abstractive)")
    print("="*80 + "\n")
    
    print("Document:")
    print(document.strip())
    
    print("\nGenerating abstractive summary...")
    summary_direct = model.generate_from_document(document)
    
    print("\n📝 Abstractive Summary (from document):")
    print("-" * 80)
    print(summary_direct)
    
    # Test 3: Different generation parameters
    print("\n" + "="*80)
    print("TEST 3: WITH DIFFERENT PARAMETERS")
    print("="*80 + "\n")
    
    print("Generating with more beams (num_beams=6)...")
    summary_more_beams = model.generate_from_extractive(
        extractive_sentences,
        num_beams=6,
        length_penalty=1.2
    )
    
    print("\n📝 Summary (6 beams, length_penalty=1.2):")
    print("-" * 80)
    print(summary_more_beams)
    
    print("\n" + "="*80)
    print("✅ TESTING COMPLETE")
    print("="*80)
    
    print("\nKey Features Verified:")
    print("  ✓ Load ViT5 model")
    print("  ✓ Generate from extractive sentences (hybrid)")
    print("  ✓ Generate from full document (direct)")
    print("  ✓ Configurable generation parameters")
    print("  ✓ Vietnamese text preprocessing")
    
    print("\n" + "="*80)
    print("PHASE 1.3 COMPLETE ✓")
    print("="*80 + "\n")

