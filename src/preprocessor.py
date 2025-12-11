"""
Vietnamese Text Preprocessor
Xử lý tiền xử lý văn bản tiếng Việt cho summarization task
"""

import re
import os
import unicodedata
from typing import List, Optional, Dict
from underthesea import sent_tokenize, word_tokenize

# VNCoreNLP import (optional)
try:
    from vncorenlp import VnCoreNLP
    VNCORENLP_AVAILABLE = True
except ImportError:
    VNCORENLP_AVAILABLE = False
    print("Warning: VnCoreNLP not available. Install with: pip install vncorenlp")


class VietnamesePreprocessor:
    """
    Class xử lý preprocessing cho văn bản tiếng Việt
    
    Features:
    - Unicode normalization (NFC)
    - HTML tag removal
    - Sentence tokenization
    - Word tokenization với VNCoreNLP hoặc underthesea (với dấu gạch nối: nhà_máy)
    - Detokenization (chuyển nhà_máy -> nhà máy)
    - Preprocessing cho:
      * PhoBERT (extractive): CÓ underscore (nhà_nước)
      * ViT5 (abstractive): KHÔNG CÓ underscore (nhà nước)
    - Preprocessing cho ROUGE metrics
    """
    
    def __init__(self, 
                 tokenizer: str = "underthesea",
                 vncorenlp_path: Optional[str] = None):
        """
        Args:
            tokenizer: "underthesea" hoặc "vncorenlp"
            vncorenlp_path: Path đến VnCoreNLP jar file (nếu dùng vncorenlp)
        """
        self.html_pattern = re.compile(r'<[^>]+>')
        self.whitespace_pattern = re.compile(r'\s+')
        self.tokenizer_type = tokenizer
        
        # Initialize VnCoreNLP nếu được chọn
        self.vncorenlp = None
        if tokenizer == "vncorenlp":
            if not VNCORENLP_AVAILABLE:
                raise ImportError("VnCoreNLP not installed. Install with: pip install vncorenlp")
            if vncorenlp_path is None:
                raise ValueError("vncorenlp_path is required when using VnCoreNLP tokenizer")
            self.vncorenlp = VnCoreNLP(vncorenlp_path, annotators="wseg", max_heap_size='-Xmx500m')
    
    def __del__(self):
        """Clean up VnCoreNLP connection"""
        if self.vncorenlp is not None:
            try:
                self.vncorenlp.close()
            except:
                pass
        
    def normalize_text(self, text: str) -> str:
        """
        Chuẩn hóa văn bản
        
        Steps:
        1. Unicode normalization (NFC)
        2. Remove HTML tags
        3. Remove extra whitespaces
        4. Strip leading/trailing spaces
        
        Args:
            text: Văn bản gốc
            
        Returns:
            Văn bản đã chuẩn hóa
        """
        # Unicode normalization
        text = unicodedata.normalize('NFC', text)
        
        # Remove HTML tags
        text = self.html_pattern.sub('', text)
        
        # Remove extra whitespaces
        text = self.whitespace_pattern.sub(' ', text)
        
        # Strip
        text = text.strip()
        
        return text
    
    def sentence_tokenize(self, text: str) -> List[str]:
        """
        Tách văn bản thành các câu
        
        Sử dụng underthesea.sent_tokenize() cho tiếng Việt
        
        Args:
            text: Văn bản đầu vào
            
        Returns:
            List các câu
        """
        # Normalize trước khi tokenize
        text = self.normalize_text(text)
        
        # Sentence tokenization
        sentences = sent_tokenize(text)
        
        return sentences
    
    def word_tokenize(self, text: str) -> str:
        """
        Tách từ tiếng Việt
        
        Output format: "nhà_máy điện_lực"
        
        Args:
            text: Văn bản đầu vào
            
        Returns:
            Văn bản đã word tokenize (dạng string)
        """
        # Normalize trước
        text = self.normalize_text(text)
        
        # Word tokenization
        if self.tokenizer_type == "vncorenlp":
            if self.vncorenlp is None:
                raise RuntimeError("VnCoreNLP not initialized")
            # VnCoreNLP tokenize
            sentences = self.vncorenlp.tokenize(text)
            # Flatten and join with underscores
            tokenized = " ".join([" ".join(sent) for sent in sentences])
        else:
            # underthesea tokenize
            tokenized = word_tokenize(text, format="text")
        
        return tokenized
    
    def detokenize(self, text: str) -> str:
        """
        Chuyển văn bản đã word tokenize về dạng bình thường
        
        Input: "nhà_máy điện_lực Việt_Nam"
        Output: "nhà máy điện lực Việt Nam"
        
        Args:
            text: Văn bản đã word tokenize (có dấu gạch nối)
            
        Returns:
            Văn bản đã detokenize (không có dấu gạch nối)
        """
        return text.replace('_', ' ')
    
    def preprocess_for_rouge(self, text: str) -> str:
        """
        Preprocess văn bản cho ROUGE metrics
        
        Steps:
        1. Word tokenize
        2. Lowercase
        
        Args:
            text: Văn bản đầu vào
            
        Returns:
            Văn bản đã preprocess cho ROUGE
        """
        # Word tokenize
        tokenized = self.word_tokenize(text)
        
        # Lowercase
        tokenized = tokenized.lower()
        
        return tokenized
    
    def preprocess_document(self, text: str) -> Dict[str, any]:
        """
        Preprocess document đầy đủ cho cả PhoBERT và ViT5
        
        Args:
            text: Document gốc
            
        Returns:
            Dict chứa:
            - 'original': Văn bản normalized gốc
            - 'tokenized': Văn bản word tokenized với _ (cho PhoBERT extractive)
            - 'detokenized': Văn bản không có _ (cho ViT5 abstractive)
            - 'sentences': List các câu (dạng original)
            - 'sentences_tokenized': List các câu (dạng tokenized với _)
        """
        # Normalize
        normalized = self.normalize_text(text)
        
        # Word tokenize
        tokenized = self.word_tokenize(normalized)
        
        # Detokenize
        detokenized = self.detokenize(tokenized)
        
        # Sentence tokenize (original)
        sentences = self.sentence_tokenize(normalized)
        
        # Sentence tokenize (tokenized version)
        sentences_tokenized = [self.word_tokenize(sent) for sent in sentences]
        
        return {
            'original': normalized,  # Văn bản gốc
            'tokenized': tokenized,  # Cho PhoBERT (CÓ underscore)
            'detokenized': detokenized,  # Cho ViT5 (KHÔNG CÓ underscore)
            'sentences': sentences,  # Câu gốc
            'sentences_tokenized': sentences_tokenized  # Câu tokenized
        }
    
    def preprocess_for_model(self, 
                            text: str, 
                            model_type: str = "vit5") -> Dict[str, any]:
        """
        Preprocess văn bản cho specific model
        
        Args:
            text: Văn bản đầu vào
            model_type: "vit5" (abstractive) hoặc "phobert" (extractive)
            
        Returns:
            Dict với format phù hợp cho model
        """
        result = self.preprocess_document(text)
        
        if model_type.lower() == "vit5":
            # ViT5 cần input KHÔNG có underscore (dùng SentencePiece tokenizer riêng)
            return {
                'input_text': result['detokenized'],
                'original_text': result['original'],
                'tokenized_text': result['tokenized']
            }
        elif model_type.lower() == "phobert":
            # PhoBERT cần input CÓ underscore (pretrain với word segmentation)
            return {
                'input_text': result['tokenized'],
                'sentences': result['sentences_tokenized']
            }
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'vit5' or 'phobert'")


# Test functions
if __name__ == "__main__":
    # Test preprocessor với underthesea
    print("=" * 70)
    print("TESTING WITH UNDERTHESEA TOKENIZER")
    print("=" * 70)
    
    # NOTE: Use underthesea for dataset text (already has underscores)
    # VnCoreNLP should only be used for RAW text (no underscores yet)
    preprocessor = VietnamesePreprocessor(tokenizer="underthesea")
    
    # Sample text (giống với data example - ĐÃ CÓ underscore)
    text = """
    Ngày 27/3 , Cơ_quan Cảnh_sát điều_tra Công_an TP. Hưng_Yên , tỉnh Hưng_Yên cho biết , đơn_vị vừa ra quyết_định khởi_tố vụ án , khởi_tố bị_can đối_với đối_tượng Mai_Văn_Thương ( SN 1989 , trú tại đội 11 , thôn An_Chiểu 1 , xã Liên_Phương , TP. Hưng_Yên ) để điều_tra về hành_vi trộm_cắp tài_sản . Theo tài_liệu điều_tra của cơ_quan công_an , vào_khoảng 7h30 ngày 13/3 , lợi_dụng gia_đình ông Mai_Văn_Thịnh ( chú ruột đối_tượng Thương ) ở cạnh nhà đi vắng , đối_tượng này đã đạp gãy chấn_song cửa_sổ , đột_nhập vào nhà ông Thịnh trộm_cắp 121kg thóc mang bán cho người cùng thôn lấy 700.000 đ . Không dừng lại , sau đó đối_tượng tiếp_tục quay lại lục_soát tủ nhà ông Thịnh trộm_cắp 8.500.000 đ tiền_mặt ( ông Thịnh để dưới đáy tủ ) , rồi dùng số tiền trên để đi mua ma_tuý về sử_dụng và tiêu_xài hết 6.080.000 đ . Đến ngày 15/3 , đối_tượng Thương đã đến Cơ_quan điều_tra Công_an TP. Hưng_Yên tự_thú và khai nhận toàn_bộ hành_vi phạm_tội của mình , đồng_thời giao_nộp cho cơ_quan công_an 3.120.000 đ . Hiện Công_an TP. Hưng_Yên đã thu_giữ toàn_bộ 121kg thóc đối_tượng đã trộm_cắp để trao_trả cho gia_đình ông Thịnh . Được biết Thương là đối_tượng nghiện ma_tuý từ nhiều năm nay , đã có 1 tiền_án về tội Tàng_trữ trái_phép chất ma_tuý bị TAND tỉnh Hưng_Yên xử_phạt 2 năm 3 tháng tù_giam . Ra tù năm 2016 , đối_tượng này tiếp_tục có hành_vi cố_ý gây thương_tích , bị Công_an TP. Hưng_Yên ra quyết_định xử_phạt 2,5 triệu đồng . Vụ án đang được Công_an TP. Hưng_Yên hoàn_thiện hồ_sơ để xử_lý Mai_Văn_Thương theo quy_định của pháp_luật .  Đối_tượng Mai_Văn_Thương tại cơ_quan công_an .
    """
    
    print("\n1. ORIGINAL TEXT:")
    print(text.strip())
    print("\n" + "="*70 + "\n")
    
    # Test normalize
    normalized = preprocessor.normalize_text(text)
    print("2. NORMALIZED TEXT:")
    print(normalized)
    print("\n" + "="*70 + "\n")
    
    # Test word tokenize
    tokenized = preprocessor.word_tokenize(text)
    print("3. WORD TOKENIZED (for ViT5):")
    print(tokenized)
    print("\n" + "="*70 + "\n")
    
    # Test detokenize
    detokenized = preprocessor.detokenize(tokenized)
    print("4. DETOKENIZED:")
    print(detokenized)
    print("\n" + "="*70 + "\n")
    
    # Test sentence tokenize
    sentences = preprocessor.sentence_tokenize(text)
    print("5. SENTENCE TOKENIZATION:")
    for i, sent in enumerate(sentences, 1):
        print(f"   {i}. {sent}")
    print("\n" + "="*70 + "\n")
    
    # Test full preprocessing
    print("6. FULL PREPROCESSING:")
    result = preprocessor.preprocess_document(text)
    print(f"\n   Original (for PhoBERT):")
    print(f"   {result['original'][:100]}...")
    print(f"\n   Tokenized (for ViT5 input):")
    print(f"   {result['tokenized'][:100]}...")
    print(f"\n   Detokenized (for output):")
    print(f"   {result['detokenized'][:100]}...")
    print(f"\n   Number of sentences: {len(result['sentences'])}")
    print("\n" + "="*70 + "\n")
    
    # Test preprocessing cho từng model
    print("7. PREPROCESSING FOR SPECIFIC MODELS:")
    
    print("\n   A. For ViT5 (abstractive - NO underscore):")
    vit5_data = preprocessor.preprocess_for_model(text, model_type="vit5")
    print(f"      Input text (detokenized): {vit5_data['input_text'][:80]}...")
    print(f"      Original: {vit5_data['original_text'][:80]}...")
    
    print("\n   B. For PhoBERT (extractive - WITH underscore):")
    phobert_data = preprocessor.preprocess_for_model(text, model_type="phobert")
    print(f"      Input text (tokenized): {phobert_data['input_text'][:80]}...")
    print(f"      Sentences: {len(phobert_data['sentences'])} sentences")
    
    print("\n" + "="*70)
    print("TESTING COMPLETE")
    print("="*70)
    
    # Note about VnCoreNLP
    print("\n" + "="*70)
    print("IMPORTANT NOTES:")
    print("="*70)
    print("""
1. USE UNDERTHESEA (default):
   ✓ For dataset text that ALREADY HAS underscores
   ✓ Fast and works well
   ✓ No issues with pre-tokenized text

2. USE VnCoreNLP (optional):
   ✓ ONLY for RAW text (NO underscores yet)
   ✓ Better quality for new raw text
   ✗ DO NOT use with pre-tokenized text (creates spacing issues)
   
3. Your dataset is ALREADY TOKENIZED:
   → Use underthesea (default)
   → VnCoreNLP will break the formatting

To use VnCoreNLP for NEW raw text:
   preprocessor = VietnamesePreprocessor(
       tokenizer="vncorenlp",
       vncorenlp_path="path/to/VnCoreNLP-1.2.jar"
   )
    """)
    print("="*70)

