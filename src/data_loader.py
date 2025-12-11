"""
Vietnamese News Dataset Loader
Load và preprocessing dataset cho summarization task
"""

import os
from typing import Dict, List
from datasets import load_from_disk

# Import preprocessor
try:
    from .preprocessor import VietnamesePreprocessor  # Package import
except ImportError:
    from preprocessor import VietnamesePreprocessor    # Direct import


class VietNewsDataset:
    """
    Class để load và quản lý Vietnamese news dataset
    
    Features:
    - Load từ folder sử dụng load_from_disk (datasets library)
    - Dataset có sẵn splits: train, validation, test
    - Batch loading
    - Preprocessing pipeline
    - Keys: 'guid', 'title', 'abstract', 'article'
    """
    
    def __init__(self, data_dir: str = None):
        """
        Initialize dataset loader
        
        Args:
            data_dir: Thư mục chứa data (default: auto-detect project root)
        """
        # Auto-detect project root if data_dir not provided
        if data_dir is None:
            # Get the directory of this file (src/)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up one level to project root (text_summarization_project/)
            project_root = os.path.dirname(current_dir)
            # Set data_dir to project_root/data
            data_dir = os.path.join(project_root, "data")
        
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        
        self.preprocessor = VietnamesePreprocessor()
        
        # Data storage - splits từ dataset
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
    def load_raw(self, dataset_name: str = "partial_dataset"):
        """
        Load dataset từ folder sử dụng load_from_disk
        
        Expected format:
        - Dataset với keys: 'guid', 'title', 'abstract', 'article'
        - Dataset đã có sẵn splits: train, validation, test
        
        Args:
            dataset_name: Tên folder dataset ('partial_dataset' hoặc 'full_dataset')
            
        Returns:
            DatasetDict với các splits
        """
        dataset_path = os.path.join(self.raw_dir, dataset_name)
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(
                f"Dataset path not found: {dataset_path}\n"
                f"Expected location: {os.path.abspath(dataset_path)}"
            )
        
        # Load dataset với splits có sẵn
        dataset = load_from_disk(dataset_path)
        
        # Validate format
        required_keys = {'guid', 'title', 'abstract', 'article'}
        first_sample = dataset['train'][0]
        if not required_keys.issubset(first_sample.keys()):
            raise ValueError(f"Missing required keys. Expected: {required_keys}")
        
        # Lưu splits
        self.train_data = dataset['train']
        self.val_data = dataset['validation']
        self.test_data = dataset['test']
        
        print(f"Loaded dataset from {dataset_name}:")
        print(f"  Train: {len(self.train_data)} samples")
        print(f"  Validation: {len(self.val_data)} samples")
        print(f"  Test: {len(self.test_data)} samples")
        
        return dataset
    
    def load_split(self, split: str = "validation", dataset_name: str = "partial_dataset"):
        """
        Load và return một split cụ thể với format chuẩn
        
        Tự động load raw dataset nếu chưa load, và convert sang format:
        - 'article': Nội dung bài báo
        - 'summary': Tóm tắt (abstract)
        - 'title': Tiêu đề
        - 'guid': ID
        
        Args:
            split: 'train', 'validation', hoặc 'test'
            dataset_name: Tên dataset folder
            
        Returns:
            List of dicts với keys: article, summary, title, guid
        """
        # Load dataset nếu chưa load
        if self.train_data is None:
            print(f"Loading dataset '{dataset_name}' first...")
            self.load_raw(dataset_name)
        
        # Lấy data theo split
        if split == "train":
            data = self.train_data
        elif split in ["validation", "val"]:
            data = self.val_data
        elif split == "test":
            data = self.test_data
        else:
            raise ValueError(f"Invalid split: {split}. Use 'train', 'validation', or 'test'")
        
        # Convert sang format chuẩn
        formatted_data = []
        for item in data:
            formatted_data.append({
                'article': item['article'],
                'summary': item['abstract'],  # abstract → summary
                'title': item['title'],
                'guid': item['guid']
            })
        
        print(f"Loaded {len(formatted_data)} samples from {split} split")
        return formatted_data
    
    def preprocess_dataset(self, split: str = "train") -> List[Dict]:
        """
        Preprocess toàn bộ dataset cho một split
        
        Thêm các fields:
        - article_processed: dict với 'original', 'tokenized', 'sentences'
        - abstract_processed: dict với 'original', 'tokenized'
        
        Args:
            split: Split để preprocess ('train', 'validation', hoặc 'test')
            
        Returns:
            Dataset đã preprocess
        """
        # Lấy data theo split
        if split == "train":
            data = self.train_data
        elif split in ["validation", "val"]:
            data = self.val_data
        elif split == "test":
            data = self.test_data
        else:
            raise ValueError(f"Invalid split: {split}")
            
        if data is None:
            raise ValueError(f"No data for split '{split}'. Call load_raw() first.")
        
        processed_data = []
        
        for item in data:
            processed_item = dict(item)
            
            # Preprocess article
            processed_item['article_processed'] = self.preprocessor.preprocess_document(
                item['article']
            )
            
            # Preprocess abstract (summary)
            abstract_normalized = self.preprocessor.normalize_text(item['abstract'])
            abstract_tokenized = self.preprocessor.word_tokenize(item['abstract'])
            processed_item['abstract_processed'] = {
                'original': abstract_normalized,
                'tokenized': abstract_tokenized
            }
            
            processed_data.append(processed_item)
        
        print(f"Preprocessed {len(processed_data)} samples from {split} split")
        
        return processed_data
    
    
    def get_batch(
        self, 
        split: str = "train", 
        batch_size: int = 32, 
        start_idx: int = 0
    ) -> List[Dict]:
        """
        Lấy một batch data
        
        Args:
            split: 'train', 'validation' (hoặc 'val'), hoặc 'test'
            batch_size: Kích thước batch
            start_idx: Index bắt đầu
            
        Returns:
            List of samples trong batch
        """
        if split == "train":
            data = self.train_data
        elif split in ["validation", "val"]:
            data = self.val_data
        elif split == "test":
            data = self.test_data
        else:
            raise ValueError(f"Invalid split: {split}")
        
        if data is None:
            raise ValueError(f"No data for split '{split}'. Load data first.")
        
        end_idx = min(start_idx + batch_size, len(data))
        batch = list(data[start_idx:end_idx])
        
        return batch
    
    def get_statistics(self, split: str = "train") -> Dict:
        """
        Tính statistics của dataset cho một split
        
        Args:
            split: Split để tính statistics ('train', 'validation', hoặc 'test')
        
        Returns:
            Dict chứa statistics
        """
        # Lấy data theo split
        if split == "train":
            data = self.train_data
        elif split in ["validation", "val"]:
            data = self.val_data
        elif split == "test":
            data = self.test_data
        else:
            raise ValueError(f"Invalid split: {split}")
        
        if data is None:
            raise ValueError(f"No data for split '{split}'. Call load_raw() first.")
        
        stats = {
            'split': split,
            'total_samples': len(data),
            'article_lengths': [],
            'abstract_lengths': [],
            'compression_ratios': []
        }
        
        for item in data:
            article_len = len(item['article'].split())
            abstract_len = len(item['abstract'].split())
            compression = abstract_len / article_len if article_len > 0 else 0
            
            stats['article_lengths'].append(article_len)
            stats['abstract_lengths'].append(abstract_len)
            stats['compression_ratios'].append(compression)
        
        # Compute averages
        stats['avg_article_length'] = sum(stats['article_lengths']) / len(stats['article_lengths'])
        stats['avg_abstract_length'] = sum(stats['abstract_lengths']) / len(stats['abstract_lengths'])
        stats['avg_compression_ratio'] = sum(stats['compression_ratios']) / len(stats['compression_ratios'])
        
        return stats


# Test functions
if __name__ == "__main__":
    # Test dataset loader
    dataset = VietNewsDataset()
    
    # Test loading dataset
    print("Testing dataset loading...")
    print("Note: Requires 'partial_dataset' or 'full_dataset' folder in data/raw/")
    print("Example: dataset.load_raw('partial_dataset')")
    print()
    
    # Uncomment below to test with actual data:
    dataset.load_raw('partial_dataset')
    
    # Test preprocessing
    print("Testing preprocessing...")
    processed = dataset.preprocess_dataset(split='train')
    print(f"Sample processed item keys: {processed[0].keys()}\n")
    
    # Test batch
    print("Testing batch loading...")
    batch = dataset.get_batch(split='train', batch_size=5)
    print(f"Got batch of {len(batch)} samples\n")
    
    # Test statistics
    print("Testing statistics...")
    stats = dataset.get_statistics(split='train')
    print(f"Average article length: {stats['avg_article_length']:.1f} words")
    print(f"Average abstract length: {stats['avg_abstract_length']:.1f} words")
    print(f"Average compression ratio: {stats['avg_compression_ratio']:.2%}")

