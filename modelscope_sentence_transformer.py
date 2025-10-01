#!/usr/bin/env python3
"""
ModelScope-compatible SentenceTransformer loader
Replaces HuggingFace model loading with ModelScope for Chinese servers
"""

import os
import sys
from sentence_transformers import SentenceTransformer
from modelscope import snapshot_download

def load_model_from_modelscope(model_path, device="cuda:0", cache_dir=None):
    """
    Load SentenceTransformer model from ModelScope instead of HuggingFace
    
    Args:
        model_path: Original HuggingFace model path (e.g., 'sentence-transformers/all-MiniLM-L6-v2')
        device: Device to load model on
        cache_dir: Cache directory for ModelScope downloads
    
    Returns:
        SentenceTransformer model loaded from ModelScope
    """
    
    # Mapping from HuggingFace to ModelScope model names
    modelscope_mapping = {
        'sentence-transformers/all-MiniLM-L6-v2': 'AI-ModelScope/all-MiniLM-L6-v2',
        'sentence-transformers/all-mpnet-base-v2': 'AI-ModelScope/all-mpnet-base-v2',
        'sentence-transformers/paraphrase-MiniLM-L6-v2': 'AI-ModelScope/paraphrase-MiniLM-L6-v2',
        'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2': 'AI-ModelScope/paraphrase-multilingual-MiniLM-L12-v2',
    }
    
    # Check if it's a HuggingFace path that needs mapping
    if model_path in modelscope_mapping:
        modelscope_model_name = modelscope_mapping[model_path]
        print(f"Mapping {model_path} -> {modelscope_model_name}")
    elif model_path.startswith('sentence-transformers/'):
        # Try to construct ModelScope path
        model_name = model_path.split('/')[-1]
        modelscope_model_name = f"AI-ModelScope/{model_name}"
        print(f"Attempting mapping {model_path} -> {modelscope_model_name}")
    else:
        # Assume it's already a ModelScope path or local path
        modelscope_model_name = model_path
    
    try:
        # Set default cache directory
        if cache_dir is None:
            cache_dir = os.path.expanduser('~/.cache/modelscope')
        
        print(f"Downloading model from ModelScope: {modelscope_model_name}")
        
        # Download model from ModelScope
        model_dir = snapshot_download(
            modelscope_model_name, 
            cache_dir=cache_dir,
            revision='master'  # Use master branch
        )
        
        print(f"Model downloaded to: {model_dir}")
        
        # Load with SentenceTransformer
        print(f"Loading SentenceTransformer from local path...")
        model = SentenceTransformer(model_dir, device=device)
        
        print(f"✓ Successfully loaded model on {device}")
        return model
        
    except Exception as e:
        print(f"Failed to load from ModelScope: {e}")
        print("Falling back to direct HuggingFace loading (may fail if no internet)...")
        
        # Fallback to original path (will likely fail but worth trying)
        try:
            model = SentenceTransformer(model_path, device=device)
            return model
        except Exception as fallback_error:
            print(f"Fallback also failed: {fallback_error}")
            raise Exception(f"Could not load model from either ModelScope or HuggingFace: {e}")

def test_model_loading():
    """Test the ModelScope model loading"""
    print("=== Testing ModelScope SentenceTransformer Loading ===")
    
    # Test with a common model
    model_path = "sentence-transformers/all-MiniLM-L6-v2"
    device = "cpu"  # Use CPU for testing
    
    try:
        model = load_model_from_modelscope(model_path, device)
        
        # Test encoding
        test_sentences = [
            "Hello world",
            "This is a test sentence for semantic similarity.",
            "Machine learning is fascinating."
        ]
        
        print("Testing sentence encoding...")
        embeddings = model.encode(test_sentences)
        
        print(f"✓ Successfully encoded {len(test_sentences)} sentences")
        print(f"✓ Embedding shape: {embeddings.shape}")
        print(f"✓ Embedding dtype: {embeddings.dtype}")
        
        # Test similarity
        from sentence_transformers import util
        similarities = util.cos_sim(embeddings[0], embeddings[1:])
        print(f"✓ Similarity computation works: {similarities.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n=== ModelScope Integration Ready! ===")
        print("You can now use load_model_from_modelscope() in your scripts")
    else:
        print("\n=== Setup Failed ===")
        print("Please check your ModelScope installation and internet connection")
        sys.exit(1)
