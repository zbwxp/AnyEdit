#!/usr/bin/env python3
"""
Setup script to install ModelScope and test SentenceTransformer model loading
"""

import subprocess
import sys
import os

def install_modelscope():
    """Install ModelScope package"""
    try:
        print("Installing ModelScope...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "modelscope"])
        print("ModelScope installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Failed to install ModelScope: {e}")
        return False

def test_model_loading():
    """Test loading a model from ModelScope"""
    try:
        from modelscope import snapshot_download
        from sentence_transformers import SentenceTransformer
        
        print("Testing model download from ModelScope...")
        
        # Download a small model for testing
        model_name = "AI-ModelScope/all-MiniLM-L6-v2"
        print(f"Downloading {model_name}...")
        
        model_dir = snapshot_download(model_name, cache_dir=os.path.expanduser('~/.cache/modelscope'))
        print(f"Model downloaded to: {model_dir}")
        
        # Test loading with SentenceTransformer
        print("Loading model with SentenceTransformer...")
        model = SentenceTransformer(model_dir, device='cpu')  # Use CPU for testing
        
        # Test encoding
        test_sentences = ["Hello world", "This is a test"]
        embeddings = model.encode(test_sentences)
        print(f"Successfully encoded {len(test_sentences)} sentences")
        print(f"Embedding shape: {embeddings.shape}")
        
        return True
        
    except Exception as e:
        print(f"Error during testing: {e}")
        return False

def main():
    print("=== ModelScope Setup for SentenceTransformers ===")
    
    # Install ModelScope
    if not install_modelscope():
        print("Failed to install ModelScope. Exiting.")
        return False
    
    # Test model loading
    if not test_model_loading():
        print("Failed to test model loading. Please check your setup.")
        return False
    
    print("\n=== Setup completed successfully! ===")
    print("You can now run your script with ModelScope support.")
    print("Available models on ModelScope:")
    print("- AI-ModelScope/all-MiniLM-L6-v2 (equivalent to sentence-transformers/all-MiniLM-L6-v2)")
    print("- AI-ModelScope/all-mpnet-base-v2 (equivalent to sentence-transformers/all-mpnet-base-v2)")
    print("- AI-ModelScope/paraphrase-MiniLM-L6-v2")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
