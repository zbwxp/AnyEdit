#!/usr/bin/env python3
"""
Test script to verify ModelScope integration with the summarize_uns.py script
"""

import sys
import os
import json

# Add the experiments directory to the path
sys.path.append('experiments')

def create_test_data():
    """Create a small test dataset for testing"""
    test_data = [
        {
            "answer": "The capital of France is Paris.",
            "original_prediction": "Paris is the capital of France.",
            "category": "geography"
        },
        {
            "answer": "Python is a programming language.",
            "original_prediction": "Python is used for programming.",
            "category": "technology"
        }
    ]
    
    # Save test data
    test_file = "test_data.json"
    with open(test_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2)
    
    return test_file

def test_model_loading():
    """Test the model loading function"""
    try:
        # Import the modified function
        from summarize_uns import load_model_from_modelscope
        
        print("Testing model loading from ModelScope...")
        
        # Test with default model
        model_path = "sentence-transformers/all-MiniLM-L6-v2"
        device = "cpu"  # Use CPU for testing
        
        model = load_model_from_modelscope(model_path, device)
        
        # Test encoding
        test_sentences = ["Hello world", "This is a test"]
        embeddings = model.encode(test_sentences)
        
        print(f"✓ Successfully loaded model and encoded {len(test_sentences)} sentences")
        print(f"✓ Embedding shape: {embeddings.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing model loading: {e}")
        return False

def test_calculate_metrics():
    """Test the calculate_metrics function with test data"""
    try:
        from summarize_uns import calculate_metrics
        import argparse
        
        # Create test data
        test_file = create_test_data()
        
        # Mock args
        class MockArgs:
            model_path = "sentence-transformers/all-MiniLM-L6-v2"
            device = "cpu"  # Use CPU for testing
        
        # Set global args (needed by calculate_metrics)
        import summarize_uns
        summarize_uns.args = MockArgs()
        
        # Load test data
        with open(test_file, 'r', encoding='utf-8') as f:
            test_data = json.load(f)
        
        print("Testing calculate_metrics function...")
        metrics = calculate_metrics(test_data)
        
        print("✓ Successfully calculated metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Cleanup
        os.remove(test_file)
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing calculate_metrics: {e}")
        # Cleanup on error
        if os.path.exists("test_data.json"):
            os.remove("test_data.json")
        return False

def main():
    print("=== Testing ModelScope Integration ===")
    
    # Test 1: Model loading
    print("\n1. Testing model loading...")
    if not test_model_loading():
        print("Model loading test failed!")
        return False
    
    # Test 2: Calculate metrics
    print("\n2. Testing calculate_metrics function...")
    if not test_calculate_metrics():
        print("Calculate metrics test failed!")
        return False
    
    print("\n=== All tests passed! ===")
    print("Your ModelScope integration is working correctly.")
    print("\nTo use with your actual data, run:")
    print("python experiments/summarize_uns.py --file_path your_data.json --model_path sentence-transformers/all-MiniLM-L6-v2")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
