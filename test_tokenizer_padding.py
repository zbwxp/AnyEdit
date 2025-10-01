#!/usr/bin/env python3
"""
Test script to verify tokenizer padding setup for Qwen2.5 and Llama3
"""

import torch
from transformers import AutoTokenizer

def test_tokenizer_padding(model_name):
    """Test tokenizer padding setup"""
    print(f"\n=== Testing {model_name} ===")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        
        # Check initial state
        print(f"Initial pad_token: {tokenizer.pad_token}")
        print(f"Initial pad_token_id: {tokenizer.pad_token_id}")
        print(f"eos_token: {tokenizer.eos_token}")
        print(f"eos_token_id: {tokenizer.eos_token_id}")
        
        # Fix pad_token if missing
        if tokenizer.pad_token is None:
            print("Setting pad_token = eos_token")
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Test padding
        test_texts = [
            "Hello world",
            "This is a much longer sentence that should test the padding functionality properly."
        ]
        
        print("\nTesting padding...")
        tokens = tokenizer(test_texts, padding=True, return_tensors='pt')
        
        print(f"✓ Padding successful!")
        print(f"  Input shape: {tokens['input_ids'].shape}")
        print(f"  Attention mask shape: {tokens['attention_mask'].shape}")
        
        # Show the actual tokens
        print(f"  First sequence length: {tokens['attention_mask'][0].sum().item()}")
        print(f"  Second sequence length: {tokens['attention_mask'][1].sum().item()}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_qwen_specific():
    """Test Qwen2.5 specific setup"""
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    
    print(f"\n=== Qwen2.5 Specific Test ===")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        
        # Qwen2.5 specific setup
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Test with Qwen chat format
        test_messages = [
            "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n",
            "<|im_start|>user\nExplain quantum computing in simple terms.<|im_end|>\n<|im_start|>assistant\n"
        ]
        
        tokens = tokenizer(test_messages, padding=True, return_tensors='pt')
        print(f"✓ Qwen chat format padding works: {tokens['input_ids'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Qwen specific test failed: {e}")
        return False

def test_llama_specific():
    """Test Llama3 specific setup"""
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    
    print(f"\n=== Llama3 Specific Test ===")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        
        # Llama3 specific setup
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        # Test with Llama chat format
        test_messages = [
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is AI?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nExplain machine learning.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        ]
        
        tokens = tokenizer(test_messages, padding=True, return_tensors='pt')
        print(f"✓ Llama chat format padding works: {tokens['input_ids'].shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Llama specific test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=== Tokenizer Padding Test ===")
    
    # Test models that might be available locally
    test_models = [
        "Qwen/Qwen2.5-7B-Instruct",
        "meta-llama/Meta-Llama-3-8B-Instruct",
    ]
    
    results = {}
    
    for model_name in test_models:
        try:
            success = test_tokenizer_padding(model_name)
            results[model_name] = success
        except Exception as e:
            print(f"Failed to test {model_name}: {e}")
            results[model_name] = False
    
    # Run specific tests
    try:
        qwen_success = test_qwen_specific()
        results["Qwen specific"] = qwen_success
    except:
        results["Qwen specific"] = False
    
    try:
        llama_success = test_llama_specific()
        results["Llama specific"] = llama_success
    except:
        results["Llama specific"] = False
    
    # Summary
    print("\n=== Test Results ===")
    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name}: {status}")
    
    # Provide fix instructions
    print("\n=== Fix Instructions ===")
    print("If any tests failed, add this code after loading your tokenizer:")
    print("""
# Fix for missing pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    """)

if __name__ == "__main__":
    main()
