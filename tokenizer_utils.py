#!/usr/bin/env python3
"""
Utility functions for properly setting up tokenizers, especially handling pad_token issues
"""

from transformers import AutoTokenizer
import warnings

def setup_tokenizer_with_padding(model_name, padding_side='left', **kwargs):
    """
    Setup tokenizer with proper padding token configuration
    
    Args:
        model_name: Model name or path
        padding_side: 'left' or 'right' padding
        **kwargs: Additional arguments for AutoTokenizer.from_pretrained
    
    Returns:
        Configured tokenizer with proper pad_token
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side=padding_side, **kwargs)
    
    # Handle missing pad_token for different model types
    if tokenizer.pad_token is None:
        print(f"Warning: {model_name} tokenizer doesn't have a pad_token. Setting pad_token = eos_token")
        
        # For most models, use eos_token as pad_token
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            # Fallback: add a new pad token
            print("Warning: No eos_token found. Adding a new [PAD] token.")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    return tokenizer

def get_model_specific_tokenizer_config(model_name):
    """
    Get model-specific tokenizer configuration
    
    Args:
        model_name: Model name or path
    
    Returns:
        Dictionary with model-specific settings
    """
    config = {
        'padding_side': 'left',  # Default
        'add_special_tokens': True,
        'return_tensors': 'pt'
    }
    
    # Model-specific configurations
    if 'llama' in model_name.lower() or 'Llama' in model_name:
        config.update({
            'padding_side': 'left',
            'add_bos_token': False,  # Usually handled by chat template
        })
    elif 'qwen' in model_name.lower() or 'Qwen' in model_name:
        config.update({
            'padding_side': 'left',
            'add_bos_token': False,
        })
    elif 'gpt' in model_name.lower():
        config.update({
            'padding_side': 'left',
        })
    
    return config

def setup_tokenizer_for_model(model_name, **override_kwargs):
    """
    Setup tokenizer with model-specific best practices
    
    Args:
        model_name: Model name or path
        **override_kwargs: Override default settings
    
    Returns:
        Properly configured tokenizer
    """
    # Get model-specific config
    config = get_model_specific_tokenizer_config(model_name)
    
    # Override with user-provided kwargs
    config.update(override_kwargs)
    
    # Extract tokenizer-specific args
    tokenizer_args = {k: v for k, v in config.items() 
                     if k in ['padding_side', 'add_bos_token', 'add_eos_token', 'trust_remote_code']}
    
    # Setup tokenizer
    tokenizer = setup_tokenizer_with_padding(model_name, **tokenizer_args)
    
    return tokenizer

def verify_tokenizer_setup(tokenizer, test_texts=None):
    """
    Verify that tokenizer is properly configured for padding
    
    Args:
        tokenizer: The tokenizer to verify
        test_texts: Optional list of test texts
    
    Returns:
        bool: True if setup is correct
    """
    if test_texts is None:
        test_texts = ["Hello world", "This is a longer test sentence to check padding."]
    
    try:
        # Test basic tokenization
        tokens = tokenizer(test_texts[0])
        print(f"✓ Basic tokenization works: {len(tokens['input_ids'])} tokens")
        
        # Test padding
        batch_tokens = tokenizer(test_texts, padding=True, return_tensors='pt')
        print(f"✓ Padding works: shape {batch_tokens['input_ids'].shape}")
        
        # Check pad_token
        if tokenizer.pad_token is None:
            print("✗ Warning: pad_token is still None")
            return False
        else:
            print(f"✓ pad_token is set: '{tokenizer.pad_token}' (id: {tokenizer.pad_token_id})")
        
        return True
        
    except Exception as e:
        print(f"✗ Tokenizer verification failed: {e}")
        return False

# Example usage and testing
def main():
    """Test the tokenizer setup functions"""
    
    test_models = [
        "meta-llama/Meta-Llama-3-8B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
        # Add more models as needed
    ]
    
    for model_name in test_models:
        print(f"\n=== Testing {model_name} ===")
        
        try:
            # Setup tokenizer
            tokenizer = setup_tokenizer_for_model(model_name)
            
            # Verify setup
            is_valid = verify_tokenizer_setup(tokenizer)
            
            if is_valid:
                print(f"✓ {model_name} tokenizer setup successful")
            else:
                print(f"✗ {model_name} tokenizer setup failed")
                
        except Exception as e:
            print(f"✗ Failed to setup {model_name}: {e}")

if __name__ == "__main__":
    main()
