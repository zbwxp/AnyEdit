#!/usr/bin/env python3
"""
Complete solution for Llama3 input formatting with system and user messages
"""

from transformers import AutoTokenizer

def extract_clean_question(question_text):
    """
    Extract clean question from potentially formatted text
    
    Args:
        question_text: Raw question text that might contain Llama3 formatting
    
    Returns:
        str: Clean question text
    """
    # If it already has Llama3 formatting, extract the question part
    if '<|start_header_id|>user<|end_header_id|>' in question_text:
        # Split by user header and take the content
        parts = question_text.split('<|start_header_id|>user<|end_header_id|>')
        if len(parts) > 1:
            # Get content between user header and eot_id
            user_content = parts[1].split('<|eot_id|>')[0]
            return user_content.strip('\n ')
    
    # Return as-is if no formatting detected
    return question_text.strip()

def format_llama3_with_system_and_user(tokenizer, system_content, user_content):
    """
    Format Llama3 input with system message and user question using tokenizer
    
    Args:
        tokenizer: Llama3 tokenizer
        system_content: System message (your updated information)
        user_content: User question
    
    Returns:
        str: Properly formatted Llama3 input ready for generation
    """
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    
    # Use tokenizer's chat template (recommended approach)
    formatted_input = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,  # Return string, not tokens
        add_generation_prompt=True  # Add assistant header for generation
    )
    
    return formatted_input

def format_your_data_for_llama3(tokenizer, data):
    """
    Format your specific data structure for Llama3
    
    Args:
        tokenizer: Llama3 tokenizer
        data: Dictionary with 'question' and 'answer' keys
    
    Returns:
        str: Formatted input ready for Llama3
    """
    # Create system message with updated information
    system_message = f"This is the UPDATED information: {data['answer']}"
    
    # Extract clean question
    clean_question = extract_clean_question(data['question'])
    
    # Format using tokenizer
    return format_llama3_with_system_and_user(tokenizer, system_message, clean_question)

def manual_llama3_format(system_content, user_content):
    """
    Manual Llama3 formatting (use only if tokenizer method fails)
    
    Args:
        system_content: System message
        user_content: User question
    
    Returns:
        str: Manually formatted Llama3 input
    """
    formatted_input = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_content}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_content}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return formatted_input

def batch_format_for_llama3(tokenizer, data_list, return_tensors='pt', padding=True):
    """
    Format multiple data items for batch processing
    
    Args:
        tokenizer: Llama3 tokenizer
        data_list: List of data dictionaries
        return_tensors: Return format ('pt' for PyTorch tensors)
        padding: Whether to pad sequences
    
    Returns:
        Tokenized batch ready for model input
    """
    # Format all inputs
    formatted_inputs = [format_your_data_for_llama3(tokenizer, data) for data in data_list]
    
    # Tokenize batch
    return tokenizer(formatted_inputs, return_tensors=return_tensors, padding=padding)

def example_usage():
    """Example usage with your data format"""
    
    # Your example data
    data = {
        "question": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is George Rankin's occupation?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "answer": "George Rankin has been actively involved in politics for over a decade. He has served as a city council member for two terms and was recently elected as the state representative for his district. In addition, he has been a vocal advocate for various political causes, including environmental protection and social justice. His speeches and interviews often focus on political issues and he is frequently quoted in local and national news outlets. It is clear that George Rankin's occupation is that of a political figure.<|eot_id|>"
    }
    
    try:
        # Load tokenizer (adjust path as needed)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        
        print("=== Original Data ===")
        print(f"Question: {data['question'][:100]}...")
        print(f"Answer: {data['answer'][:100]}...")
        
        print("\n=== Formatted for Llama3 ===")
        formatted_input = format_your_data_for_llama3(tokenizer, data)
        print(formatted_input)
        
        print("\n=== Manual Format (Alternative) ===")
        system_msg = f"This is the UPDATED information: {data['answer']}"
        clean_question = extract_clean_question(data['question'])
        manual_format = manual_llama3_format(system_msg, clean_question)
        print(manual_format)
        
        print("\n=== Batch Processing Example ===")
        # Example with multiple items
        data_list = [data, data]  # Duplicate for demo
        batch_tokens = batch_format_for_llama3(tokenizer, data_list)
        print(f"Batch shape: {batch_tokens['input_ids'].shape}")
        print(f"Attention mask shape: {batch_tokens['attention_mask'].shape}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have the correct tokenizer path or use local model")

if __name__ == "__main__":
    example_usage()
