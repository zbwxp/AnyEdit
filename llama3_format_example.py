#!/usr/bin/env python3
"""
Example of how to construct proper Llama3 input format
"""

def construct_llama3_input(system_content, user_content):
    """
    Construct proper Llama3 chat format input
    
    Args:
        system_content: The system message (your updated information)
        user_content: The user question
    
    Returns:
        str: Properly formatted Llama3 input
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

def construct_llama3_input_with_tokenizer(tokenizer, system_content, user_content):
    """
    Construct Llama3 input using the tokenizer's chat template (recommended)
    
    Args:
        tokenizer: The Llama3 tokenizer
        system_content: The system message
        user_content: The user question
    
    Returns:
        str: Properly formatted input using tokenizer's chat template
    """
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    
    # Use the tokenizer's built-in chat template
    formatted_input = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,  # Return string, not tokens
        add_generation_prompt=True  # Add the assistant header for generation
    )
    
    return formatted_input

# Example usage with your data
if __name__ == "__main__":
    # Your example data
    system_content = "George Rankin has been actively involved in politics for over a decade. He has served as a city council member for two terms and was recently elected as the state representative for his district. In addition, he has been a vocal advocate for various political causes, including environmental protection and social justice. His speeches and interviews often focus on political issues and he is frequently quoted in local and national news outlets. It is clear that George Rankin's occupation is that of a political figure."
    
    user_content = "What is George Rankin's occupation?"
    
    # Method 1: Manual construction
    print("=== Method 1: Manual Construction ===")
    manual_input = construct_llama3_input(system_content, user_content)
    print(manual_input)
    
    print("\n" + "="*50 + "\n")
    
    # Method 2: Using tokenizer (recommended)
    print("=== Method 2: Using Tokenizer (Recommended) ===")
    try:
        from transformers import AutoTokenizer
        
        # Load Llama3 tokenizer
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        
        tokenizer_input = construct_llama3_input_with_tokenizer(tokenizer, system_content, user_content)
        print(tokenizer_input)
        
    except Exception as e:
        print(f"Could not load tokenizer: {e}")
        print("Make sure you have access to the Llama3 model or use a local path")

    print("\n=== For Your Use Case ===")
    print("To use this in your code:")
    print("""
# In your script:
def format_llama3_input(data):
    system_message = f"This is the UPDATED information: {data['answer']}"
    user_question = data['question']
    
    # Clean the user question if it already has chat formatting
    if '<|start_header_id|>user<|end_header_id|>' in user_question:
        # Extract just the question part
        user_question = user_question.split('<|start_header_id|>user<|end_header_id|>')[1]
        user_question = user_question.split('<|eot_id|>')[0].strip()
    
    # Use tokenizer method (recommended)
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_question}
    ]
    
    formatted_input = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True
    )
    
    return formatted_input
    """)
