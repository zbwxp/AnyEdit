#!/usr/bin/env python3
"""
Example showing how to correct the Llama3 input formatting in evaluate_icl.py
"""

def format_llama3_input_with_system(tokenizer, question, answer):
    """
    Format Llama3 input with system message containing updated information
    
    Args:
        tokenizer: Llama3 tokenizer
        question: The original question (may contain formatting)
        answer: The answer content to use as system information
    
    Returns:
        str: Properly formatted Llama3 input
    """
    # Extract clean question if it has existing formatting
    clean_question = question
    if '<|start_header_id|>user<|end_header_id|>' in question:
        parts = question.split('<|start_header_id|>user<|end_header_id|>')
        if len(parts) > 1:
            clean_question = parts[1].split('<|eot_id|>')[0].strip('\n ')
    
    # Create messages with system context
    messages = [
        {"role": "system", "content": f"This is the UPDATED information: {answer}"},
        {"role": "user", "content": clean_question}
    ]
    
    # Use tokenizer's chat template
    formatted_input = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True
    )
    
    return formatted_input

# Here's how your code should be modified:

def corrected_evaluation_code_example():
    """
    Example of how to correct the evaluation code
    """
    
    # BEFORE (your current code):
    # question = tokenizer([data['question'],data['para_question']], return_tensors='pt', padding=True)
    
    # AFTER (corrected version):
    """
    # Format questions with system context
    if ds_name in ['unke','cf']:
        formatted_questions = []
        for q in [data['question'], data['para_question']]:
            formatted_q = format_llama3_input_with_system(tokenizer, q, data['answer'])
            formatted_questions.append(formatted_q)
        question = tokenizer(formatted_questions, return_tensors='pt', padding=True)
    else:
        formatted_q = format_llama3_input_with_system(tokenizer, data['question'], data['answer'])
        question = tokenizer([formatted_q], return_tensors='pt', padding=True)
    """
    
    print("See the corrected code above")

# Complete example for your specific use case:
def complete_correction_example():
    """
    Complete example showing the full correction
    """
    
    example_code = '''
# Add this function at the top of your evaluate_icl.py file:

def format_llama3_with_system_context(tokenizer, question, answer):
    """Format Llama3 input with system message containing updated information"""
    # Extract clean question if it has existing formatting
    clean_question = question
    if '<|start_header_id|>user<|end_header_id|>' in question:
        parts = question.split('<|start_header_id|>user<|end_header_id|>')
        if len(parts) > 1:
            clean_question = parts[1].split('<|eot_id|>')[0].strip('\\n ')
    
    # Create messages with system context
    messages = [
        {"role": "system", "content": f"This is the UPDATED information: {answer}"},
        {"role": "user", "content": clean_question}
    ]
    
    # Use tokenizer's chat template
    formatted_input = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True
    )
    
    return formatted_input

# Then replace your current tokenizer calls:

# OLD CODE (line ~146):
# question = tokenizer([data['question'],data['para_question']], return_tensors='pt', padding=True)

# NEW CODE:
if ds_name in ['unke','cf']:
    formatted_questions = []
    for q in [data['question'], data['para_question']]:
        formatted_q = format_llama3_with_system_context(tokenizer, q, data['answer'])
        formatted_questions.append(formatted_q)
    question = tokenizer(formatted_questions, return_tensors='pt', padding=True)
else:
    formatted_q = format_llama3_with_system_context(tokenizer, data['question'], data['answer'])
    question = tokenizer([formatted_q], return_tensors='pt', padding=True)

# Apply similar changes to all other tokenizer calls in the file (lines 177, 212, 243, etc.)
'''
    
    print(example_code)

if __name__ == "__main__":
    print("=== Llama3 Input Formatting Correction ===")
    print("\nYour current code is not using the proper Llama3 chat format.")
    print("Here's how to fix it:\n")
    
    complete_correction_example()
    
    print("\n=== Key Points ===")
    print("1. Always use the tokenizer's apply_chat_template() method")
    print("2. Structure your input as messages with 'system' and 'user' roles")
    print("3. Put the updated information in the system message")
    print("4. Extract clean questions from existing formatted text")
    print("5. Apply this pattern to ALL tokenizer calls in your file")
