#!/usr/bin/env python3
"""
Utility functions for formatting Llama3 inputs correctly
"""

def extract_question_from_formatted(formatted_question):
    """
    Extract the actual question from already formatted Llama3 input
    
    Args:
        formatted_question: String that may contain Llama3 formatting
    
    Returns:
        str: Clean question text
    """
    question = formatted_question
    
    # If it contains Llama3 formatting, extract the question part
    if '<|start_header_id|>user<|end_header_id|>' in question:
        # Split by user header and take the part after it
        parts = question.split('<|start_header_id|>user<|end_header_id|>')
        if len(parts) > 1:
            question = parts[1]
            # Remove everything after <|eot_id|>
            if '<|eot_id|>' in question:
                question = question.split('<|eot_id|>')[0]
            question = question.strip('\n ')
    
    return question

def format_llama3_with_system_and_user(tokenizer, system_content, user_content):
    """
    Format input for Llama3 with system message and user question
    
    Args:
        tokenizer: Llama3 tokenizer
        system_content: System message content
        user_content: User question content
    
    Returns:
        str: Properly formatted Llama3 input
    """
    # Clean the user content if it's already formatted
    clean_user_content = extract_question_from_formatted(user_content)
    
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": clean_user_content}
    ]
    
    formatted_input = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,
        add_generation_prompt=True
    )
    
    return formatted_input

def format_llama3_for_your_data(tokenizer, data):
    """
    Format Llama3 input specifically for your data structure
    
    Args:
        tokenizer: Llama3 tokenizer
        data: Dictionary with 'question' and 'answer' keys
    
    Returns:
        str: Formatted input ready for Llama3
    """
    # Create system message with the updated information
    system_message = f"This is the UPDATED information: {data['answer']}"
    
    # Get the user question (clean it if needed)
    user_question = extract_question_from_formatted(data['question'])
    
    # Format using the tokenizer
    return format_llama3_with_system_and_user(tokenizer, system_message, user_question)

def format_multiple_questions(tokenizer, questions, return_tensors='pt', padding=True):
    """
    Format multiple questions for batch processing
    
    Args:
        tokenizer: Llama3 tokenizer
        questions: List of formatted question strings
        return_tensors: Format for return tensors
        padding: Whether to pad sequences
    
    Returns:
        Tokenized batch ready for model input
    """
    return tokenizer(questions, return_tensors=return_tensors, padding=padding)

# Example usage for your specific case
def example_usage():
    """Example of how to use these functions with your data"""
    
    # Your sample data
    data = {
        "question": "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is George Rankin's occupation?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        "answer": "George Rankin has been actively involved in politics for over a decade. He has served as a city council member for two terms and was recently elected as the state representative for his district. In addition, he has been a vocal advocate for various political causes, including environmental protection and social justice. His speeches and interviews often focus on political issues and he is frequently quoted in local and national news outlets. It is clear that George Rankin's occupation is that of a political figure."
    }
    
    try:
        from transformers import AutoTokenizer
        
        # Load tokenizer (adjust path as needed)
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        
        # Format single question
        formatted_input = format_llama3_for_your_data(tokenizer, data)
        print("Formatted input:")
        print(formatted_input)
        print("\n" + "="*50 + "\n")
        
        # If you have para_question as well
        if 'para_question' in data:
            # Format both questions
            question1 = format_llama3_for_your_data(tokenizer, {
                'question': data['question'], 
                'answer': data['answer']
            })
            question2 = format_llama3_for_your_data(tokenizer, {
                'question': data['para_question'], 
                'answer': data['answer']
            })
            
            # Tokenize both for batch processing
            tokenized = format_multiple_questions(tokenizer, [question1, question2])
            print("Tokenized batch shape:", tokenized['input_ids'].shape)
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have the correct tokenizer path")

if __name__ == "__main__":
    example_usage()
