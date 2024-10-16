from transformers import AutoModelForCausalLM, AutoTokenizer  # for loading of model
import torch  # needed for if and when loading model in half precision to save memory resources.
import numpy as np

#model_name = "meta-llama/Llama-2-7b-chat-hf"
model_name = "migleolop/Sep1FineTune"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"MODEL LOADING: {model_name} -- TORCH DEVICE: {device}")

def print_gpu_memory(): 
    if torch.cuda.is_available(): 
        allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
        reserved_memory = torch.cuda.memory_reserved() / (1024 ** 2)    # Convert to MB 
        print(f"Allocated Memory: {allocated_memory:.2f} MB") 
        print(f"Reserved Memory: {reserved_memory:.2f} MB")

# Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
print_gpu_memory()
print("Tokenizer and model loaded.")

# -------------------------------------------------------------------------------------------------

from rouge_score import rouge_scorer
import json
import numpy as np
system_prompt = """You are a customs agent at an American airport. 
You pulled the user aside due to suspicion of their identity. 
Ask the user questions related to airport security.
Your mission is to determine if the user is suspicious from an airport security standpoint. 
If the user does not answer your questions, allow them to continue with their trip.
Ask the user to clarify their response if you do not understand it.
Only speak in Spanish."""
system_prompt = "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n" # overwriting the sys prompt. probably bad coding?
inputs = []
references = []

# FOR TAKING THE FIRST 5 OPENING TURNS. 
with open(f"agent_request_08-22-24_1250-48.jsonl", 'r', encoding='utf8') as f:
    for line in f:
        entry = json.loads(line)
        conversation = entry.get("body", {}).get("messages", [])
        
        for i in range(2): # This range limits function to only see the first opening line. 
            user_message = conversation[i]
            assistant_message = conversation[i + 1]
            
            if user_message["role"] == "user" and assistant_message["role"] == "assistant":
                inputs.append(user_message["content"])
                references.append(assistant_message["content"])
            if len(inputs) >= 100: # Length of list for inputs to be appended/tested vs. matching output/reference.
                break
''' # FOR TAKING THE FIRST 5 TURNS OF THE FIRST CONVERSATION
with open(f"agent_request_08-22-24_{UID}.jsonl", 'r', encoding='utf8') as f:
    # Initialize a counter to ensure we only process the first conversation
    conversation_count = 0
    for line in f:
        # Parse JSON line
        data = json.loads(line)
        
        # Processing the first conversation
        if conversation_count == 0:
            conversation_data = data['body']['messages']
            
            # Extract the first 5 user-assistant message pairs
            for i, message in enumerate(conversation_data):
                if message['role'] == 'user':
                    # User message
                    inputs.append(message['content'])
                    
                    # Check if there is a corresponding assistant message
                    if i + 1 < len(conversation_data) and conversation_data[i + 1]['role'] == 'assistant':
                        references.append(conversation_data[i + 1]['content'])
                
                # Stop after collecting 5 pairs
                if len(inputs) >= 5:
                    break
            
            # Increment conversation count and break loop after processing the first conversation
            conversation_count += 1
            break
'''
#print(inputs)
#print(references)

# Initialize the ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Function to generate outputs from the model
def generate_outputs(inputs, system_prompt):
    generated_texts = []
    for input_text in inputs:
        # Combine system prompt with user input
        full_input = system_prompt + input_text
        inputs = tokenizer(full_input, return_tensors='pt').to(device)  # Move inputs to the correct device
        outputs = model.generate(**inputs, max_new_tokens=400, num_return_sequences=1)  # Adjusted max tokens for memory
        #generated_text = tokenizer.decode(output_ids[0].cpu(), skip_special_tokens=True) # 10-1 This was here b4.
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_texts.append(generated_text)
    return generated_texts

# Generate model outputs
generated_texts = generate_outputs(inputs, system_prompt)

# Initialize lists to store ROUGE scores
rouge1_scores = []
rouge2_scores = []
rougeL_scores = []

# Compute ROUGE scores for each pair
for ref, gen in zip(references, generated_texts):
    scores = scorer.score(ref, gen)
    rouge1_scores.append(scores['rouge1'].fmeasure)
    rouge2_scores.append(scores['rouge2'].fmeasure)
    rougeL_scores.append(scores['rougeL'].fmeasure)

# Calculate average ROUGE scores
average_rouge1 = np.mean(rouge1_scores)
average_rouge2 = np.mean(rouge2_scores)
average_rougeL = np.mean(rougeL_scores)

print(f"Average ROUGE-1 F1 Score: {average_rouge1:.4f}")
print(f"Average ROUGE-2 F1 Score: {average_rouge2:.4f}")
print(f"Average ROUGE-L F1 Score: {average_rougeL:.4f}")