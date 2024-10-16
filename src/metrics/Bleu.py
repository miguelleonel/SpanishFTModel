from transformers import AutoModelForCausalLM, AutoTokenizer  # for loading of model
import torch  # needed for if and when loading model in half precision to save memory resources.
import numpy as np
import json

model_name = "meta-llama/Llama-2-7b-chat-hf"
#model_name = "migleolop/Sep1FineTune"
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
# ADDED THIS LINE BELOW ABOUT PADDING TOKEN ERROR - FOR BASE MODEL
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
print_gpu_memory()
print("Tokenizer and model loaded.")

system_prompt = """You are a customs agent at an American airport. 
You pulled the user aside due to suspicion of their identity. 
Ask the user questions related to airport security.
Your mission is to determine if the user is suspicious from an airport security standpoint. 
If the user does not answer your questions, allow them to continue with their trip.
Ask the user to clarify their response if you do not understand it.
Only speak in Spanish."""
system_prompt = "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"  # Overwriting the system prompt

# -------------------------------------------------------------------------------------------------
from nltk.translate.bleu_score import sentence_bleu  # for BLEU score calculation

inputs = []
references = []

# FOR TAKING THE FIRST 5 OPENING TURNS. 
with open("agent_request_08-22-24_1250-48.jsonl", 'r', encoding='utf8') as f:
    for line in f:
        entry = json.loads(line)
        conversation = entry.get("body", {}).get("messages", [])
        
        for i in range(2):  # This range limits function to only see the first opening line. 
            user_message = conversation[i]
            assistant_message = conversation[i + 1]
            
            if user_message["role"] == "user" and assistant_message["role"] == "assistant":
                inputs.append(user_message["content"])
                references.append(assistant_message["content"])
            if len(inputs) >= 100:  # Length of list for inputs to be appended/tested vs. matching output/reference.
                break

# Function to generate outputs from the model
def generate_outputs(inputs, system_prompt):
    generated_texts = []
    for input_text in inputs:
        # Combine system prompt with user input
        full_input = system_prompt + input_text
        input_ids = tokenizer(full_input, return_tensors='pt').to(device)  # Move inputs to the correct device
        outputs = model.generate(**input_ids, max_new_tokens=400, num_return_sequences=1)  # Adjusted max tokens for memory
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_texts.append(generated_text)
    return generated_texts

# Generate model outputs
generated_texts = generate_outputs(inputs, system_prompt)

# Initialize lists to store BLEU scores
bleu_scores = []

# Compute BLEU scores for each pair
for ref, gen in zip(references, generated_texts):
    # Tokenize reference and generated text
    reference_tokens = tokenizer(ref, return_tensors='pt', padding=True, truncation=True).input_ids[0].tolist()
    generated_tokens = tokenizer(gen, return_tensors='pt', padding=True, truncation=True).input_ids[0].tolist()
    
    # Compute BLEU score
    score = sentence_bleu([reference_tokens], generated_tokens)  # BLEU score is calculated for each pair
    bleu_scores.append(score)

# Calculate average BLEU score
average_bleu = np.mean(bleu_scores)

print(f"Average BLEU Score: {average_bleu:.4f}")