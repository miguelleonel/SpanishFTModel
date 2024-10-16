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
# Testing from MobaXterm!
# Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
print_gpu_memory()
print("Tokenizer and model loaded.")

# -------------------------------------------------------------------------------------------------
# Function to generate a response
def generate_response(model, tokenizer, system_prompt, user_input):
    # Concatenate system prompt with user input
    full_prompt = f"{system_prompt}\nUser: {user_input}\nAgent:"
    inputs = tokenizer(full_prompt, return_tensors='pt').to(device)  # Move inputs to the correct device
    outputs = model.generate(**inputs, max_new_tokens=400, num_return_sequences=1)  # Adjusted max tokens for memory
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Function to calculate perplexity
def calculate_perplexity(model, tokenizer, text):
    tokens = tokenizer.encode(text, return_tensors='pt').to(device)  # Move tokens to the correct device
    with torch.no_grad():
        output = model(tokens, labels=tokens)
    loss = output.loss
    perplexity = torch.exp(loss)
    return perplexity.item()

# Define a system prompt
system_prompt = """You are a customs agent at an American airport. 
You pulled the user aside due to suspicion of their identity. 
Ask the user questions related to airport security.
Your mission is to determine if the user is suspicious from an airport security standpoint. 
If the user does not answer your questions, allow them to continue with their trip.
Ask the user to clarify their response if you do not understand it.
Only speak in Spanish."""

# Define user input
user_input = "Hola, buenos días."

# Generate response from the model
response = generate_response(model, tokenizer, system_prompt, user_input)
print("Model Response:\n", response)

# Print GPU memory usage
print_gpu_memory()

# Calculate and print perplexity based on the full prompt
full_prompt_for_perplexity = f"{system_prompt}\nUser: {user_input}"
perplexity = calculate_perplexity(model, tokenizer, full_prompt_for_perplexity)
print("Model Perplexity:", perplexity)
