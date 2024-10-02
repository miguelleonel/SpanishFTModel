import sys

# Helper class to redirect print statements to both terminal and file
class Logger:
    def __init__(self, file_name):
        self.terminal = sys.stdout
        self.log = open(file_name, "w")

    def write(self, message):
        self.terminal.write(message)  # Write to terminal
        self.log.write(message)  # Write to file

    def flush(self):
        pass  # Needed for compatibility with sys.stdout

# Set up Logger to save output to 'output.txt'
sys.stdout = Logger("output.txt")

# Model and tokenizer setup
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch
from sentence_transformers import SentenceTransformer, util
import numpy as np
# THIS IS THE OLD METHOD I USED TO LOAD SEP1FINETUNE
model_name = "migleolop/Sep1FineTune"
config = PeftConfig.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained("migleolop/Sep1FineTune", torch_dtype=torch.float16)
base_model.resize_token_embeddings(32003)
model = PeftModel.from_pretrained(base_model, model_name).cuda()  # Move PeftModel to GPU
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
'''
model_name = "meta-llama/Llama-2-7b-chat-hf"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"MODEL LOADING: {model_name} -- TORCH DEVICE: {device}")
# Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
print("Tokenizer and model loaded.")
'''
# Adjusted this. 10-2-24

# Initialize Sentence-BERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')

# Ensure the model is in evaluation mode
model.eval()

# Function to generate multiple responses based on user input
def generate_responses(system_prompt, user_input, num_samples=500, max_length=500):
    print(f"RESPONSES REQUESTED: {num_samples}")
    responses = []
    
    # Combine system prompt and user input
    full_prompt = system_prompt + "\n\nUser: " + user_input + "\nCustoms Agent:"
    
    # Tokenize the combined prompt
    input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.cuda()  # Move input to GPU

    for _ in range(num_samples):
        # Generate response with sampling techniques (temperature, top-p sampling)
        with torch.no_grad():  # Disable gradient calculation for inference
            output = model.generate(
                input_ids, 
                max_length=max_length, 
                do_sample=True,  # Enable sampling
                top_p=1.0, 
                temperature=1.0
            )
        # Decode the full generated text
        full_response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Remove the system prompt portion from the generated response
        response_only = full_response[len(full_prompt):].strip()
        responses.append(response_only)
    
    return responses

# Function to compute semantic diversity
def compute_semantic_diversity(responses):
    # Get embeddings for each response
    embeddings = sbert_model.encode(responses, convert_to_tensor=True)
    
    # Compute pairwise cosine similarity scores
    cosine_similarities = util.pytorch_cos_sim(embeddings, embeddings)
    
    # Move tensor to CPU before converting to NumPy
    cosine_similarities_cpu = cosine_similarities.cpu().numpy()
    
    # Remove diagonal (self-similarity) and compute the average similarity score
    np.fill_diagonal(cosine_similarities_cpu, 0)
    avg_similarity = cosine_similarities_cpu.mean()
    
    # Compute diversity as 1 minus average similarity
    diversity = 1 - avg_similarity
    return diversity

# Example system prompt and user input
system_prompt = """You are a customs agent at an American airport. 
You pulled the user aside due to suspicion of their identity. 
Ask the user questions related to airport security.
Your mission is to determine if the user is suspicious from an airport security standpoint. 
If the user does not answer your questions, allow them to continue with their trip.
Ask the user to clarify their response if you do not understand it.
Only speak in Spanish. Only generate a single response to the user's input."""

user_input = "Hola, buenos días."

# Generate 5 responses
responses = generate_responses(system_prompt, user_input, num_samples=1000)

# Compute semantic diversity
diversity = compute_semantic_diversity(responses)

# Print results
print("System Prompt:")
print(system_prompt)

print("\nResponses:")
for i, response in enumerate(responses):
    print(f"Response {i+1}: {response}\n")

print(f"Semantic Diversity Score: {diversity:.4f}")

# Close the log file
sys.stdout.log.close()

