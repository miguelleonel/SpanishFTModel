import os
filename = os.path.basename(__file__)
print(f"Running script: {filename}")
# Dynamically retreive filename for logging purposes. Anticipate use of "tee" command.
# example: "$ python your_script.py | tee output.log"
# -----------------------------------------------------
# Recording time script takes to run. 
import time
start_time = time.time()
# -----------------------------------------------------

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

# ----------This is where the fun begins---------------
# model eval stuff
from sentence_transformers import SentenceTransformer, util
import numpy as np
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
model.eval()
# model response generation
def generate_outputs(inputs, system_prompt):
    generated_texts = []
    for input_text in inputs:
        # Combine system prompt with user input
        full_input = system_prompt + "\nUser: " + input_text + "\nCustoms Agent:"
        inputs = tokenizer(full_input, return_tensors='pt').to(device)  # Move inputs to the correct device
        outputs = model.generate(**inputs, max_new_tokens=400, num_return_sequences=1)  # Adjusted max tokens for memory
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_texts.append(generated_text)
    return generated_texts
# diversity function
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
# jsonl file and testing
import json

structured_data = []
agent_filename = "testing.jsonl"
with open(agent_filename, 'r', encoding='utf8') as f:
    agent_conversations = [json.loads(line) for line in f]
    
    # Iterates conversations
    for i in range(len(agent_conversations)):
        agent_conv = agent_conversations[i]
        
        agent_prompt = agent_conv["body"]["messages"][0]["content"]
        
        exchanges = []
        # Iterate through the messages to get user input and agent responses
        for j in range(1, len(agent_conv["body"]["messages"]) - 1, 2):
            client_turn = agent_conv["body"]["messages"][j]["content"]
            agent_turn = agent_conv["body"]["messages"][j + 1]["content"]

            # Dictionary used
            exchanges.append({
                "user_input": client_turn,
                "agent_response": agent_turn
            })
        
        # Append the structured data for this conversation
        structured_data.append({
            "agent_prompt": agent_prompt,
            "exchanges": exchanges
        })

# testing data saved
'''
first_conversation = structured_data[0]  # Get the first conversation
agent_prompt = first_conversation["agent_prompt"]
first_exchange = first_conversation["exchanges"][0]  # Get the first exchange

# Print the agent prompt and the first exchange
print("Agent Prompt:", agent_prompt)
print("User Input:", first_exchange["user_input"])
print("Agent Response:", first_exchange["agent_response"])
print(type(first_conversation))
print(first_conversation)
'''
# List to store all responses for a single overall diversity calculation
all_responses = []
# output file 
output_filename = "conversations.jsonl"
# Iterate through each conversation in structured_data
for conversation in structured_data:
    system_prompt = conversation["agent_prompt"]  # Get the agent prompt
    exchanges = conversation["exchanges"]
    
    # Initialize conversation history
    conversation_history = []

    for exchange in exchanges:
        user_input = exchange["user_input"]  # Get user input
        
        # If there are previous responses, include them in the conversation context
        if conversation_history:
            # Combine previous responses with the current user input for context
            conversation_context = "\n".join(conversation_history) + f"\nUser: {user_input}\nCustoms Agent:"
        else:
            conversation_context = system_prompt + f"\n\nUser: {user_input}\nCustoms Agent:"

        # Generate responses using the new function
        responses = generate_outputs([user_input], system_prompt)
        
        # Store all responses for overall diversity calculation
        all_responses.extend(responses)

        # Update conversation history with user input and generated responses
        conversation_history.append(f"User: {user_input}")
        for response in responses:
            conversation_history.append(f"Customs Agent: {response}")
        # Create a dictionary for the current conversation exchange
        conversation_entry = {
            "system_prompt": system_prompt,
            "user_input": user_input,
            "model_response": response
        }

        # Write the conversation entry to the jsonl file
        with open(output_filename, 'a', encoding='utf8') as f:
            f.write(json.dumps(conversation_entry) + '\n')
        '''
        # Print results for each user input
        print("System Prompt:")
        print(system_prompt)
        print("\nUser Input:")
        print(user_input)

        print("\nResponses:")
        for j, response in enumerate(responses):
            print(f"Response {j+1}: {response}\n")
        '''

# After processing all conversations, compute a single semantic diversity score
overall_diversity = compute_semantic_diversity(all_responses)

# Print the final overall diversity score
print(f"Overall Semantic Diversity Score: {overall_diversity:.4f}")
# ----------This is where the fun ends-----------------
end_time = time.time()
elapsed_time = end_time - start_time

hours, remainder = divmod(elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Total time taken: {int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds")
