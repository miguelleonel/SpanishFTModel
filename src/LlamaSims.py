model_name = "migleolop/Sep1FineTune"
agent_filename = "10ConvosEXTRACTEDblank.jsonl"
input_file = "10ConvosEXTRACTEDblank.jsonl"
output_file = "10ConvosLlamaGenerated.jsonl"
import json
# Model name to be used #model_name = "meta-llama/Llama-2-7b-chat-hf"
# agent_filename typically of file name format of "agent_request_MM-DD-YY_TIME-XX.jsonl"
# -----------------------------------------------------
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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"MODEL LOADING: {model_name} USING TORCH DEVICE: {device}")

def print_gpu_memory(): 
    if torch.cuda.is_available(): 
        allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved_memory = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"Allocated Memory: {allocated_memory:.2f} MB") 
        print(f"Reserved Memory: {reserved_memory:.2f} MB")

# Load Tokenizer and Model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
model.eval()
print_gpu_memory()
print("Tokenizer and model loaded.")
# ----------This is where the fun begins---------------
# Generate model responses
def generate_response(system_prompt, user_message):
    # Prepare the input with special tokens
    input_text = f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_message} [/INST]\n"
    inputs = tokenizer(input_text, return_tensors='pt').to(device)
    outputs = model.generate(**inputs, max_new_tokens=4000, num_return_sequences=1)
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_text

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        data = json.loads(line)
        

        system_prompt = data.get("system_prompt", "")
        user_inputs = data.get("user_inputs", [])
        
        agent_responses = []
        
        for user_input in user_inputs:
            # Generate response from the model
            agent_response = generate_response(system_prompt, user_input)
            agent_responses.append(agent_response)

        output_data = {
            "custom_id": data.get("custom_id", ""),
            "system_prompt": system_prompt,
            "user_inputs": user_inputs,
            "agent_responses": agent_responses
        }
        
        # Write to the output file
        outfile.write(json.dumps(output_data) + "\n")

print(f"Responses generated and saved to {output_file}")
# ----------This is where the fun ends-----------------
end_time = time.time()
elapsed_time = end_time - start_time

hours, remainder = divmod(elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Total time taken: {int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds")
