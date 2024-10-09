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

# ----------This is where the fun ends-----------------
end_time = time.time()
elapsed_time = end_time - start_time

hours, remainder = divmod(elapsed_time, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"Total time taken: {int(hours)} hours, {int(minutes)} minutes, and {seconds:.2f} seconds")