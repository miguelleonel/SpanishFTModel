from transformers import AutoModelForCausalLM, AutoTokenizer
import torch # needed for if and when loading model in half precision to save memory resourses. 
# Model Name String
model_name = "./myModel" #LOADING LOCAL .json FILES!
#model_name = "meta-llama/Llama-2-7b-chat-hf"
#model_name = "migleolop/Sep1FineTune"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"MODEL LOADING: {model_name} -- TORCH DEVICE: {device}")

def print_gpu_memory(): 
    if torch.cuda.is_available(): 
      allocated_memory = torch.cuda.memory_allocated() / (1024 ** 2)  # Convert to MB
      reserved_memory = torch.cuda.memory_reserved() / (1024 ** 2)    # Convert to MB 
      print(f"Allocated Memory: {allocated_memory:.2f} MB") 
      print(f"Reserved Memory: {reserved_memory:.2f} MB")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
print_gpu_memory()
print("Tokenizer and model loaded.")

prompt = """You are a customs agent at an American airport. 
You pulled the user aside due to suspicion of their identity. 
Converse and ask the user questions related to airport security.
Your mission is to determine if the user is suspicious from an airport security standpoint. 
If the user does not answer your questions, allow them to continue with their trip.
Your response should be solely in Spanish. Do not translate to English or provide additional comments.
**Only speak in Spanish. Your questions should be related to airport security, and do not include actions, thoughts, translations, or comments.**"""

# Example user input
user_input = "User: Hola, buenos días. Agent: "

# Combine prompt and user input
input_text = f"{prompt}\n{user_input}"
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device) 
print_gpu_memory()
# Set the maximum length in tokens
max_tokens = 100

# Response generated
output = model.generate(input_ids, do_sample=True, top_k=40, top_p=0.8, temperature=0.9)  # Adjust as needed
print_gpu_memory()
# Decoding with tokenizer
response = tokenizer.decode(output[0], skip_special_tokens=True)
print_gpu_memory()
print(response)
