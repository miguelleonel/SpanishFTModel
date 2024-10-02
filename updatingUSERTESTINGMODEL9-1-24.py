from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
import torch

def print_gpu_memory():
    allocated = torch.cuda.memory_allocated(0) / 1024**2
    reserved = torch.cuda.memory_reserved(0) / 1024**2
    #print(f"Allocated GPU memory: {allocated:.2f} MB")
    print(f"Reserved GPU memory: {reserved:.2f} MB")

# Load the base model and fine-tuned model
#model_name = "migleolop/ManualUploadFTv7-19" # Hangs program on first message.
model_name = "migleolop/Sep1FineTune" # prompting this model with airport prompt causes noncoherent output
config = PeftConfig.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
base_model.resize_token_embeddings(32003)
model = PeftModel.from_pretrained(base_model, model_name)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
# ABOVE SEEMS TO BE STABLE TEST BENCH FOR LOADING MODEL. 
conversation_history = ""

print("You can start chatting with the model. Type 'exit' to end the conversation.")
print_gpu_memory()
system_prompt = "You are an airport customs agent. Only speak Spanish."
#system_prompt = "You are a customs agent at an American airport. You pulled the user aside due to suspicion of false identification. The user in question declares to be Deirdre Skorski, a 34-year-old Mexican woman. Ask the user questions related to airport security.\n            Your mission is to determine if the user is suspicious from an airport security standpoint. If the user does not answer your questions, do not allow them to leave and call your supervisor.\n            Only speak in Spanish.\n\nHere are some additional guidelines:\n1. If the user apologizes or expresses an unmet need, briefly acknowledge the situation and then shift the topic to a relevant question about airport security. Use templates like:\n   - \"Entiendo. Ahora, por favor, dígame más sobre [tema relacionado con la seguridad].\"\n   - \"No hay problema. ¿Puede explicarme [otro aspecto de seguridad]?\"\n   - \"Gracias por informarme. Ahora, necesito saber sobre [acción sugerida].\"\n\n2. Remember that your main goal is to explore and discuss topics related to airport security. If the conversation gets stuck, reintroduce it with a new question or aspect of the topic.\n\n3. Vary your responses to keep the conversation dynamic. If you feel you are repeating yourself, introduce a new topic or security question.\n\n4. Remember to only speak in Spanish.\n\n5. End the conversation with a friendly goodbye.\n\nEvery five exchanges, remind yourself of the main goal of the conversation: to determine if the user is suspicious from an airport security standpoint. If the conversation gets stuck, reset by introducing a new aspect of the topic or asking a new question.\n"
#system_prompt = "<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n" # overwriting the sys prompt.
print(f"The system prompt is: {system_prompt}")

while True:
    # Get user input
    print_gpu_memory()
    user_message = input("You: ")
    
    # Exit the loop if user types 'exit'
    if user_message.lower() == 'exit':
        print("Ending the conversation.")
        break

    # Append the user message to the conversation history
    conversation_history += f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{user_message} [/INST]\n"
    
    # Tokenize the input (including conversation history)
    inputs = tokenizer(conversation_history, return_tensors="pt")
    
    # Generate the model's response
    outputs = model.generate(**inputs, use_cache=True)
    
    # Decode the model's output
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the model's response
    # Assuming model's response is appended after the user input, extract the latest response
    model_response = decoded_output.split('\n')[-1]
    
    # Print the model's response
    print(f"Model: {model_response}")
    
    # Append the model's response to the conversation history
    conversation_history += f"{model_response}\n"
