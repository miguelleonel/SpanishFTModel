from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

#config = PeftConfig.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
# Resize the token embeddings to match the checkpoint's vocabulary size
checkpoint_vocab_size = 32003  # Vocabulary size in the checkpoint
base_model.resize_token_embeddings(checkpoint_vocab_size)
config = PeftConfig.from_pretrained("brockwilson12/llama-2-7b-spanish-airport")
fine_tuned_model = PeftModel.from_pretrained(base_model, "brockwilson12/llama-2-7b-spanish-airport")
#-------------Model loading
base_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
fine_tuned_tokenizer = AutoTokenizer.from_pretrained("brockwilson12/llama-2-7b-spanish-airport")
# Function for GPU Memory
def print_gpu_memory():
    allocated = torch.cuda.memory_allocated(0) / 1024**2
    reserved = torch.cuda.memory_reserved(0) / 1024**2
    print(f"Allocated GPU memory: {allocated:.2f} MB")
    print(f"Reserved GPU memory: {reserved:.2f} MB")
# Function to generate a response
def generate_response(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(**inputs, max_new_tokens=4000, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
# Function to calculate perplexity
def calculate_perplexity(model, tokenizer, text):
    tokens = tokenizer.encode(text, return_tensors='pt')
    with torch.no_grad():
        output = model(tokens, labels=tokens)
    loss = output.loss
    perplexity = torch.exp(loss)
    return perplexity.item()
# Define a prompt
prompt = "You are an American airport customs agent. You pulled the user aside because of a suspicion about identification. Do not speak English. Ask the user questions in relation to airport security. Respond without actions. Only speak in Spanish."

# Generate responses from both models
base_response = generate_response(base_model, base_tokenizer, prompt)
fine_tuned_response = generate_response(fine_tuned_model, fine_tuned_tokenizer, prompt)
'''
# Print the responses - Normal opening prompt.
print("Base Model Response:")
print(base_response)
print("\nFine-Tuned Model Response:")
print(fine_tuned_response)
'''
print_gpu_memory()
# Printing perplexities.
base_perplexity = calculate_perplexity(base_model, base_tokenizer, prompt)
fine_tuned_perplexity = calculate_perplexity(fine_tuned_model, fine_tuned_tokenizer, prompt)
print("Base Model Perplexity:", base_perplexity)
print("Fine-Tuned Model Perplexity:", fine_tuned_perplexity)


#----------------------------------
from nltk.translate.bleu_score import sentence_bleu
reference = [["Hola."]]
candidate = base_response.split()
bleu_score = sentence_bleu(reference, candidate)
print("Base Model BLEU Score:", bleu_score)
#----------------------------------
'''
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(reference_text, generated_text)
print("ROUGE Scores:", scores)
'''
#----------------------------------
from bleurt import score
bleurt_scorer = score.BleurtScorer('bleurt/bleurt-base-128')
scores = bleurt_scorer.score([candidate], [reference])
print("BLEURT Score:", scores[0])

