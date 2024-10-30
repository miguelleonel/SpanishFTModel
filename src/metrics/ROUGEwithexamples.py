import json
import numpy as np
from rouge_score import rouge_scorer

# File paths
generated_file = "1KConvosLlamaBASEMODEL.jsonl"
ground_truth_file = "1KConvosOpenAIextracted.jsonl"

# Load the JSONL files
with open(generated_file, 'r') as f_generated, open(ground_truth_file, 'r') as f_ground_truth:
    generated_data = [json.loads(line) for line in f_generated]
    ground_truth_data = [json.loads(line) for line in f_ground_truth]

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Initialize lists to store ROUGE scores and metadata for each example
rouge1_scores = []
rouge2_scores = []
rougeL_scores = []
metadata = []  # To keep track of custom_id, user inputs, and responses

# Counter for ROUGE-1 F1 score of 0.00
rouge1_zero_count = 0

# Process each pair of ground truth and model response
for gen, truth in zip(generated_data, ground_truth_data):
    # Extract relevant information
    custom_id = gen["custom_id"]
    model_responses = gen["model_responses"]
    user_inputs = truth["user_inputs"]
    ground_truth_responses = truth["agent_responses"]
    
    for i, (model_response, ground_truth_response, user_input) in enumerate(zip(model_responses, ground_truth_responses, user_inputs)):
        # Compute ROUGE scores
        scores = scorer.score(ground_truth_response, model_response)
        rouge1_scores.append(scores['rouge1'].fmeasure)
        rouge2_scores.append(scores['rouge2'].fmeasure)
        rougeL_scores.append(scores['rougeL'].fmeasure)
        
        # Count occurrences of ROUGE-1 F1 score of 0.00
        if scores['rouge1'].fmeasure == 0.0:
            rouge1_zero_count += 1
        
        # Store metadata for this example
        metadata.append({
            'custom_id': custom_id,
            'turn': i + 1,
            'user_input': user_input,
            'ground_truth': ground_truth_response,
            'model_response': model_response
        })

# Convert scores to numpy arrays for easier handling
rouge1_scores = np.array(rouge1_scores)
rouge2_scores = np.array(rouge2_scores)
rougeL_scores = np.array(rougeL_scores)

# Calculate average ROUGE scores
average_rouge1 = np.mean(rouge1_scores)
average_rouge2 = np.mean(rouge2_scores)
average_rougeL = np.mean(rougeL_scores)

# Print average scores
print(f"Average ROUGE-1 F1 Score: {average_rouge1:.4f}")
print(f"Average ROUGE-2 F1 Score: {average_rouge2:.4f}")
print(f"Average ROUGE-L F1 Score: {average_rougeL:.4f}")

# Print count of ROUGE-1 F1 scores equal to 0.00
print(f"\nCount of ROUGE-1 F1 Score of 0.00: {rouge1_zero_count}")

# Sort by ROUGE-1 scores to find top and bottom examples
sorted_data = sorted(zip(rouge1_scores, metadata), key=lambda x: x[0])

# Output the top 5 highest-scoring examples
print("\nTop 5 highest-scoring examples:")
for score, meta in sorted_data[-5:]:
    print(f"Custom ID: {meta['custom_id']} | Turn: {meta['turn']}")
    print(f"User Input: {meta['user_input']}")
    print(f"Ground Truth: {meta['ground_truth']}")
    print(f"Model Response: {meta['model_response']}")
    print(f"ROUGE-1 F1 Score: {score:.4f}\n")

# Output the bottom 5 lowest-scoring examples
print("\nBottom 5 lowest-scoring examples:")
for score, meta in sorted_data[:5]:
    print(f"Custom ID: {meta['custom_id']} | Turn: {meta['turn']}")
    print(f"User Input: {meta['user_input']}")
    print(f"Ground Truth: {meta['ground_truth']}")
    print(f"Model Response: {meta['model_response']}")
    print(f"ROUGE-1 F1 Score: {score:.4f}\n")

