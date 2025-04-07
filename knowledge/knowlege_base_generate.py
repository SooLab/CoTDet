ds={}
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle
import re

# Setup model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B-Instruct" #replace with LLM of your choice
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Ensure the model is on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def run_chain_of_thought(selected_task):
    problem_statement = "I am a highly intelligent question answering bot and I answer questions from a human perspective."

    objects_prompt = """Q: Which common objects in daily life can be used as a tool for humans to {}?
    Please list 20 most suitable objects. Objects should be different from each other."""

    rationales_prompt = """For each object listed above, please explain the rationales for why they afford the task of {}
    from the perspective of visual features."""

    features_prompt = """For each object and its rationales, please summarize the corresponding visual features in one sentence,
    with comma-separated values of features, where each feature is described briefly.

    For example:
    1. [Object name]:
    Rationales: [Rationale from previous response]
    Visual Features: [feature 1], [feature 2], [feature 3], ..."""



    # Step 1: Get objects
    messages_step1 = [
        {"role": "system", "content": problem_statement},
        {"role": "user", "content": objects_prompt.format(selected_task)}
    ]
    input_text_step1 = tokenizer.apply_chat_template(messages_step1, tokenize=False)
    

    inputs_step1 = tokenizer.encode(input_text_step1, return_tensors="pt").to(device)
    outputs_step1 = model.generate(inputs_step1, max_new_tokens=1000, temperature=0.2, top_p=0.9, do_sample=True)
    response_step1 = tokenizer.decode(outputs_step1[0])
   

    # Step 2: Get rationales
    messages_step2 = [
        {"role": "system", "content": problem_statement},
        {"role": "user", "content": objects_prompt.format(selected_task)},
        {"role": "assistant", "content": response_step1},
        {"role": "user", "content": rationales_prompt.format(selected_task)}
    ]
    input_text_step2 = tokenizer.apply_chat_template(messages_step2, tokenize=False)


    inputs_step2 = tokenizer.encode(input_text_step2, return_tensors="pt").to(device)
    outputs_step2 = model.generate(inputs_step2, max_new_tokens=1500, temperature=0.2, top_p=0.9, do_sample=True)
    response_step2 = tokenizer.decode(outputs_step2[0])


    # Step 3: Get visual features
    messages_step3 = [
        {"role": "system", "content": problem_statement},
        {"role": "user", "content": objects_prompt.format(selected_task)},
        {"role": "assistant", "content": response_step1},
        {"role": "user", "content": rationales_prompt.format(selected_task)},
        {"role": "assistant", "content": response_step2},
        {"role": "user", "content": features_prompt}
    ]
    input_text_step3 = tokenizer.apply_chat_template(messages_step3, tokenize=False)

    inputs_step3 = tokenizer.encode(input_text_step3, return_tensors="pt").to(device)
    outputs_step3 = model.generate(inputs_step3, max_new_tokens=2000, temperature=0.2, top_p=0.9, do_sample=True)
    response_step3 = tokenizer.decode(outputs_step3[0])

    return response_step3

# List of tasks
tasks = ["step on", "sit comfortably", "place flowers", "get potatoes out of fire", "water plant", "get lemon out of tea", "dig hole", "open bottle of beer", "open parcel", "serve wine", "pour sugar", "smear butter", "extinguish fire", "pound carpet"]
tasks_prep = ["step on", "sit comfortably on", "place flowers in", "get potatoes out of fire with", "water plant with", "get lemon out of tea with", "dig hole with", "open bottle of beer with", "open parcel with", "serve wine with", "pour sugar with", "smear butter with", "extinguish fire with", "pound carpet with"]





from transformers import RobertaTokenizer, RobertaModel
import torch
r_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model1 = RobertaModel.from_pretrained('roberta-base')
model1.to(device)
def get_padded_embeddings(text, max_tokens=42):
    # Tokenize with device awareness
    inputs = r_tokenizer(
        text,
        return_tensors="pt",
        max_length=max_tokens,
        truncation=True,
        padding='max_length'
    ).to(device)  # Move inputs to correct device

    # Get model outputs
    with torch.no_grad():
        outputs = model1(**inputs)

    # Get embeddings
    word_embeddings = outputs.last_hidden_state.squeeze()
    sentence_embedding = word_embeddings.mean(dim=0)

    return {
        'text': text,
        'input_ids': inputs['input_ids'].squeeze().cpu().tolist(),  
        'attention_mask': inputs['attention_mask'].squeeze().cpu().tolist(),
        'word_embeddings': word_embeddings.cpu(),  # Move to CPU
        'sentence_embedding': sentence_embedding.cpu(),  # Move to CPU
        'total_tokens': max_tokens
    }

import re
n=len(tasks_prep)
for i in range(n):
  response_step3 = run_chain_of_thought(tasks_prep[i])
  features = re.findall(r"Visual Features: (.+)", response_step3)
  print(features)
  features = features[1:11]
  embedding_results = [get_padded_embeddings(elem) for elem in features]

  # Create word embeddings tensor with requires_grad=True
  word_emb = [result["word_embeddings"] for result in embedding_results]
  word_emb_tensor = torch.stack(word_emb)
  word_emb_tensor.requires_grad = True

  # Create sentence embeddings tensor with requires_grad=True
  sentence_emb = [result["sentence_embedding"] for result in embedding_results]
  sentence_emb_tensor = torch.stack(sentence_emb)
  sentence_emb_tensor.requires_grad = True

  # Store in ds dictionary
  ds[tasks[i]] = (word_emb_tensor, sentence_emb_tensor)

  print(f"DONE {tasks[i]}")
  with open('knowledge_base.pkl', 'wb') as handle:
    pickle.dump(ds, handle, protocol=pickle.HIGHEST_PROTOCOL)
