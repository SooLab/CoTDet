from openai import OpenAI

client = OpenAI(api_key=YOUR_OPENAI_KEY)

tasks_name = ["step on", "sit comfortably", "place flowers", "get potatoes out of fire", "water plant", "get lemon out of tea", "dig hole", "open bottle of beer", "open parcel", "serve wine", "pour sugar", "smear butter", "extinguish fire", "pound carpet"]

# add prepositions manually
tasks = ["step on", "sit comfortably on", "place flowers in", "get potatoes out of fire with", "water plant with", "get lemon out of tea with", "dig hole with", "open bottle of beer with", "open parcel with", "serve wine with", "pour sugar with", "smear butter with", "extinguish fire with", "pound carpet with"]

problem_statement = "I am a highly intelligent question answering bot and I answer questions from a human perspective. Given the target task, I will list candidate objects in daily life that can be used as a vehicle for it, think about the rationales for why they afford the task and the corresponding visual features. Finally, summarize the corresponding visual features in one sentence. The description of each object is best distinguished from the other."

CoT_zero_shot = "Q: Which common objects in daily life that can be used as a vehicle for human to {}? Please list 20 most suitable objects. For each object, let\'s think the rationales why they afford the task from the perspective of visual features and summary corresponding visual features of the object for each rationale. \n"

output_format_prompt = "The answer of each object should be in two parts. The first part includes as many rationales as possible and the second part is the summary of corresponding visual features for each rationale in one sentence. Note that the summary in one sentence should not include the object names.\nA:"

update_prompt = [
    {"role": "system", "content": problem_statement},
    {"role": "user", "content": CoT_zero_shot.format(tasks[8]) + output_format_prompt},
]

print(update_prompt)

# take task 0 "open parcel" as an example
# We used text-davinci-003 for paper, however, gpt-3.5-turbo is now easier to use.
COMPLETIONS_MODEL = "gpt-3.5-turbo-0125"
temperature = 0.1
max_tokens = 2048

response = client.chat.completions.create(
    model=COMPLETIONS_MODEL,
    messages=update_prompt,
    max_tokens=max_tokens,
    temperature=temperature,
)
answer = response.choices[0].message

print(answer)

# Sample output

# 1. Scissors: 
# Rationales: Scissors have two sharp blades that can be used to cut through the tape or plastic wrap on the parcel. 
# Visual Features: Sharp blades and two handles. 

# 2. Knife: 
# Rationales: Knives have a sharp blade that can be used to cut through the tape or plastic wrap on the parcel. 
# Visual Features: Sharp blade and a handle. 

# ...