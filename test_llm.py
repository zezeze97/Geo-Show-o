from models.modeling_qwen2 import Qwen2ForCausalLM
from transformers import AutoTokenizer

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
device = "cuda" # the device to load the model onto

model = Qwen2ForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f'num of llm_vocab is: {len(tokenizer)}')
print(f'bos: {tokenizer.bos_token}')
print(f'eos: {tokenizer.eos_token}')
print(f'pad: {tokenizer.pad_token}')

prompt = "Find the value of $x$ that satisfies the equation $4x+5 = 6x+7$."

text = f"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think> <answer> answer here </answer>.<｜User｜>{prompt}<｜Assistant｜>"
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=4096,
    use_cache=True
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]
print(f'Response is: {response}')
