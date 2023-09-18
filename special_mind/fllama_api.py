from flask import Flask, request, jsonify
import os
import json
import torch
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
from datasets import load_dataset
from peft import PeftModel
from real_world import toolkit
runtime = "gpu"
runtimeFlag = "auto" if runtime == "gpu" else "cpu"

# Choose the model ID
model_id = "/DATA4T/text-generation-webui/models/CodeLlama-34b-Instruct-hf"
# Google Drive path for cache (optional)
drive_path = "/DATA4T/text-generation-webui/models/CodeLlama-34b-Instruct-hf"
cache_dir = drive_path if os.path.exists(drive_path) else None
adapter_model = "/DATA4T/text-generation-webui/loras/CodeLlama-34b-Instruct-hf-function-calling-adapters-v2"
# Function metadata

toolkits = toolkit.toolkits
functionList =''
for f in toolkits:
    functionList += json.dumps(f, indent=4, separators=(',', ': '))

# ================== Model and Tokenizer Setup ================== #

# Load the model
if runtime == "gpu":
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map=runtimeFlag, cache_dir=cache_dir)
    #model = AutoModelForCausalLM.from_pretrained(model_id,  device_map=runtimeFlag, cache_dir=cache_dir)
else:
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map=runtimeFlag, cache_dir=cache_dir)

# load perf model with new adapters
model = PeftModel.from_pretrained(
    model,
    adapter_model,
)

# Initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir, use_fast=True)


# ================== Inference ================== #

# Stream function with function calling capabilities
def stream_with_function(user_prompt):
    prompt = f"<FUNCTIONS>{functionList.strip()}</FUNCTIONS>[INST] {user_prompt.strip()} [/INST]\n\n"
    inputs = tokenizer([prompt], return_tensors="pt").to('cuda')
    # streamer = TextStreamer(tokenizer)
    output = model.generate(**inputs, max_length=1000)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Stream function without function calling capabilities
def stream_without_function(user_prompt):
    prompt = f"[INST] <<SYS>>You are a helpful assistant that provides accurate and concise responses<</SYS>>{user_prompt.strip()} [/INST]\n\n"
    inputs = tokenizer([prompt], return_tensors="pt").to('cuda')
    output = model.generate(**inputs, max_length=500)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

app = Flask(__name__)

# Wrap your function in an API endpoint
@app.route('/stream_with_function', methods=['POST'])
def api_stream_with_function():
    try:
        data = request.json
        user_prompt = data['user_prompt']
        result = stream_with_function(user_prompt)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stream_without_function', methods=['POST'])
def api_stream_without_function():
    try:
        data = request.json
        user_prompt = data['user_prompt']
        result = stream_without_function(user_prompt)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7784)
