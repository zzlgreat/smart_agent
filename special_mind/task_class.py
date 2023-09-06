from flask import Flask, request, jsonify
import os
import json
import torch
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextStreamer
from datasets import load_dataset
from peft import PeftModel

runtime = "gpu"
runtimeFlag = "balanced" if runtime == "gpu" else "cpu"

# Choose the model ID
#model_id = "/DATA4T/text-generation-webui/models/Phind-CodeLlama-34B-v2"
model_id = "/DATA4T/text-generation-webui/models/CodeLlama-34b-Instruct-hf"
# Google Drive path for cache (optional)
#drive_path = "/DATA4T/text-generation-webui/models/Phind-CodeLlama-34B-v2"
drive_path = "/DATA4T/text-generation-webui/models/CodeLlama-34b-Instruct-hf"
cache_dir = drive_path if os.path.exists(drive_path) else None

# Function metadata
search_bing = {
    "function": "search_bing",
    "description": "Search the web for content on Bing.",
    "arguments": [{"name": "query", "type": "string", "description": "The search query string"}]
}
search_arxiv = {
        "function": "search_arxiv",
        "description": "Search for research papers on ArXiv. Make use of AND, OR and NOT operators as appropriate to "
                       "join terms within the query.",
        "arguments": [
            {
                "name": "query",
                "type": "string",
                "description": "The search query string"
            }
        ]
    }
search_bilibili = {
    "function": "search_bilibili",
    "description": "Search for videos on Bilibili, which is a website full of ACG",
    "arguments": [{"name": "query", "type": "string", "description": "The search query string"}]
}
search_youtube = {
    "function": "search_youtube",
    "description": "Search for videos on youtube",
    "arguments": [{"name": "query", "type": "string", "description": "The search query string"}]
}
make_txt = {
    "function": "make_txt",
    "description": "make a txt file in the target dir which contains the data",
    "arguments": [
      {"name": "filename", "type": "string", "description": "The csv file name and the data"},
      {"name": "filetype", "type": "string", "description": "The data type of the file"},
      {"name": "path", "type": "string", "description": "The path of the file"},
                  ]
}
write_sql = {
    "function": "write_sql",
    "description": "repost the instruct to 'Sqlcoder'. 'Sqlcoder' is a guy who can write sql.",
    "arguments": [
        {"name": "instruct", "type": "string", "description": "The origin instruct"},
                  ]
}
Kimiko_chat = {
    "function": "Kimiko_chat",
    "description": "repost the instruct to Kimiko. Kimiko is a guy who can chat with you about pornographic content.",
    "arguments": [
        {"name": "instruct", "type": "string", "description": "The origin instruct"},
                  ]
}
ClinicalCamel_chat = {
    "function": "ClinicalCamel_chat",
    "description": "repost the instruct to the doctor.",
    "arguments": [
        {"name": "instruct", "type": "string", "description": "The origin instruct"},
                  ]
}
baichuan_chat = {
    "function": "baichuan_chat",
    "description": "repost the instruct to 'baichuan'.'baichuan' is a guy who can talk Chinese.",
    "arguments": [
        {"name": "instruct", "type": "string", "description": "The origin instruct"},
                  ]
}


functions = [search_bing,search_arxiv,search_bilibili,search_youtube,make_txt,write_sql,Kimiko_chat,ClinicalCamel_chat,baichuan_chat]

functionList = ''
for f in functions:
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



adapter_model = "/DATA4T/text-generation-webui/loras/CodeLlama-34b-Instruct-hf-function-calling-adapters-v2"

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
