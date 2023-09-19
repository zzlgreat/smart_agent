

# base_model = "meta-llama/Llama-2-7b-hf"
base_model = "/DATA4T/text-generation-webui/models/Phind-CodeLlama-34B-v2"
# base_model = "meta-llama/Llama-2-13b-chat-hf"
# base_model = "meta-llama/Llama-2-70b-chat-hf"

import re

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import transformers
import torch
from torch.utils.data import DataLoader, Dataset
import torch
from transformers import AutoTokenizer, AutoConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Now you can create the model with the modified configuration
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map='balanced',
    # device_map="cpu",
    trust_remote_code=True,
    # cache_dir=cache_dir
)

tokenizer = AutoTokenizer.from_pretrained(base_model)
print("EOS token:", tokenizer.eos_token)
print("EOS token id:", tokenizer.eos_token_id)

print("Pad token: ", tokenizer.pad_token)
print("Pad token ID: ", tokenizer.pad_token_id)

tokenizer.padding_side='right'
print(tokenizer)

# Check if the pad token is already in the tokenizer vocabulary
if '|<pad>|' not in tokenizer.get_vocab():

    # Add the pad token
    tokenizer.add_tokens(['|<pad>|'])

# Set the pad token
tokenizer.pad_token = '|<pad>|'

# Resize token embeddings
model.resize_token_embeddings(len(tokenizer))

# Update pad token id in model and its config
model.pad_token_id = tokenizer.pad_token_id
model.config.pad_token_id = tokenizer.pad_token_id

# Check if they are equal
assert model.pad_token_id == tokenizer.pad_token_id, "The model's pad token ID does not match the tokenizer's pad token ID!"

# Print the pad token ids
print('Tokenizer pad token ID:', tokenizer.pad_token_id)
print('Model pad token ID:', model.pad_token_id)
print('Model config pad token ID:', model.config.pad_token_id)

# Print the model configuration
print(model.config)


# Sample string
# sample_string = ['hello [/INST]', 'my good friend</s>']
sample_string = ['[INST]']

# Tokenize the stringified JSON object
# encoded_sample = tokenizer(sample_string, truncation=True, padding=True, max_length=1024, return_tensors='pt', add_special_tokens=True)
encoded_sample = tokenizer(sample_string, truncation=True, padding=True, max_length=1024, return_tensors='pt', add_special_tokens=False)

# Count the number of tokens
token_count = len(encoded_sample)

# Fetch BOS and EOS tokens
BOS_token_id = tokenizer.bos_token_id
EOS_token_id = tokenizer.eos_token_id
BOS_token = tokenizer.decode([BOS_token_id])
EOS_token = tokenizer.decode([EOS_token_id])

# Check and print BOS and EOS tokens
print(f"Beginning of the sequence: {sample_string[0]} (BOS token: {BOS_token}, id: {BOS_token_id})")
print(f"End of the sequence: {sample_string[-1]} (EOS token: {EOS_token}, id: {EOS_token_id})")

print(f"The number of tokens in the string is: {token_count}")
print(f"The ids are: {encoded_sample}")

# Decode the input_ids
decoded_sample = tokenizer.decode(encoded_sample['input_ids'][0], skip_special_tokens=False)

# Print the decoded string
print(f"The decoded string is: {decoded_sample}")

# Print the attention mask
print(f"The attention mask is: {encoded_sample['attention_mask']}")

#only if you want to load with adapters (it can be faster to download a base model and add adapters)
from peft import PeftModel


from datasets import load_dataset

data = load_dataset(
    "Trelis/function_calling_extended",
    revision="explicit" # optionally specify a branch
)

# Modify the 'train' and 'test' datasets to keep only the rows from 53 to 55
data['train'] = data['train'].select(range(52, 55))
# data['test'] = data['test'].select(range(52, 55))

print(data)

class TextDataset(Dataset):
    def __init__(self, encodings, response_lengths, input_lengths):
        self.encodings = encodings
        self.response_lengths = response_lengths
        self.input_lengths = input_lengths

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}

        # Set labels to be the same as input_ids
        item["labels"] = item["input_ids"].clone()

        # Calculate the start and end positions of the response
        response_start_position = self.input_lengths[idx]
        response_end_position = self.input_lengths[idx] + self.response_lengths[idx]

        # Create a loss mask that covers only the response tokens
        item["loss_mask"] = torch.zeros_like(item["input_ids"])
        item["loss_mask"][response_start_position:response_end_position] = 1

        # Shift the loss mask to the left by one position
        shifted_loss_mask = torch.cat([item["loss_mask"][1:], torch.tensor([0])])
        item["loss_mask"] = shifted_loss_mask

        # Shift the labels to the left by one position
        item["labels"][:-1] = item["input_ids"][1:]

        # Replace the token after the response with an EOS token
        item["labels"][response_end_position - 1] = 2

        # Replace the token after the response with an 1 in the loss mask
        item["loss_mask"][response_end_position - 1] = 1

        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

def prepare_dataset(dataset, tokenizer):
    # Define the roles and markers
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    # Create the formatted text with the correct roles for each part of the dialogue
    formatted_dataset = dataset.map(
        lambda x: {
            "input_text": "".join([
                f"{B_INST} {B_SYS}{x['systemPrompt'].strip()}{E_SYS}",
                f"{x['userPrompt'].strip()} {E_INST}\n\n",
                f"{x['assistantResponse'].strip()}",  # appending the EOS token in TextData...
            ]),
            "response_text": "".join([
                f"{x['assistantResponse'].strip()}",  # appending the EOS token in TextData...
            ]),
        }
    )

    # Tokenize the datasets
    encodings = tokenizer([dialogue["input_text"] for dialogue in formatted_dataset], truncation=True, padding=True, max_length=1024, return_tensors='pt', add_special_tokens=True)

    # Tokenize the response one by one without padding and special tokens for the purpose of calculating length
    response_lengths = [len(tokenizer.encode(dialogue["response_text"], truncation=True, max_length=1024, padding=False, add_special_tokens=False)) for dialogue in formatted_dataset]

    # Tokenize the input one by one without padding and with the initial special token for the purpose of calculating length
    total_lengths = [len(tokenizer.encode(dialogue["input_text"], truncation=True, max_length=1024, padding=False, add_special_tokens=True)) for dialogue in formatted_dataset]
    input_lengths = [total_length - response_length for total_length, response_length in zip(total_lengths, response_lengths)]

    # Create TextDataset
    text_dataset = TextDataset(encodings, response_lengths, input_lengths)

    return text_dataset

# Apply function to your datasets
train_dataset = prepare_dataset(data['train'], tokenizer)
test_dataset = prepare_dataset(data['test'], tokenizer)

# Print the number of items in the dataset
print(f"Number of samples in the dataset: {len(train_dataset)}")

# Get a sample item
sample_item = train_dataset[1]  # replace 0 with the index of the sample you want to examine

# Print the dimensions of the sample item
print(f"Dimensions of input_ids: {sample_item['input_ids'].shape}")
print(f"Dimensions of attention_mask: {sample_item['attention_mask'].shape}")
print(f"Dimensions of loss_mask: {sample_item['loss_mask'].shape}")
print(f"Dimensions of labels: {sample_item['labels'].shape}")

# Print some tokens from the start and end of the sample
num_tokens_to_print = 336  # replace with the number of tokens you want to print

print("\nTokens at the start of the sample:")
print(sample_item['input_ids'][:num_tokens_to_print].tolist())
print(tokenizer.convert_ids_to_tokens(sample_item['input_ids'][:num_tokens_to_print].tolist()))

print("\nLabels at the start of the sample:")
print(sample_item['labels'][:num_tokens_to_print].tolist())
print(tokenizer.convert_ids_to_tokens(sample_item['labels'][:num_tokens_to_print].tolist()))

print("Attention mask at the start of the sample:")
print(sample_item['attention_mask'][:num_tokens_to_print].tolist())

print("Loss mask at the start of the sample:")
print(sample_item['loss_mask'][:num_tokens_to_print].tolist())

print("\nTokens at the end of the sample:")
print(sample_item['input_ids'][-num_tokens_to_print:].tolist())
print(tokenizer.convert_ids_to_tokens(sample_item['input_ids'][-num_tokens_to_print:].tolist()))

print("\nLabels at the end of the sample:")
print(sample_item['labels'][-num_tokens_to_print:].tolist())
print(tokenizer.convert_ids_to_tokens(sample_item['labels'][-num_tokens_to_print:].tolist()))

print("Attention mask at the end of the sample:")
print(sample_item['attention_mask'][-num_tokens_to_print:].tolist())

print("Loss mask at the end of the sample:")
print(sample_item['loss_mask'][-num_tokens_to_print:].tolist())


import textwrap
wrapper = textwrap.TextWrapper(width=80)
import re  # import regular expressions module
#%%
def generate(index):
    # Define the roles and markers
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    system_prompt = data['test'][index]['systemPrompt']
    user_prompt = data['test'][index]['userPrompt']
    correct_answer = data['test'][index]['assistantResponse']

    # Format your prompt template
    prompt = f"{B_INST} {B_SYS}{system_prompt.strip()}{E_SYS}{user_prompt.strip()} {E_INST}\n\n"

    print("Prompt:")
    print(prompt)

    encoding = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    output = model.generate(input_ids=encoding.input_ids,
                            attention_mask=encoding.attention_mask,
                            max_new_tokens=200,
                            do_sample=True,
                            temperature=0.01,
                            eos_token_id=tokenizer.eos_token_id,
                            top_k=0)

    print()

    # Subtract the length of input_ids from output to get only the model's response
    output_text = tokenizer.decode(output[0, len(encoding.input_ids[0]):], skip_special_tokens=False)
    output_text = re.sub('\n+', '\n', output_text)  # remove excessive newline characters

    print("Generated Assistant Response:")
    print(output_text)

    print()

    print("Correct Assistant Response:")
    print(correct_answer)

    print()
#%%
generate(14)
import torch.nn as nn

class CustomTrainer(transformers.Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        # Define the number of tokens you want to display
        num_tokens = 25  # This displays info on the actual and predicted tokens at the end of each sequence.

        labels = inputs.pop("labels")
        loss_mask = inputs.pop("loss_mask")

        # Forward pass
        outputs = model(**inputs)

        logits = outputs.logits

        # Check for NaN in logits and labels
        if torch.isnan(logits).any():
            print("NaN detected in logits")
            print(logits)

        # Convert logits to probabilities using softmax function
        probs = nn.functional.softmax(logits, dim=-1)

        # Get the most probable tokens
        predicted_token_ids = torch.argmax(probs, dim=-1)

        # Compute the loss
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        losses = loss_fct(logits.view(-1, self.model.config.vocab_size), labels.view(-1))

        # Reshaping the losses to have dimensions [batch_size, seq_length]
        losses = losses.view(-1, inputs['input_ids'].size(1))

        # Apply the loss mask
        masked_loss = losses * loss_mask

        # Check for NaN in losses and zero in loss_mask.sum()
        if torch.isnan(losses).any():
            print("NaN detected in losses")
            # print(losses)

        if loss_mask.sum() == 0:
            print("Sum of loss_mask is zero")
            return (torch.tensor(0).to(loss_mask.device), outputs) if return_outputs else torch.tensor(0).to(loss_mask.device)  # Early return

        # Aggregate the masked losses
        loss = masked_loss.sum() / (loss_mask.sum() + 1e-9)  # normalizing by the number of tokens considered + epsilon to prevent division by zero

        # Print formatted tokens
        batch_size, seq_length = inputs['input_ids'].size()

        # Useful for debugging training - recommend just training a small number of steps
        print("-" * 120)
        print(f"Token analysis for last {num_tokens} tokens:")
        header_format = "{:<10}{:<20}{:<20}{:<20}{:<20}{:<30}{:<30}".format("Index", "Input Token", "Predicted Token", "True Token", "Loss Mask", "Raw Loss", "Masked Loss")

        num_tokens=len(inputs['input_ids'][0])

        for batch_idx in range(batch_size):
            input_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][batch_idx])  # Using batch_idx
            predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_token_ids[batch_idx])  # Using batch_idx
            true_tokens = tokenizer.convert_ids_to_tokens(labels[batch_idx])  # Using batch_idx

            print(f"\nBatch {batch_idx + 1} of {batch_size}:")
            print(header_format)
            for i in range(-num_tokens, 0, 1):
                index = seq_length + i  # Correct index based on sequence length
                print("{:<10}{:<20}{:<20}{:<20}{:<20.1f}{:<30.6f}{:<30.6f}".format(index, input_tokens[index], predicted_tokens[index], true_tokens[index], loss_mask[batch_idx, i].item(), losses[batch_idx, i], masked_loss[batch_idx, i]))
            print("-" * 120)

        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self):
      train_dataset = self.train_dataset
      data_collator = self.data_collator

      dataloader_params = {
          "batch_size": self.args.train_batch_size,
          "collate_fn": data_collator,
          "num_workers": self.args.dataloader_num_workers,
          "pin_memory": self.args.dataloader_pin_memory,
      }

      if not isinstance(train_dataset, torch.utils.data.IterableDataset):
          dataloader_params["sampler"] = self._get_train_sampler()
          dataloader_params["drop_last"] = self.args.dataloader_drop_last

      return DataLoader(train_dataset, **dataloader_params)

class CustomDataCollator: # Needed if the EOS token is to be included in training.
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):

        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        loss_mask = torch.stack([item['loss_mask'] for item in batch])

        # # Debugging: print details of the first sequence in the batch
        # num_elements_to_view = 20  # Number of last elements to view

        # # Decoding the input_ids
        # decoded_input_tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

        # print("Debugging last", num_elements_to_view, "elements of the first sequence in the batch:")
        # print("{:<20}{:<20}{:<20}{:<20}".format("Token", "Input ID", "Label", "Loss Mask"))
        # for i in range(-num_elements_to_view, 0, 1):
        #   print("{:<20}{:<20}{:<20}{:<20}".format(decoded_input_tokens[i], input_ids[0, i].item(), labels[0, i].item(), loss_mask[0, i].item()))

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'loss_mask': loss_mask
        }

data_collator = CustomDataCollator(tokenizer)


trainer = CustomTrainer(
    model=model,
    train_dataset=train_dataset,
    #eval_dataset=test_dataset, #turn on the eval dataset if you want to use evaluation functionality versus the test dataset (not provided in this script)
    args=transformers.TrainingArguments(
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_grad_norm=1,
        warmup_ratio=0.1,
        # max_steps=1,
        learning_rate=1e-4,
        # fp16=True,
        logging_steps=1,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        lr_scheduler_type='cosine',
    ),
    data_collator=data_collator,
    # data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

trainer.train()

model.config.use_cache = True
model.eval()
generate(14)
# Custom prompt
def generate(index):
    # Define the roles and markers
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    systemPrompt = data['test'][index]['systemPrompt']
    user_prompt = 'Search Bill Gates, who is a billionaire' ### custom prompt here
    correct_answer = ''

    # Format your prompt template
    prompt = f"{B_INST} {B_SYS}{systemPrompt.strip()}{E_SYS}{user_prompt.strip()} {E_INST}\n\n"

    print("Prompt:")
    print(prompt)

    encoding = tokenizer(prompt, return_tensors="pt").to("cuda:0")
    output = model.generate(input_ids=encoding.input_ids,
                            attention_mask=encoding.attention_mask,
                            max_new_tokens=200,
                            do_sample=True,
                            temperature=0.01,
                            eos_token_id=tokenizer.eos_token_id,
                            top_k=0)

    print()

    # Subtract the length of input_ids from output to get only the model's response
    output_text = tokenizer.decode(output[0, len(encoding.input_ids[0]):], skip_special_tokens=False)
    output_text = re.sub('\n+', '\n', output_text)  # remove excessive newline characters

    print("Generated Assistant Response:")
    print(output_text)

    print()

    print("Correct Assistant Response:")
    print(correct_answer)

    print()
#%%
generate(1)

# Extract the last portion of the base_model
base_model_name = base_model.split("/")[-1]

# Define the save and push paths
adapter_model = "zzlgreat/34b_function_call_adapter"
new_model = 'zzlgreat/34b_function_call' #adjust 'Trelis' to your HuggingFace organisation

# Save the model
model.save_pretrained(adapter_model)

# Push the model to the hub
model.push_to_hub(adapter_model, use_auth_token=True)

from transformers import AutoModelForCausalLM, PretrainedConfig
import torch

# reload the base model (you might need a pro subscription for this because you may need a high RAM environment since this is loading the full original model, not quantized)
model = AutoModelForCausalLM.from_pretrained(base_model, device_map='balance', trust_remote_code=True, torch_dtype=torch.float16)

from peft import PeftModel

# load perf model with new adapters
model = PeftModel.from_pretrained(
    model,
    adapter_model,
)

model = model.merge_and_unload() # merge adapters with the base model.

model.push_to_hub(new_model, use_auth_token=True, max_shard_size="5GB")

#Push the tokenizer
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.push_to_hub(new_model, use_auth_token=True)
