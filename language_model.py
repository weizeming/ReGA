import os
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["TRANSFORMERS_OFFLINE"] = "1"
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
device = 'cuda'
from fastchat.model import get_conversation_template



MODEL_CONFIG = {
    'vicuna': 'lmsys/vicuna-7b-v1.5',
    'llama': 'meta-llama/Llama-2-7b-chat-hf',
    'qwen': '/data/models/models--Qwen--Qwen2.5-7B-Instruct',
    'mistral': '/data/models/models--mistralai--Mistral-7B-Instruct-v0.3',
    'koala': '/data/models/koala',
    'baichuan': '/data/models/baichuan'
}

def load_model(model_path,):
    model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                trust_remote_code=True,
            ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False
    )

    return model, tokenizer

def load_template(model_path):
    return get_conversation_template(model_path)


def get_prompt(template, query, response=None, no_chat=False):
    template.append_message('user', query)
    if response is None and not no_chat:
        template.append_message('assistant', '')
    else:
        template.append_message('assistant', response)
    return template.get_prompt()


def generate(model, tokenizer, prompt, generated_only=True ,max_length=256):
    with torch.no_grad():
        tokenized_input = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
        attn_masks = torch.ones_like(tokenized_input).to(model.device)
        output = model.generate(tokenized_input, max_new_tokens=max_length, attention_mask=attn_masks,pad_token_id=tokenizer.pad_token_id,do_sample=False)
        if generated_only is True:
            decoded_output = tokenizer.decode(output[0][len(tokenized_input):])
        else:
            decoded_output = tokenizer.decode(output[0])
        return decoded_output


def get_representation(model, tokenizer, prompt):
    with torch.no_grad():
        tokenized_input = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        output = model(tokenized_input, output_hidden_states=1)
        representation = output.hidden_states
        concat_representation = torch.concat(representation).cpu()
        return concat_representation # [layers, tokens, dimensions]



