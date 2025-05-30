import torch
from tqdm import tqdm

from language_model import load_model, load_template, get_prompt, generate, get_representation, MODEL_CONFIG



def fit_dataset(dataloader, model, tokenizer, new_template, abs_model, layer_idx, lambd=1):
    final_reps = []
    safe_rep_seqs = []
    labels = []

    data = dataloader.data
    for idx, (x, y) in enumerate(data):
        template = new_template()
        if isinstance(x, list):
            prompt = get_prompt(template, x[0], x[1], False)
        else:            
            prompt = get_prompt(template, x, None, True)
        rep = get_representation(model, tokenizer, prompt)

        final_rep = rep[:,-1,:]
        final_reps.append(
            torch.flatten(final_rep[layer_idx]).unsqueeze(0).to(model.device)
        )
        labels.append(y)

        rep_seq = rep[layer_idx, :, :].transpose(0,1)
        rep_seq = torch.flatten(rep_seq, start_dim=1, end_dim=2)
   
        if y == True:
            safe_rep_seqs.append(rep_seq)

    abs_model.fit(torch.concat(final_reps, dim=0), labels)
    abs_model.fit_transition(safe_rep_seqs, lambd)

def pred_dataset(dataloader, model, tokenizer, new_template, abs_model, layer_idx):
    labels = []
    preds = []

    data = dataloader.data

    for idx, (x, y) in enumerate(tqdm(data)):
        template = new_template()
        if isinstance(x, list):
            prompt = get_prompt(template, x[0], x[1], False)
            template = new_template()
            non_conv_prompt = get_prompt(template, x[0], None, True)
        else:            
            prompt = get_prompt(template, x, None, True)
            non_conv_prompt = None

        rep = get_representation(model, tokenizer, prompt)
        rep_seq = rep[layer_idx, :, :].transpose(0,1)
        rep_seq = torch.flatten(rep_seq, start_dim=1, end_dim=2)
        # rep_seqs.append(rep_seq)

        pred = abs_model.predict(rep_seq)
        if non_conv_prompt is not None:
            user_rep = get_representation(model, tokenizer, non_conv_prompt)
            user_rep_seq = user_rep[layer_idx, :, :].transpose(0,1)
            user_rep_seq = torch.flatten(user_rep_seq, start_dim=1, end_dim=2)
            prompt_pred = abs_model.predict(user_rep_seq)
            pred = min(pred, prompt_pred)
        if isinstance(pred, torch.Tensor):
            pred = pred.item()
        preds.append(pred)
        del rep_seq
        torch.cuda.empty_cache()
        
        labels.append(y)
    preds = torch.tensor(preds)
    
    return preds, labels

