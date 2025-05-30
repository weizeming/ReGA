import torch
import os
import argparse
from language_model import load_model, load_template, get_prompt, generate, get_representation, MODEL_CONFIG
from abstract_model import *
from data import load_hf_dataset, get_all_loaders, DataLoader, load_sorrybench_attacks, load_sorrybench_perspectives
from datasets import load_dataset
from evaluation import fit_dataset, pred_dataset
import json

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["TRANSFORMERS_OFFLINE"] = "1"

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, required=True)
    parser.add_argument('--model-name', choices=['all', 'vicuna', 'llama', 'qwen', 'mistral', 'koala', 'baichuan'], default='all')
    parser.add_argument('--abs-model', choices=['rep', 'uni', 'mlp', 'lmj', 'ppl'], default='rep')
    # parser.add_argument('--train-loaders', choices=)
    parser.add_argument('--test-loaders', choices=['all',
                                                   'iid_test',
                                                   'perspectives',
                                                   'harmbench', 'jailbreakbench', 'jailbreakbench_conv', 'hhi_harmful', 'cbt_conv', 
                                                   'hhi_safe', 'chat1m', 'chat1m_conv', 'sorrybench', 'wildjailbreak'], default='all')
    parser.add_argument('--layer-num', type=int, default=1)
    parser.add_argument('--layer-id', type=int, default=16)
    parser.add_argument('--pca-dim', type=int, default=8)
    parser.add_argument('--state-num', type=int, default=32)
    parser.add_argument('--safe-data', default=256, type=int)
    parser.add_argument('--harmful-data', default=64, type=int)
    parser.add_argument('--test-data', default=1000, type=int)
    parser.add_argument('--threshold', type=str, choices=['mca', 'mfn', 'auroc', 'all'], default='all')
    parser.add_argument('--verbal', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    layers = slice(args.layer_id, args.layer_id + 1)
    
    if args.abs_model == 'rep':
        abs_model = RepModel(args.state_num, args.pca_dim)
    elif args.abs_model == 'mlp':
        abs_model = MlpModel()
    elif args.abs_model == 'lmj':
        abs_model = LmjModel()
    elif args.abs_model == 'ppl':
        abs_model = PplModel()
    elif args.abs_model == 'uni':
        abs_model = UniModel(args.state_num, args.pca_dim)
    else:
        raise NotImplementedError
    
    if args.model_name == 'all':
        model_names = ['vicuna', 'llama', 'qwen', 'mistral', 'koala', 'baichuan']
    else:
        model_names = [args.model_name]
    
    loaders = get_all_loaders(args.test_data)
    
    train_loader = DataLoader()    
    adv_loader = loaders['advbench']
    adv_conv_loader = loaders['advbench_conv']
    alpaca_loader = loaders['alpaca']
    alpaca_conv_loader = loaders['alpaca_conv']
    
    train_loader.merge(adv_loader, args.harmful_data)
    train_loader.merge(adv_conv_loader, args.harmful_data)
    train_loader.merge(alpaca_loader, args.safe_data)
    train_loader.merge(alpaca_conv_loader, args.safe_data)

    test_loaders = []

    if args.test_loaders == 'all':
        for test_name in ['harmbench', 'jailbreakbench', 'hhi_harmful', 'jailbreakbench_conv',
                         'mtbench', 'chat1m', 'hhi_safe', 'chat1m_conv',
                         'wildjailbreak']:
            test_loaders.append(loaders[test_name])
        for attack in ['technical_terms', 'misspellings','authority_endorsement', 'role_play']:
            test_loaders.append(
                load_sorrybench_attacks(attack, args.test_data)
            )
    elif args.test_loaders == 'iid_test':
        adv_loader.split(args.harmful_data, args.test_data),
        adv_conv_loader.split(args.harmful_data, args.test_data),
        alpaca_loader.split(args.safe_data, args.test_data),
        alpaca_conv_loader.split(args.safe_data, args.test_data)
        
        prompt_loader = DataLoader()
        prompt_loader.merge(adv_loader)
        prompt_loader.merge(alpaca_loader)
        prompt_loader.name = 'Prompt'
        conv_loader = DataLoader()
        conv_loader.merge(adv_conv_loader)
        conv_loader.merge(alpaca_conv_loader)
        conv_loader.name = 'Conv'
        test_loaders = [
            prompt_loader, conv_loader
        ]
    elif args.test_loaders == 'perspectives':
        for perspective in ['hate_speech', 'crimes_or_torts','inappropriate_topics', 'unqualified_advice']:
            test_loaders.append(
                load_sorrybench_perspectives(perspective, args.test_data)
            )
    else:
        test_loaders = [loaders[args.test_loaders]]

    if args.threshold == 'all':
        threshold = ['mca', 'mfn'] 
    elif args.threshold == 'auroc':
        threshold = ['auroc', 'mca', 'mfn'] 
    else:
        threshold = [args.threshold]

    log = abs_model.auto(
        model_names,
        train_loader,
        test_loaders,
        layers,
        threshold,
        args.verbal
    )
    log['params'] = vars(args)
    fname = f"{args.fname}_{args.abs_model}_{args.model_name}"
    with open(f'logs/{fname}.json', 'w') as f:
        json.dump(log, f, indent=4)