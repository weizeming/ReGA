from datasets import load_dataset
from random import shuffle

class DataLoader():
    def __init__(self):
        self.data = []
        self.name = None
        
    def load_dataset(self, dataset, tranform_func, max_num=None):
        for idx, sample in enumerate(dataset):
            tranform_func(self.data, sample)

            if max_num is not None and len(self.data) >= max_num:
                break

    def clear_data(self):
        self.data = []
    
    def shuffle_data(self):
        shuffle(self.data)

    def merge(self, dataloader, max_num=None, start=0):
        if max_num is not None:
            self.data += dataloader.data[start:start+max_num]
        else:
            self.data += dataloader.data[start:]

    def split(self, start, num):
        self.data = self.data[start: start+num]
        return self
    
def trans_hhi_safe(data, sample):
    if sample['label'][0] is True:
        data.append((sample['sentence'][0], sample['label'][0]))
    elif sample['label'][1] is True:
        data.append((sample['sentence'][1], sample['label'][1]))

def trans_hhi_harmful(data, sample):
    if sample['label'][0] is False:
        data.append((sample['sentence'][0], sample['label'][0]))
    elif sample['label'][1] is False:
        data.append((sample['sentence'][1], sample['label'][1]))


def trans_sorrybench(data, sample):
    data.append((sample['turns'][0], False))



def trans_jbb(data, sample):
    data.append((sample['Goal'], False))
def trans_jbb_conv(data, sample):
    data.append(([sample['Goal'],sample['Target']], False))
        
def trans_alpaca(data, sample):
    if sample['input'] == '':
        data.append((sample['instruction'], True))
    else:
        data.append((sample['instruction'] + '\n' + sample['input'], True))

def trans_alpaca_conv(data, sample):
    if sample['input'] == '':
        data.append((
            [sample['instruction'], 
             sample['output']],
            True
        ))
    else:
        data.append((
            [sample['instruction'] + '\n' + sample['input'],
             sample['output']],
            True
        ))
    
def trans_advbench(data, sample):
    data.append((sample['prompt'], False))

def trans_adv_conv(data, sample):
    data.append((
        [sample['prompt'], sample['target']],
        False
        ))
    
def trans_harmbench(data, sample):
    data.append((sample['prompt'], False))

def trans_wildjailbreak(data, sample):
    if sample['data_type'] == 'adversarial_harmful':
        data.append((sample['adversarial'], False))

def trans_chat(data, sample):
    content = sample['conversation'][0]["content"]
    data.append((content, True))

def trans_chat_conv(data, sample):
    content_1 = sample['conversation'][0]["content"]
    content_2 = sample['conversation'][1]["content"]
    data.append((
        [content_1, content_2],
        True
    ))

def trans_cbt_conv(data, sample):
    response = sample['response'].split('\n')[0]
    data.append((
        [sample['prompt'], response],
        False
        ))
def trans_mtbench(data, sample):
    data.append((sample['prompt'][0], True))
    data.append((sample['prompt'][1], True))

    
DATASET_CONFIG = {
    "sorrybench": {
        "path": "/data/hf_datasets/sorrybench",
        "transform": trans_sorrybench
    },
    "jailbreakbench": {
        "path": "/data/hf_datasets/jailbreakbench",
        "transform": trans_jbb
    },
    "jailbreakbench_conv": {
        "path": "/data/hf_datasets/jailbreakbench",
        "transform": trans_jbb_conv
    },
    "alpaca": {
        "path": "/data/hf_datasets/alpaca",
        "transform": trans_alpaca
    },
    "advbench": {
        "path": "/data/hf_datasets/advbench",
        "transform": trans_advbench
    },
    "alpaca_conv": {
        "path": "/data/hf_datasets/alpaca",
        "transform": trans_alpaca_conv,
    },
    "adv_conv": {
        "path": "/data/hf_datasets/advbench",
        "transform": trans_adv_conv
    },
    "harmbench": {
        "path": "/data/hf_datasets/harmbench",
        "transform": trans_harmbench
    },
    "wildjailbreak": {
        "path": "/data/hf_datasets/wildjailbreak",
        "transform": trans_wildjailbreak
    },
    "chat1m": {
        "path": '/data/hf_datasets/chat1m/data',
        "transform": trans_chat
    },
    "chat1m_conv": {
        "path": '/data/hf_datasets/chat1m/data',
        "transform": trans_chat_conv
    },
    "cbt_conv": {
        "path": "/data/hf_datasets/cb_train/data",
        "transform": trans_cbt_conv
    },
    "hhi_safe": {
        "path": "/data/hf_datasets/harmful_harmless_instructions/data",
        "transform": trans_hhi_safe
    },
    "hhi_harmful": {
        "path": "/data/hf_datasets/harmful_harmless_instructions/data",
        "transform": trans_hhi_harmful
    },
    "mtbench": {
        "path": "/data/hf_datasets/mtbench/data",
        "transform": trans_mtbench
    },
}

def load_hf_dataset(dataset_name, max_num=None, split_name='train', subset_name=None, key=None, value=None):
    
    if subset_name is not None:
        dataset = load_dataset(DATASET_CONFIG[dataset_name]['path'], subset_name)
        train = dataset[split_name]
    else:
        dataset = load_dataset(DATASET_CONFIG[dataset_name]['path'])
        train = dataset[split_name]
    if key is not None:
        if isinstance(value, list):
            train = train.filter(lambda x: int(x[key]) >= value[0] and int(x[key]) <= value[1] and x['prompt_style'] == 'base')
        else:
            train = train.filter(lambda x: x[key] == value)
    loader = DataLoader()
    transform = DATASET_CONFIG[dataset_name]['transform']
    loader.load_dataset(train, transform, max_num)
    loader.name = dataset_name
    return loader

def get_all_loaders(max_num=1000):
    loaders = {}
    loaders['alpaca'] = load_hf_dataset("alpaca", max_num, 'train')
    loaders['alpaca_conv'] = load_hf_dataset("alpaca_conv", max_num, 'train')
    loaders['advbench'] = load_hf_dataset("advbench", max_num)
    loaders['advbench_conv'] = load_hf_dataset("adv_conv", max_num, 'train')

    loaders['harmbench'] = load_hf_dataset("harmbench", max_num, 'train', 'standard')
    loaders['jailbreakbench'] = load_hf_dataset("jailbreakbench", max_num, 'harmful', 'behaviors')
    loaders['jailbreakbench_conv'] = load_hf_dataset("jailbreakbench_conv", max_num, 'harmful', 'behaviors')
    loaders['hhi_harmful'] = load_hf_dataset("hhi_harmful", max_num, 'test')
    loaders['cbt_conv'] = load_hf_dataset("cbt_conv", max_num)
    
    loaders['hhi_safe'] = load_hf_dataset("hhi_safe", max_num, 'test')
    loaders['chat1m'] = load_hf_dataset("chat1m", max_num)
    loaders['chat1m_conv'] = load_hf_dataset("chat1m_conv", max_num)
    
    loaders['sorrybench'] = load_hf_dataset("sorrybench", max_num)
    loaders['wildjailbreak'] = load_hf_dataset("wildjailbreak", max_num, 'train', 'eval')
    loaders['mtbench'] = load_hf_dataset("mtbench", max_num)
    return loaders
    
def load_sorrybench_attacks(attack_name, max_num):
    loader = load_hf_dataset("sorrybench", max_num, split_name='train', subset_name=None, key="prompt_style", value=attack_name)
    loader.name = f"sorrybench_{attack_name}"
    return loader
    
def load_sorrybench_perspectives(perspective_name, max_num):
    if perspective_name == 'hate_speech':
        values = [1,5]
    elif perspective_name == 'crimes_or_torts':
        values = [6,24]
    elif perspective_name == 'inappropriate_topics':
        values = [25,39]
    elif perspective_name == 'unqualified_advice':
        values = [40,44]    
    else:
        raise ValueError    
    
    loader = load_hf_dataset("sorrybench", max_num, split_name='train', subset_name=None, key="category", value=values)
    loader.name = f"sorrybench_{perspective_name}"
    return loader
    
