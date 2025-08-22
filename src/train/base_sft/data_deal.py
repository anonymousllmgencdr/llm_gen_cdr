import json, pickle

import torch
from torch.utils.data import Dataset

import sys, os
from typing import Dict
from transformers import DataCollatorForSeq2Seq
from dataclasses import dataclass


import transformers

GEN_PROMPT = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
"""
ANSWER_PROMPT = """{}<|im_end|>"""



def pad_array(input_ids, pad_value, cutoff_len, flag):
    add_num = cutoff_len - len(input_ids)
    attention_mask = [1] * len(input_ids)
    if add_num == 0:
        return input_ids, attention_mask
    elif add_num < 0:

        raise ValueError("padding的时候出错了，检查dataset, 当前长度: {}".format(len(input_ids)))
    else:
        pad_ids = [pad_value for _ in range(add_num)]
        pad_attention_mask = [0 for _ in range(add_num)]
    if flag == "left":
        input_ids = pad_ids + input_ids
        attention_mask = pad_attention_mask + attention_mask
    else:
        input_ids = input_ids + pad_ids
        attention_mask = attention_mask + pad_attention_mask
    return input_ids, attention_mask


def pad_array_array(input_ids, pad_value, cutoff_len, flag):
    max_num = 0
    for val in input_ids:
        if len(val) > max_num:
            max_num = len(val)
    if max_num > cutoff_len:
        print("padding的时候超长, 当前长度: {}".format(max_num))
    res_input_ids, res_attention_mask = [], []
    for val in input_ids:
        now_input_ids, now_attention_mask = pad_array(val, pad_value, max_num, flag)

        # if max_num > cutoff_len:
        #     if flag == "left":
        #         now_input_ids = now_input_ids[-cutoff_len: ]
        #         now_attention_mask = now_attention_mask[-cutoff_len: ]
        #     elif flag == "right":
        #         now_input_ids = now_input_ids[: cutoff_len]
        #         now_attention_mask = now_attention_mask[: cutoff_len]

        res_input_ids.append(now_input_ids)
        res_attention_mask.append(now_attention_mask)
    return res_input_ids, res_attention_mask




def get_key_line_idx(input_path, watch_key=None):
    if os.path.exists(input_path + "_dict.pickle"):
        with open(input_path + "_dict.pickle", "rb") as fp:
            id_idx = pickle.load(fp)
        return id_idx
    id_idx = {}
    now_idx = 0
    now_count = 0
    with open(input_path, encoding="utf8") as f:
        line = f.readline()
        while line is not None and len(line) > 0:
            line = json.loads(line)
            if watch_key is None:
                id_idx[now_count] = now_idx
            else:
                id_idx[line[watch_key]] = now_idx
            now_idx = f.tell()
            line = f.readline()
            now_count += 1
            
    return id_idx


def load_key_data(input_path, idx):
    with open(input_path, encoding="utf8") as f:
        f.seek(idx)
        line = f.readline()
        line = json.loads(line)
    return line


class BaselineDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self, 
        data_path=None, 
        tokenizer = None,
        max_seq_len=1024, 
        need_pad=True,
    ):
        super(BaselineDataset, self).__init__()
        self.data_path = data_path
        self.train_data_keys = []
        if self.data_path is not None:
            self.train_data_idx = get_key_line_idx(self.data_path)
            self.train_data_keys = [k for k in self.train_data_idx.keys()]
        self.max_seq_len = max_seq_len
        self.tokenizer=tokenizer
        self.need_pad = need_pad


    def __len__(self):
        return len(self.train_data_keys)


    def preprocess(self, row_data):
        instruction = row_data["instruction"]
        label = row_data["output"]

        input_text = GEN_PROMPT.format(instruction)
        label_text = ANSWER_PROMPT.format(label)

        input_ids, labels = [], []
        token_ids = self.tokenizer.encode(input_text)
        input_ids.extend(token_ids)
        labels.extend([-100 for _ in token_ids])
        token_ids = self.tokenizer.encode(label_text)
        input_ids.extend(token_ids)
        labels.extend(token_ids)

        if self.need_pad:
            input_ids, attention_mask = pad_array(input_ids, self.tokenizer.pad_token_id, self.max_seq_len, "right")
            labels, _ = pad_array(labels, -100, self.max_seq_len, "right")
        else:
            attention_mask = [1 for _ in input_ids]
        
        result_data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return result_data

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        row_data_idx = self.train_data_keys[i]
        row_data = load_key_data(self.data_path, self.train_data_idx[row_data_idx])
        ret = self.preprocess(row_data)
        return ret



class BaselinePredictDataset(object):
    def __init__(
        self, 
        data_path=None, 
        tokenizer = None,
        max_seq_len=1024, 
    ):
        super(BaselinePredictDataset, self).__init__()
        self.data_path = data_path
        self.train_data_keys = []
        if self.data_path is not None:
            self.train_data_idx = get_key_line_idx(self.data_path)
            self.train_data_keys = [k for k in self.train_data_idx.keys()]
        self.max_seq_len = max_seq_len
        self.tokenizer=tokenizer

    def get_input(self, instruction):
        input_text = GEN_PROMPT.format(instruction)

        input_ids = []
        token_ids = self.tokenizer.encode(input_text)
        input_ids.extend(token_ids)
        attention_mask = [1 for _ in input_ids]

        return input_ids, attention_mask, input_text
        
        

@dataclass
class SpecialDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        return_fea = {

        }
        for nf in features:
            for k, v in nf.items():
                if k not in return_fea.keys():
                    return_fea[k] = []
                
                return_fea[k].append(v)
        all_key = [k for k in return_fea.keys()]
        for k in all_key:
            return_fea[k] = torch.LongTensor(return_fea[k])
        return return_fea



def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    max_query_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    data_path = data_args.dataset[0]

    train_dataset = BaselineDataset(
        data_path=data_path, 
        tokenizer = tokenizer,
        max_seq_len=max_query_len,
    )

    eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)




