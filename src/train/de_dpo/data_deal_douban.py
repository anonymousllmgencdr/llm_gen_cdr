import json, pickle

import torch
from torch.utils.data import Dataset

import sys, os
from typing import Dict
from transformers import DataCollatorForSeq2Seq
from dataclasses import dataclass


import random
import transformers

prompt = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
{}<|im_end|>"""

TEXT1 = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
"""

TEXT2 = """<|im_end|>
<|im_start|>assistant
"""

TEXT3 = """<|im_end|>"""


def pad_array(input_ids, pad_value, cutoff_len, flag):
    add_num = cutoff_len - len(input_ids)
    attention_mask = [1] * len(input_ids)
    if add_num == 0:
        return input_ids, attention_mask
    elif add_num < 0:
        if flag == "right":
            return input_ids[:cutoff_len], attention_mask[:cutoff_len]
        else:
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






class OuterloopDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self, 
        data_path=None, 
        tokenizer=None,
        max_seq_len=1024, 
        need_pad=True,
    ):
        super(OuterloopDataset, self).__init__()
        self.data_path = data_path
        self.train_data = []
        with open(data_path, "r", encoding="utf-8") as f_in:
            for line in f_in:
                line = json.loads(line)
                self.train_data.append(line)
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.need_pad = need_pad

    def __len__(self):
        return len(self.train_data)

    def split_instruction(self, text):
        all_split_texts = [TEXT1]
        all_categotys = [0]

        texts = text.split("\n\n")
        for val in texts:
            # 指令 type 1
            if val.startswith("你是豆瓣网站的用户兴趣预测专家"):
                all_split_texts.append(val + "\n\n")
                all_categotys.append(1)
            elif val.startswith("请基于以上用户在音乐和书籍领域的兴趣"):
                all_split_texts.append(val)
                all_categotys.append(1)
            # 指令 type 1
            elif val.startswith("用户感兴趣的"):
                val_split2 = val.split("\n")
                for val2 in val_split2:
                    if val.startswith("用户感兴趣的音乐的标签为:"):
                        all_split_texts.append("用户感兴趣的音乐的标签为:")
                        all_categotys.append(1)
                        all_split_texts.append(val.replace("用户感兴趣的音乐的标签为:", "") + "\n")
                        all_categotys.append(2)
                        
                    elif val.startswith("用户感兴趣的书籍的标签为:"):
                        all_split_texts.append("用户感兴趣的书籍的标签为:")
                        all_categotys.append(1)
                        all_split_texts.append(val.replace("用户感兴趣的书籍的标签为:", "") + "\n")
                        all_categotys.append(3)

                all_split_texts.append("\n")
                all_categotys.append(1)
            
        all_split_texts.append(TEXT2)
        all_categotys.append(0)
        return all_split_texts, all_categotys




    def preprocess(self, row_data):
        instruction = row_data["instruction"]
        output_chosen = row_data["output_chosen"]
        output_rejected = row_data["output_rejected"]
        

        all_split_texts, all_categotys = self.split_instruction(instruction)
        input_ids, categorys_ids, labels = [], [], []
        for split_text, category in zip(all_split_texts, all_categotys):
            token_ids = self.tokenizer.encode(split_text)
            input_ids.extend(token_ids)
            categorys_ids.extend([category for _ in token_ids])
            labels.extend([-100 for _ in token_ids])

        out_text_chosen = output_chosen + TEXT3
        token_ids = self.tokenizer.encode(out_text_chosen)
        input_ids_chosen = input_ids + token_ids
        categorys_ids_chosen = categorys_ids + [0 for _ in token_ids]
        labels_chosen = labels + token_ids
        

        out_text_rejected = output_rejected + TEXT3
        token_ids = self.tokenizer.encode(out_text_rejected)
        input_ids_rejected = input_ids + token_ids
        categorys_ids_rejected = categorys_ids + [0 for _ in token_ids]
        labels_rejected = labels + token_ids
        
        assert len(input_ids_chosen) == len(categorys_ids_chosen), "数据处理有问题"
        assert len(input_ids_chosen) == len(labels_chosen), "数据处理有问题"
        assert len(input_ids_rejected) == len(categorys_ids_rejected), "数据处理有问题"
        assert len(input_ids_rejected) == len(labels_rejected), "数据处理有问题"
        
        if self.need_pad is True:
            input_ids_chosen, attention_mask_chosen = pad_array(input_ids_chosen, self.tokenizer.pad_token_id, self.max_seq_len, "right")
            categorys_ids_chosen, _ = pad_array(categorys_ids_chosen, 0, self.max_seq_len, "right")
            labels_chosen, _ = pad_array(labels_chosen, -100, self.max_seq_len, "right")
            
            input_ids_rejected, attention_mask_rejected = pad_array(input_ids_rejected, self.tokenizer.pad_token_id, self.max_seq_len, "right")
            categorys_ids_rejected, _ = pad_array(categorys_ids_rejected, 0, self.max_seq_len, "right")
            labels_rejected, _ = pad_array(labels_rejected, -100, self.max_seq_len, "right")
        else:
            attention_mask_chosen = [1 for _ in input_ids_chosen]
            attention_mask_rejected = [1 for _ in input_ids_rejected]
            

        result_data = {
            "input_ids_chosen": input_ids_chosen,
            "attention_mask_chosen": attention_mask_chosen,
            "token_type_ids_chosen": categorys_ids_chosen,
            "labels_chosen": labels_chosen,
            "input_ids_rejected": input_ids_rejected,
            "attention_mask_rejected": attention_mask_rejected,
            "token_type_ids_rejected": categorys_ids_rejected,
            "labels_rejected": labels_rejected,
            
        }
        return result_data

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        try:
            ret = self.preprocess(self.train_data[i])
        except Exception as e:
            ret = self.preprocess(self.train_data[i+1])
        # ret["idx"] = [i]
        return ret

class PredictDataset(Dataset):
    def __init__(
        self,  
        tokenizer=None, 
    ):
        super(PredictDataset, self).__init__()
        self.tokenizer = tokenizer


    def split_instruction(self, text):
        all_split_texts = [TEXT1]
        all_categotys = [0]

        texts = text.split("\n\n")
        for val in texts:
            # 指令 type 1
            if val.startswith("你是豆瓣网站的用户兴趣预测专家"):
                all_split_texts.append(val + "\n\n")
                all_categotys.append(1)
            elif val.startswith("请基于以上用户在音乐和书籍领域的兴趣"):
                all_split_texts.append(val)
                all_categotys.append(1)
            # 指令 type 1
            elif val.startswith("用户感兴趣的"):
                val_split2 = val.split("\n")
                for val2 in val_split2:
                    if val.startswith("用户感兴趣的音乐的标签为:"):
                        all_split_texts.append("用户感兴趣的音乐的标签为:")
                        all_categotys.append(1)
                        all_split_texts.append(val.replace("用户感兴趣的音乐的标签为:", "") + "\n")
                        all_categotys.append(2)
                        
                    elif val.startswith("用户感兴趣的书籍的标签为:"):
                        all_split_texts.append("用户感兴趣的书籍的标签为:")
                        all_categotys.append(1)
                        all_split_texts.append(val.replace("用户感兴趣的书籍的标签为:", "") + "\n")
                        all_categotys.append(3)

                all_split_texts.append("\n")
                all_categotys.append(1)
            
        all_split_texts.append(TEXT2)
        all_categotys.append(0)
        return all_split_texts, all_categotys


    def process_texts(self, texts):
        all_input_ids, all_attention_mask, all_categorys_ids = [], [], []
        for now_text in texts:
            all_split_texts, all_categotys = self.split_instruction(now_text)
            input_ids, categorys_ids = [], []
            for split_text, category in zip(all_split_texts, all_categotys):
                token_ids = self.tokenizer.encode(split_text)
                input_ids.extend(token_ids)
                categorys_ids.extend([category for _ in token_ids])
            all_input_ids.append(input_ids)
            all_categorys_ids.append(categorys_ids)
        all_input_ids, all_attention_mask = pad_array_array(all_input_ids, 0, cutoff_len=4096, flag="left")
        all_categorys_ids, _ = pad_array_array(all_categorys_ids, 0, cutoff_len=4096, flag="left")
        return all_input_ids, all_attention_mask, all_categorys_ids




@dataclass
class SpecialDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        return_fea = {

        }
        return_fea2 = {}
        for nf in features:
            for k, v in nf.items():
                if k not in return_fea.keys():
                    return_fea[k] = []
                    tk = "_".join(k.split("_")[: -1])
                    return_fea2[tk] = []
                
                return_fea[k].append(v)
        for d_name in ["_chosen", "_rejected"]:
            for k in return_fea2.keys():
                return_fea2[k].extend(return_fea[k+d_name])

        all_key = [k for k in return_fea2.keys()]
        for k in all_key:
            return_fea2[k] = torch.LongTensor(return_fea2[k])
        return return_fea2


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    max_query_len,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    data_path = data_args.dataset[0]

    train_dataset = OuterloopDataset(
        data_path=data_path,
        max_seq_len=max_query_len,
        tokenizer=tokenizer
    )

    eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)






