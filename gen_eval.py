import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json, tqdm
import numpy as np
import torch
import faiss
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from FlagEmbedding import BGEM3FlagModel


class GenQWen(object):
    def __init__(self, model_path, max_new_tokens=512, max_model_len=2048, temperature=0.7, top_p=0.8,
                 gpu_memory_utilization=0.9, repetition_penalty=1.05, n=1):

        self.sampling_params = SamplingParams(n=n, temperature=temperature, top_p=top_p,
                                              repetition_penalty=repetition_penalty, max_tokens=max_new_tokens)
        self.model = LLM(model=model_path, max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
                         enable_lora=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt):
        texts = []
        if isinstance(prompt, str):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            texts.append(text)
        else:
            for val in prompt:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": val}
                ]
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                texts.append(text)
        response = []

        outputs = self.model.generate(texts, sampling_params=self.sampling_params)
        for output in outputs:
            now_res = []
            for idx in range(len(output.outputs)):
                now_res.append(output.outputs[idx].text)
            response.append(now_res)
        return response


class BGE_M3(object):
    def __init__(self, BGE_PATH, device="cuda:0"):
        self.model = BGEM3FlagModel(BGE_PATH, use_fp16=True, device=device)

    def get_embeddings(self, texts, batch_size=128, max_length=512):
        embeddings = self.model.encode(texts, batch_size=batch_size, max_length=max_length, )['dense_vecs']
        return embeddings

    def get_embedding(self, text, max_length=512):
        return self.get_embeddings([text], max_length=max_length)


class RecallUtilBatch(object):
    def __init__(self, texts, bge_model, max_length=1024, batch_size=128, flat_fun="IP"):

        self.max_length = max_length
        self.embedding_func = bge_model
        embeddings = self.embedding_func.get_embeddings(texts, max_length=self.max_length, batch_size=batch_size)
        embeddings = np.array(embeddings, dtype=np.float32)
        self.embeddings = embeddings
        print(embeddings.shape)
        if flat_fun == "IP":
            emb_index = faiss.IndexFlatIP(embeddings.shape[1])
        elif flat_fun == "L2":
            emb_index = faiss.IndexFlatL2(embeddings.shape[1])

        self.emb_index = emb_index
        self.emb_index.add(embeddings)

    def search_topn(self, texts=None, embedding=None, top=10):
        if texts is not None:
            if isinstance(texts, str):
                texts = [texts]
            text_embedding = self.embedding_func.get_embeddings(texts, max_length=self.max_length)
            text_embedding = text_embedding.astype(np.float32)
        else:
            text_embedding = embedding

        base_distances, base_indices = self.emb_index.search(text_embedding, top)

        retrun_info = {
            "base_distances": base_distances,
            "base_index": base_indices,
        }

        return retrun_info


def load_json(input_path):
    result_data = []
    with open(input_path, encoding="utf8") as f_in:
        for line in f_in:
            line = json.loads(line)
            result_data.append(line)
    return result_data


def generate_and_write_model_gen(gen_model, data_infos, output_path, watch_prompt_key="instruction", gen_num=5):
    if gen_num == 5:
        inputs1 = [val[watch_prompt_key].replace("推测用户可能感兴趣的10个", "推测用户可能感兴趣的5个") for val in
                   data_infos]
    else:
        inputs1 = [val[watch_prompt_key] for val in data_infos]
    res1 = gen_model.generate(inputs1)

    with open(output_path, "a", encoding="utf8") as f_out:
        for ori_json, product_info, input_info in zip(data_infos, res1, inputs1):
            product_info = product_info[0]

            data_flag = "test"
            if "data_flag" in ori_json.keys():
                data_flag = ori_json["data_flag"]
            wj = {
                "user_id": ori_json["user_id"],
                "product_names": ori_json["output"].split("、"),
                "gen_result": product_info,
                "data_flag": data_flag
            }

            f_out.write(json.dumps(wj, ensure_ascii=False) + "\n")


def get_predict(ckpt_path, input_path, output_path, gen_num=5):
    gen_model = GenQWen(
        model_path=ckpt_path,
    )

    with open(output_path, "w", encoding="utf8"):
        pass

    input_data = load_json(input_path)
    step = 1000
    for start in tqdm.tqdm(range(0, len(input_data), step)):
        now_deal = input_data[start: start + step]
        generate_and_write_model_gen(gen_model, now_deal, output_path, "instruction", gen_num=gen_num)
    del gen_model
    torch.cuda.empty_cache()


def eval_recall_hr_base(result_path, train_path=None, bge_model=None):
    name2simname = {}
    all_items = set()
    if train_path is not None and bge_model is not None:
        with open(train_path, encoding="utf8") as f_in:
            train_data = json.load(f_in)
            for val in train_data:
                output = val["output"].split("、")
                for tt in output:
                    all_items.add(tt)
        with open(result_path, encoding="utf8") as f_in:
            for line in f_in:
                line = json.loads(line)
                product_names = line["product_names"]
                for tt in product_names:
                    all_items.add(tt)
        not_found_name = set()
        with open(result_path, encoding="utf8") as f_in:
            for line in f_in:
                line = json.loads(line)
                gen_names = line["gen_result"].split("、")
                for tt in gen_names:
                    if tt not in all_items:
                        not_found_name.add(tt)
        all_items = list(all_items)
        not_found_name = list(not_found_name)
        print("oom num: {}".format(len(not_found_name)))
        if len(not_found_name) > 0:
            recall_util = RecallUtilBatch(texts=all_items, bge_model=bge_model, max_length=1024, batch_size=128,
                                          flat_fun="IP")
            recall_info = recall_util.search_topn(texts=not_found_name, top=10)
            base_index = recall_info["base_index"]
            for idx, nfname in enumerate(not_found_name):
                name2simname[nfname] = [all_items[sidx] for sidx in base_index[idx]]

    hr_test, recall_test = [], []
    hr_eval, recall_eval = [], []

    with open(result_path, encoding="utf8") as f_in:
        for line in tqdm.tqdm(f_in, total=80000):
            line = json.loads(line)
            data_flag = line["data_flag"]

            product_names = line["product_names"]
            gen_result = line["gen_result"].split("、")

            answer_names = set(product_names)
            new_gen_name = []
            for val in gen_result:
                if val in all_items or len(name2simname) == 0:
                    new_gen_name.append(val)
                else:
                    for simname in name2simname[val]:
                        if simname not in new_gen_name:
                            new_gen_name.append(simname)
                            break
            gen_name = set(new_gen_name)
            now_hr = len(gen_name & answer_names)

            if data_flag == "test":
                recall_test.append(now_hr / len(answer_names))
                hr_test.append(1 if now_hr > 0 else 0)
            else:
                recall_eval.append(now_hr / len(answer_names))
                hr_eval.append(1 if now_hr > 0 else 0)
    print("eval recall: {:.4f} hr:{:.4f} test recall:{:.4f} hr:{:.4f}".format(np.mean(recall_eval), np.mean(hr_eval),
                                                                              np.mean(recall_test), np.mean(hr_test)))


ckpt_path = "/checkpoint-{}"
test_path = "./deal_data/train_data/qwen_test.json"
train_path = "./deal_data/train_data/qwen_train.json"
output_path = "./deal_data/train_data/qwen_test_gen.json"
bge_m3_path = ""

bge_model = BGE_M3(BGE_PATH=bge_m3_path)

for ckpt in []:
    now_ckpt = ckpt_path.format(ckpt)
    get_predict(now_ckpt, test_path, output_path, gen_num=5)
    eval_recall_hr_base(output_path, train_path=train_path, bge_model=bge_model)
    get_predict(now_ckpt, test_path, output_path, gen_num=10)
    eval_recall_hr_base(output_path, train_path=train_path, bge_model=bge_model)
