
import json, os
import random
import numpy as np
from collections import Counter

seed = 42
random.seed(seed)
np.random.seed(seed)

input_path_book = "./deal_data/bookreviews_cleaned_rating.json"
input_path_movie = "./deal_data/moviereviews_cleaned_rating.json"
input_path_music = "./deal_data/musicreviews_cleaned_rating.json"

input_path_book_info = "./deal_data/bookitem_info_all.json"
input_path_movie_info = "./deal_data/movieitem_info_all.json"
input_path_music_info = "./deal_data/musicitem_info_all.json"


def sort_ab(a, b):
    assert len(a) == len(b), "length error"
    zipped = list(zip(a, b))
    zipped.sort(key=lambda x: x[1], reverse=True)
    a_sorted, b_sorted = zip(*zipped)
    return list(a_sorted), list(b_sorted)


def load_info(data_name, flag):
    print("load {} {}".format(flag, data_name))

    if data_name == "movie":
        input_inter_path = input_path_movie
        input_info_path = input_path_movie_info
    elif data_name == "book":
        input_inter_path = input_path_book
        input_info_path = input_path_book_info

    elif data_name == "music":
        input_inter_path = input_path_music
        input_info_path = input_path_music_info

    with open(input_inter_path, 'r', encoding='utf-8') as file:
        input_inter_data = json.load(file)
    with open(input_info_path, 'r', encoding='utf-8') as file:
        input_info_data = json.load(file)

    info_data = {val["item_id"]: val for val in input_info_data}
    out_data = {}
    for val in input_inter_data:
        user_id = val["user_id"]
        item_id = val["item_id"]
        timestamp = val["timestamp"]
        item_id, timestamp = sort_ab(item_id, timestamp)
        new_item_id, new_rating, new_timestamp = [], [], []

        new_tags = []
        for idx in range(len(item_id)):
            if item_id[idx] in info_data.keys():
                if info_data[item_id[idx]]["tags"] is None or len(info_data[item_id[idx]]["tags"]) == 0:
                    continue
                new_tags.append(info_data[item_id[idx]]["tags"])

                new_item_id.append(item_id[idx])
                new_timestamp.append(timestamp[idx])

        if len(new_item_id) > 0:
            wj = {
                "user_id": user_id,
                "item_id": new_item_id,
                "timestamp": new_timestamp,
                "tag": new_tags
            }
            out_data[user_id] = wj
    print("\tdata loading completed. number of users: {}".format(len(out_data)))
    return out_data


def load_all_find_user(source_domins, target_domin):
    target_data = load_info(target_domin, "target")
    source_data = {}
    for val in source_domins:
        source_data[val] = load_info(val, "source")

    data_info = {}
    for user_id in target_data.keys():
        target_info = target_data[user_id]
        sd_info = {}
        for sd in source_domins:
            if user_id in source_data[sd].keys() and len(source_data[sd][user_id]["item_id"]) > 0:
                sd_info[sd] = source_data[sd][user_id]
        if len(sd_info.keys()) > 0:
            data_info[user_id] = {"source": sd_info, "target": target_info}

    return data_info



def load_all_find_user_topk_split(source_domins, target_domin, split=False):
    user_infos = load_all_find_user(source_domins, target_domin)
    new_user_infos = {}
    for user_id in user_infos.keys():
        target_info = user_infos[user_id]["target"]
        source_info = user_infos[user_id]["source"]
        now_user_step = 0
        while len(target_info["item_id"]) > 0:
            max_time = target_info["timestamp"][0]

            min_time = max_time - 60 * 60 * 24 * 90

            new_target_data = {
                "item_id": [target_info["item_id"][idx] for idx in range(len(target_info["item_id"])) if
                            target_info["timestamp"][idx] <= max_time and target_info["timestamp"][idx] >= min_time],
                "timestamp": [val for val in target_info["timestamp"] if val <= max_time and val >= min_time],
                "tag": [target_info["tag"][idx] for idx in range(len(target_info["tag"])) if
                        target_info["timestamp"][idx] <= max_time and target_info["timestamp"][idx] >= min_time],
            }

            new_target_info = {
                "item_id": [target_info["item_id"][idx] for idx in range(len(target_info["item_id"])) if
                            target_info["timestamp"][idx] < min_time],
                "timestamp": [val for val in target_info["timestamp"] if val < min_time],
                "tag": [target_info["tag"][idx] for idx in range(len(target_info["tag"])) if
                        target_info["timestamp"][idx] < min_time],
            }


            target_info = new_target_info

            source_max_time = max_time
            source_min_time = max_time - 60 * 60 * 24 * 365

            new_source_data = {}
            for sd in source_domins:
                if sd not in source_info.keys():
                    continue
                now_source_data = {
                    "item_id": [source_info[sd]["item_id"][idx] for idx in range(len(source_info[sd]["item_id"])) if
                                source_info[sd]["timestamp"][idx] < source_max_time and source_info[sd]["timestamp"][
                                    idx] >= source_min_time],
                    "timestamp": [val for val in source_info[sd]["timestamp"] if
                                  val < source_max_time and val >= source_min_time],
                    "tag": [source_info[sd]["tag"][idx] for idx in range(len(source_info[sd]["tag"])) if
                            source_info[sd]["timestamp"][idx] < source_max_time and source_info[sd]["timestamp"][
                                idx] >= source_min_time],
                }
                if len(now_source_data["item_id"]) > 0:
                    new_source_data[sd] = now_source_data
            if len(new_source_data.keys()) > 0 and len(new_target_data["tag"]) > 0:
                new_user_infos["{}_step_{}".format(user_id, now_user_step)] = {
                    "user_id": "{}_step_{}".format(user_id, now_user_step), "source": new_source_data,
                    "target": new_target_data}

            if split is False:
                break
            now_user_step += 1
    return new_user_infos


def collect_top_n(tags_list, n=30):
    counter = Counter()
    for val in tags_list:
        counter.update(val)
    tags = [k for k, v in counter.most_common(n)]

    return tags


def collect_top_n_titletag(tags_list, n=30):
    counter = Counter()
    for val in tags_list:
        new_val = []
        for tt in val:
            tt_tmp = tt.replace("0", "").replace("1", "").replace("2", "").replace("3", "").replace("4", "").replace(
                "5", "").replace("6", "").replace("7", "").replace("8", "").replace("9", "").replace(" ", "")
            if len(tt_tmp) == 0:
                continue
            if "、" in tt:
                if len(tt) < 10:
                    tt = tt.replace("、", "")
                else:
                    continue
            new_val.append(tt)
        counter.update(new_val)

    tags = [k for k, v in counter.most_common(10000) if v > 2]

    return tags



def get_split_byuser_split(sources, target, base_output_path=None, split=False):
    user_all_info = load_all_find_user_topk_split(sources, target, split=split)

    all_source_items = set()
    all_target_items = set()

    all_source_items_idformat = set()

    now_train_data = []
    now_val_data = []
    now_test_data = []

    now_train_data_idformat = []
    now_val_data_idformat = []
    now_test_data_idformat = []

    target_title_counter = Counter()
    uid2data = {}

    for uid, now_data in user_all_info.items():
        tmp_uid = uid.split("_step_")
        now_uid = tmp_uid[0]
        step = int(tmp_uid[1])
        if now_uid not in uid2data.keys():
            uid2data[now_uid] = {}
        uid2data[now_uid][step] = now_data
    for uid, uid_data in uid2data.items():
        all_step = sorted(uid_data.keys())
        test_find, val_find = False, False
        for now_idx, now_step in enumerate(all_step):
            now_data = uid_data[all_step[now_idx]]
            step_uid = now_data["user_id"]
            target_title = collect_top_n_titletag(now_data["target"]["tag"], n=10)
            for val in target_title:
                if "、" in val:
                    print("tag 、:", val)
                all_target_items.add(val)
            if len(target_title) == 0:
                continue
            sk_tags = {}
            sk_ids = {}
            for sd in now_data["source"].keys():
                now_tag = collect_top_n(now_data["source"][sd]["tag"], 30)
                now_tag = ["{}__{}".format(sd, val) for val in now_tag]
                for val in now_tag:
                    all_source_items.add(val)
                for val in now_data["source"][sd]["item_id"]:
                    all_source_items_idformat.add(val)
                sk_tags[sd] = now_tag
                sk_ids[sd] = now_data["source"][sd]["item_id"]

            if test_find is False:
                now_test_data.append([step_uid, sk_tags, target_title])
                now_test_data_idformat.append([step_uid, sk_ids, target_title])
                test_find = True
            elif val_find is False:
                now_val_data.append([step_uid, sk_tags, target_title])
                now_val_data_idformat.append([step_uid, sk_ids, target_title])
                val_find = True
            else:
                now_train_data.append([step_uid, sk_tags, target_title])
                now_train_data_idformat.append([step_uid, sk_ids, target_title])


    tag_count = {}
    train_count = 0
    for val in now_train_data:
        if len(val[2]) == 0:
            continue
        train_count += 1
        now_title = val[2]
        now_title = list(set(now_title))
        for nti in now_title:
            if nti not in tag_count.keys():
                tag_count[nti] = 0
            tag_count[nti] += 1
    need_deal = set()
    all_title_tags = set()
    for tag in tag_count.keys():
        if tag_count[tag] / train_count > 0.2:
            need_deal.add(tag)
        else:
            all_title_tags.add(tag)

    all_target_items = [val for val in all_target_items if val not in need_deal]
    need_deal = list(need_deal)
    new_now_train_data = []
    new_now_train_data_idformat = []
    del_count, del_count_id = 0, 0
    for val in now_train_data:
        uid, sk_tags, target_title = val[0], val[1], val[2]
        n_target_title = [tt for tt in target_title if tt not in need_deal]
        n_target_title = [tt for tt in n_target_title if tt in all_target_items]
        if len(n_target_title) > 0:
            new_now_train_data.append([uid, sk_tags, n_target_title])
        else:
            del_count += 1
    for val in now_train_data_idformat:
        uid, sk_tags, target_title = val[0], val[1], val[2]
        n_target_title = [tt for tt in target_title if tt not in need_deal]
        n_target_title = [tt for tt in n_target_title if tt in all_target_items]
        if len(n_target_title) > 0:
            new_now_train_data_idformat.append([uid, sk_tags, n_target_title])
        else:
            del_count_id += 1
    now_train_data = new_now_train_data
    now_train_data_idformat = new_now_train_data_idformat

    new_now_val_data = []
    new_now_val_data_idformat = []
    del_count, del_count_id = 0, 0
    for val in now_val_data:
        uid, sk_tags, target_title = val[0], val[1], val[2]
        n_target_title = [tt for tt in target_title if tt not in need_deal]
        n_target_title = [tt for tt in n_target_title if tt in all_target_items]
        if len(n_target_title) > 0:
            new_now_val_data.append([uid, sk_tags, n_target_title])
        else:
            del_count += 1
    for val in now_val_data_idformat:
        uid, sk_tags, target_title = val[0], val[1], val[2]
        n_target_title = [tt for tt in target_title if tt not in need_deal]
        n_target_title = [tt for tt in n_target_title if tt in all_target_items]
        if len(n_target_title) > 0:
            new_now_val_data_idformat.append([uid, sk_tags, n_target_title])
        else:
            del_count_id += 1
    now_val_data = new_now_val_data
    now_val_data_idformat = new_now_val_data_idformat

    new_now_test_data = []
    new_now_test_data_idformat = []
    del_count = 0
    for val in now_test_data:
        uid, sk_tags, target_title = val[0], val[1], val[2]
        n_target_title = [tt for tt in target_title if tt not in need_deal]
        n_target_title = [tt for tt in n_target_title if tt in all_target_items]
        if len(n_target_title) > 0:
            new_now_test_data.append([uid, sk_tags, n_target_title])
        else:
            del_count += 1
    for val in now_test_data_idformat:
        uid, sk_tags, target_title = val[0], val[1], val[2]
        n_target_title = [tt for tt in target_title if tt not in need_deal]
        n_target_title = [tt for tt in n_target_title if tt in all_target_items]
        if len(n_target_title) > 0:
            new_now_test_data_idformat.append([uid, sk_tags, n_target_title])
        else:
            del_count_id += 1
    now_test_data = new_now_test_data
    now_test_data_idformat = new_now_test_data_idformat

    all_target_items = [val for val in all_target_items if val not in need_deal]

    print("train:{} eval:{} test:{}".format(len(now_train_data), len(now_val_data), len(now_test_data)))
    print("tag source item:{} target item: {}".format(len(all_source_items), len(all_target_items)))
    print("id sourceitem:{} target item: {}".format(len(all_source_items_idformat), len(all_target_items)))

    write_base_idfromat(all_source_items_idformat, all_target_items, now_train_data_idformat, now_val_data_idformat,
                        now_test_data_idformat, base_output_path)
    write_qwen_data_more(now_train_data, now_val_data, now_test_data, target_title_counter, base_output_path)


def write_qwen_data_more(now_train_data, now_val_data, now_test_data, target_title_counter, base_output_path):
    qwen_train_data = []
    qwen_test_data = []

    qwen_train_path = base_output_path + "/qwen_train.json"
    qwen_test_path = base_output_path + "/qwen_test.json"

    for val in now_train_data:

        uid, sk_tags, movie_titles = val[0], val[1], val[2]

        prompt = START_PROMPT
        ad_w = 0
        for sd, tags in sk_tags.items():
            tag = [tag.split("__")[1] for tag in tags]
            if sd == "music":
                prompt += "用户感兴趣的音乐的标签为: {}\n".format("、".join(tag))
                ad_w += 1
            elif sd == "book":
                prompt += "用户感兴趣的书籍的标签为: {}\n".format("、".join(tag))
                ad_w += 1
            elif sd == "movie":
                prompt += "用户感兴趣的电影的标签为: {}\n".format("、".join(tag))
                ad_w += 1
            else:
                print("{} data not found".format(sd), tag)
                exit(0)
        if ad_w == 0:
            print("data error", sk_tags, movie_titles)
        prompt_format = prompt + END_PROMPT
        for choice_title in movie_titles:
            wj = {
                "user_id": uid,
                "output": "、".join([choice_title]),
                "input": "",
                "instruction": prompt_format.format(1)
            }
            qwen_train_data.append(wj)
        if len(movie_titles) < 15:
            wj = {
                "user_id": uid,
                "output": "、".join(movie_titles),
                "input": "",
                "instruction": prompt_format.format(len(movie_titles))
            }
            qwen_train_data.append(wj)
        if len(movie_titles) > 6:
            idxs = [val for val in range(2, min(20, len(movie_titles)) - 1)]
            selected_end = random.sample(idxs, 3)
            for end_idx in selected_end:
                wj = {
                    "user_id": uid,
                    "output": "、".join(movie_titles[: end_idx]),
                    "input": "",
                    "instruction": prompt_format.format(len(movie_titles[:end_idx]))
                }
                qwen_train_data.append(wj)

    for val in now_val_data:
        uid, sk_tags, movie_titles = val[0], val[1], val[2]

        prompt = START_PROMPT
        ad_w = 0
        for sd, tags in sk_tags.items():
            tag = [tag.split("__")[1] for tag in tags]
            if sd == "music":
                prompt += "用户感兴趣的音乐的标签为: {}\n".format("、".join(tag))
                ad_w += 1
            elif sd == "book":
                prompt += "用户感兴趣的书籍的标签为: {}\n".format("、".join(tag))
                ad_w += 1
            elif sd == "movie":
                prompt += "用户感兴趣的电影的标签为: {}\n".format("、".join(tag))
                ad_w += 1
            else:
                print("{} data not found".format(sd), tag)
                exit(0)
        if ad_w == 0:
            print("data error", sk_tags, movie_titles)
        prompt_format = prompt + END_PROMPT
        wj = {
            "user_id": uid,
            "output": "、".join(movie_titles),
            "input": "",
            "instruction": prompt_format.format(10),
            "data_flag": "val"
        }
        qwen_test_data.append(wj)

    for val in now_test_data:
        uid, sk_tags, movie_titles = val[0], val[1], val[2]

        prompt = START_PROMPT
        ad_w = 0
        for sd, tags in sk_tags.items():
            tag = [tag.split("__")[1] for tag in tags]
            if sd == "music":
                prompt += "用户感兴趣的音乐的标签为: {}\n".format("、".join(tag))
                ad_w += 1
            elif sd == "book":
                prompt += "用户感兴趣的书籍的标签为: {}\n".format("、".join(tag))
                ad_w += 1
            elif sd == "movie":
                prompt += "用户感兴趣的电影的标签为: {}\n".format("、".join(tag))
                ad_w += 1
            else:
                print("{} data not found".format(sd), tag)
                exit(0)
        if ad_w == 0:
            print("data error", sk_tags, movie_titles)
        prompt_format = prompt + END_PROMPT
        wj = {
            "user_id": uid,
            "output": "、".join(movie_titles),
            "input": "",
            "instruction": prompt_format.format(10),
            "data_flag": "test"
        }
        qwen_test_data.append(wj)

    with open(qwen_test_path, "w", encoding="utf8") as f_out:
        for val in qwen_test_data:
            f_out.write(json.dumps(val, ensure_ascii=False) + "\n")
    json.dump(qwen_train_data, open(qwen_train_path, mode='w', encoding="utf8"), indent=4, ensure_ascii=False)



def write_base_idfromat(all_source_items, all_target_items, now_train_data, now_val_data, now_test_data,base_output_path):
    inter_header = "user_id:token\titem_id:token\ttrainflag:float"
    user_header = "user_id:token"
    item_header = "item_id:token"
    base_output_path = base_output_path
    if not os.path.exists(base_output_path):
        os.makedirs(base_output_path)

    if not os.path.exists(base_output_path + "/source/"):
        os.makedirs(base_output_path + "/source/")
    if not os.path.exists(base_output_path + "/target/"):
        os.makedirs(base_output_path + "/target/")

    f_out_source_item = open(base_output_path + "/source/source.item", "w", encoding="utf8")
    f_out_source_user = open(base_output_path + "/source/source.user", "w", encoding="utf8")
    f_out_source_inter = open(base_output_path + "/source/source.inter", "w", encoding="utf8")
    f_out_source_inter_with_val = open(base_output_path + "/source/source.inter_with_val", "w", encoding="utf8")

    f_out_target_item = open(base_output_path + "/target/target.item", "w", encoding="utf8")
    f_out_target_user = open(base_output_path + "/target/target.user", "w", encoding="utf8")
    f_out_target_inter = open(base_output_path + "/target/target.inter", "w", encoding="utf8")

    f_out_source_item.write(item_header + "\n")
    f_out_target_item.write(item_header + "\n")
    f_out_source_user.write(user_header + "\n")
    f_out_target_user.write(user_header + "\n")
    f_out_source_inter.write(inter_header + "\n")
    f_out_target_inter.write(inter_header + "\n")
    f_out_source_inter_with_val.write(inter_header + "\n")

    for val in all_source_items:
        f_out_source_item.write("source_{}\n".format(val))
    for val in all_target_items:
        f_out_target_item.write("target_{}\n".format(val))
    all_user_id = set()

    for val in now_train_data:
        uid, sk_tags, movie_titles = val[0], val[1], val[2]
        all_user_id.add(uid)
        for sk, tags in sk_tags.items():
            for tag in tags:
                f_out_source_inter.write("{}\tsource_{}\t{}\n".format(uid, tag, 0))
                f_out_source_inter_with_val.write("{}\tsource_{}\t{}\n".format(uid, tag, 0))
        for movie_title in movie_titles:
            f_out_target_inter.write("{}\ttarget_{}\t{}\n".format(uid, movie_title, 0))

    for val in now_val_data:
        uid, sk_tags, movie_titles = val[0], val[1], val[2]
        all_user_id.add(uid)
        for sk, tags in sk_tags.items():
            for tag in tags:
                f_out_source_inter.write("{}\tsource_{}\t{}\n".format(uid, tag, 0))
        for movie_title in movie_titles:
            f_out_target_inter.write("{}\ttarget_{}\t{}\n".format(uid, movie_title, 1))

        for sk, tags in sk_tags.items():
            random.shuffle(tags)
            random.shuffle(tags)
            for idx, tag in enumerate(tags):
                if idx < len(tags) // 2:
                    f_out_source_inter_with_val.write("{}\tsource_{}\t{}\n".format(uid, tag, 0))
                else:
                    f_out_source_inter_with_val.write("{}\tsource_{}\t{}\n".format(uid, tag, 1))

    for val in now_test_data:
        uid, sk_tags, movie_titles = val[0], val[1], val[2]
        all_user_id.add(uid)
        for sk, tags in sk_tags.items():
            for tag in tags:
                f_out_source_inter.write("{}\tsource_{}\t{}\n".format(uid, tag, 0))
        for movie_title in movie_titles:
            f_out_target_inter.write("{}\ttarget_{}\t{}\n".format(uid, movie_title, 2))

        for sk, tags in sk_tags.items():
            random.shuffle(tags)
            random.shuffle(tags)
            for idx, tag in enumerate(tags):
                if idx < len(tags) // 2:
                    f_out_source_inter_with_val.write("{}\tsource_{}\t{}\n".format(uid, tag, 0))
                else:
                    f_out_source_inter_with_val.write("{}\tsource_{}\t{}\n".format(uid, tag, 1))

    for val in all_user_id:
        f_out_source_user.write("{}\n".format(val))
        f_out_target_user.write("{}\n".format(val))



target = "movie"
sources = ["music", "book"]
START_PROMPT = "你是豆瓣网站的用户兴趣预测专家，我会给你提供一个用户在豆瓣平台上音乐和书籍领域的兴趣，请你结合这些信息进行全方面地分析，分析给出用户可能感兴趣的电影信息，以下是用户的基础信息和行为信息：\n\n"
END_PROMPT = "\n请基于以上用户在音乐和书籍领域的兴趣，推测用户可能感兴趣的{}个电影标签，直接给出标签名，不要有额外描述。"


base_output_path = "./deal_data/train_data"
get_split_byuser_split(sources, target, base_output_path=base_output_path, split=True)




