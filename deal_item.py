import pandas as pd
import json, tqdm


def load_uid2id(data_name):
    if data_name == "music":
        data_path = "data/music_cleaned.txt"
        nameid = "music_id"
    elif data_name == "book":
        data_path = "data/books_cleaned.txt"
        nameid = "book_id"
    data_item = pd.read_csv(data_path, sep='\t', skipinitialspace=True)
    uid2id = dict(zip(data_item["UID"], data_item[nameid]))
    return uid2id


def deal_item(data_name):
    if data_name == "book":
        input_path = "data/bookreviews_cleaned.txt"
        output_path = "deal_data/bookitem_info_all.json"
        id_name = "book_id"
        name_name = None
        tag_name = "labels"
        uid2id = load_uid2id(data_name)
    elif data_name == "music":
        input_path = "data/musicreviews_cleaned.txt"
        output_path = "deal_data/musicitem_info_all.json"
        id_name = "music_id"
        name_name = None
        tag_name = "labels"
        uid2id = load_uid2id(data_name)
    elif data_name == "movie":
        input_path = "data/movies_cleaned.txt"
        output_path = "deal_data/movieitem_info_all.json"
        id_name = "movie_id"
        name_name = "name"
        tag_name = "tag"
        uid2id = None



    data_item = pd.read_csv(input_path, sep='\t', skipinitialspace=True)
    all_infos = {

    }
    for _, row in tqdm.tqdm(data_item.iterrows()):
        item_id = row[id_name]
        item_tag = row[tag_name]
        if isinstance(item_tag, str):
            if "|" in item_tag:
                item_tag = [val for val in item_tag.split("|") if len(val) > 0]
            elif "[" in item_tag:
                item_tag = item_tag.replace("['", "").replace("']", "")
                item_tag = [val for val in item_tag.split("', '") if len(val) > 0]
            else:
                item_tag = [item_tag] if len(item_tag) > 0 else []
        else:
            item_tag = []

        item_name = ""
        if name_name is not None:
            item_name = row[name_name]

        if item_id not in all_infos:
            all_infos[item_id] = {
                "item_id": item_id,
                "tags": item_tag,
                "title": item_name,
            }
        else:
            all_infos[item_id]["tags"].extend(item_tag)


    wj_infos = []
    for k, v in all_infos.items():
        v["item_id"] = v["item_id"] if uid2id is None else uid2id[v["item_id"]]
        v["tags"] = list(set(v["tags"]))
        wj_infos.append(v)
    json.dump(wj_infos, open(output_path, mode='w', encoding="utf8"), indent=4, ensure_ascii=False)


deal_item("music")
deal_item("book")
deal_item("movie")
