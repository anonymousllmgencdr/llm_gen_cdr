import pandas as pd
import json, os
import numpy as np

# 0 for no filiter, 1 for filter label, 2 for filter limit
def deal(DATATYPE=0):
    GENERARE_TYPE = 3
    GENERATELIST = ["", "_label", "_limit", "_rating"]
    DATALIST = ["book", "movie", "music"]
    base_path = "./dataset/"
    base_out_path = "./dataset/"
    if not os.path.exists(base_out_path):
        os.makedirs(base_out_path)

    file_name = base_path + DATALIST[DATATYPE] + "reviews_cleaned.txt"
    output_name = base_out_path + DATALIST[DATATYPE] + "reviews_cleaned" + GENERATELIST[GENERARE_TYPE] + ".json"
    item_column = DATALIST[DATATYPE] + "_id"
    if DATATYPE == 2:
        file_name_item = base_path + DATALIST[DATATYPE] + "_cleaned.txt"
    else:
        file_name_item = base_path + DATALIST[DATATYPE] + "s_cleaned.txt"
    file_name_user = base_path + "users_cleaned.txt"
    user_column = "user_id"
    label_column = "labels"

    data_item = pd.read_csv(file_name_item, sep='\t', skipinitialspace=True)
    data_user = pd.read_csv(file_name_user, sep='\t', skipinitialspace=True)
    item_uid_to_id = dict(zip(data_item['UID'], data_item[item_column]))
    user_uid_to_id = dict(zip(data_user['UID'], data_user[user_column]))

    data = pd.read_csv(file_name, sep='\t', skipinitialspace=True)
    data['timestamp'] = pd.to_datetime(data['time']).astype(int) / 10 ** 9  # 转换为秒级时间戳

    filtered_data = data

    if GENERARE_TYPE == 1:
        if DATATYPE != 1:
            filtered_data = data[data[label_column].notna()]
    elif GENERARE_TYPE == 2:
        while True:

            item_user_count = data.groupby(item_column)['user_id'].nunique()
            valid_items = item_user_count[item_user_count >= 5].index
            filtered_data = data[data[item_column].isin(valid_items)]

            user_item_count = filtered_data.groupby('user_id')[item_column].nunique()
            valid_users = user_item_count[user_item_count >= 5].index
            filtered_data = filtered_data[filtered_data['user_id'].isin(valid_users)]

            if len(filtered_data) == len(data):
                break

            data = filtered_data
    elif GENERARE_TYPE == 3:
        filtered_data = filtered_data[filtered_data["rating"] >= 4]


    movie_id_to_tag = []
    if DATATYPE == 1 and GENERARE_TYPE == 1:
        movie_id_to_tag = dict(zip(data['movie_id'], data['tag']))

    result = {}
    for _, row in filtered_data.iterrows():
        user_id = user_uid_to_id[row['user_id']]
        item_id = item_uid_to_id[row[item_column]]
        rating = row['rating']
        timestamp = row['timestamp']

        writabale = (DATATYPE == 1 and GENERARE_TYPE == 1 and item_id in movie_id_to_tag and pd.notna(
            movie_id_to_tag[item_id])) or (DATATYPE == 1 and GENERARE_TYPE != 1) or (DATATYPE != 1)
        if writabale:
            if user_id not in result:
                result[user_id] = {"user_id": user_id, "item_id": [], "rating": [], "timestamp": []}

            result[user_id]["item_id"].append(item_id)
            result[user_id]["rating"].append(rating)
            result[user_id]["timestamp"].append(timestamp)

    output_data = [value for value in result.values()]
    with open(output_name, 'w', encoding='utf-8') as json_file:
        json.dump(output_data, json_file, ensure_ascii=False, indent=4)

    with open(output_name, 'r', encoding='utf-8') as f:
        data = json.load(f)

    user_count = len(data)
    item_counts = {user['user_id']: len(user['item_id']) for user in data}
    item_count_series = pd.Series(item_counts)

    max_count = item_count_series.max()
    min_count = item_count_series.min()
    mean_count = item_count_series.mean()

    sorted_item_counts = item_count_series.sort_values(ascending=False)
    percentiles = np.percentile(item_count_series, np.arange(0, 101, 10))

    all_item_ids = []
    for user in data:
        all_item_ids.extend(user['item_id'])
    unique_item_count = len(set(all_item_ids))

    print("######## {} #########".format(DATALIST[DATATYPE]))
    print(f"user num: {user_count}")
    print(f"item num: {unique_item_count}")
    print(f"inter num: {len(filtered_data)}")
    print(f"max inter num: {max_count}")
    print(f"min inter num: {min_count}")
    print(f"mean inter num: {mean_count:.2f}")

deal(0)
deal(1)
deal(2)



