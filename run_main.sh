python ./dataset/deal_inter.py
python ./dataset/deal_item.py
python ./dataset/deal_data.py
python -u main.py train_config/sft_douban.yaml
python -u eval/gen_eval.py
