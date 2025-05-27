# Enviroment
- conda create -n env_name python==3.11
- conda activate env_name
- pip install -r requirements.txt

# Dataset
- Download the [Douban](https://www.researchgate.net/publication/350793434_Douban_dataset_ratings_item_details_user_profiles_tags_and_reviews) dataset, extract it and store it in ./dataset folder

# Pre-trained Model
- Download [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct), store it in ./plms/Qwen2.5-7B-Instruct folder
- Download [bge-m3](https://huggingface.co/BAAI/bge-m3), store it in ./plms/bge-m3 folder folder

# Train & Eval
- execute run_main.sh