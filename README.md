# Enviroment
- Configuring the [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) environment
- Configuring the [vllm](https://github.com/vllm-project/vllm) environment
- Configuring the [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding/tree/master) environment
- Configuring the [faiss](https://github.com/facebookresearch/faiss) environment

# Dataset
- Download the [Douban](https://www.researchgate.net/publication/350793434_Douban_dataset_ratings_item_details_user_profiles_tags_and_reviews) dataset, extract it and store it in ./data folder

# Pre-trained Model
- [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [bge-m3](https://huggingface.co/BAAI/bge-m3)

# Train
- execute run.sh
- modify the training script sft_douban.yaml
- train the model using LLaMA-Factory

# Eval
modify and run gen_eval.py