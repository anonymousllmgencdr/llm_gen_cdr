

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from transformers import AutoConfig
from ...extras.logging import get_logger
from ...extras.misc import count_parameters, skip_check_imports, try_download_model_from_ms
from ...model.model_utils.misc import register_autoclass
from ...model.patcher import patch_config, patch_model
from ...model.adapter import init_adapter
from transformers import AutoModelForCausalLM

if TYPE_CHECKING:
    from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer

    from ...hparams import FinetuningArguments, ModelArguments


logger = get_logger(__name__)

def load_model_special(
    tokenizer: "PreTrainedTokenizer",
    model_args: "ModelArguments",
    finetuning_args: "FinetuningArguments",
    # train_args,
    is_trainable: bool = False,
    add_valuehead: bool = False,
) -> "PreTrainedModel":
    r"""
    Loads pretrained model.
    """
    init_kwargs = _get_init_kwargs(model_args)
    config = load_config(model_args)
    patch_config(config, tokenizer, model_args, init_kwargs, is_trainable)
    lazy_load = False
    if model_args.use_unsloth:
        if model_args.adapter_name_or_path is not None:
            lazy_load = True
    
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config).to(torch.bfloat16)
    print("tokenizer: {}. model_embedding: {}".format(len(tokenizer), model.model.embed_tokens.weight.size()[0]))
    if len(tokenizer) > model.model.embed_tokens.weight.size()[0]:
        model.resize_token_embeddings(len(tokenizer))
        print("修改后 tokenizer: {}. model_embedding: {}".format(len(tokenizer), model.model.embed_tokens.weight.size()[0]))
    
    

    if not lazy_load:
        patch_model(model, tokenizer, model_args, is_trainable, add_valuehead)
        register_autoclass(config, model, tokenizer)

    model = init_adapter(config, model, model_args, finetuning_args, is_trainable)


    if not is_trainable:
        model.requires_grad_(False)
        for param in model.parameters():
            if param.data.dtype == torch.float32 and model_args.compute_dtype != torch.float32:
                param.data = param.data.to(model_args.compute_dtype)

        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    if is_trainable:
        param_stats = "trainable params: {:,} || all params: {:,} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    else:
        param_stats = "all params: {:,}".format(all_param)
    
    logger.info(param_stats)

    if model_args.print_param_status:
        for name, param in model.named_parameters():
            print(
                "name: {}, dtype: {}, device: {}, trainable: {}".format(
                    name, param.dtype, param.device, param.requires_grad
                )
            )
        model.print_trainable_parameters()
        exit(0)

    return model




def _get_init_kwargs(model_args: "ModelArguments") -> Dict[str, Any]:
    r"""
    Gets arguments to load config/tokenizer/model.

    Note: including inplace operation of model_args.
    """
    skip_check_imports()
    model_args.model_name_or_path = try_download_model_from_ms(model_args)
    return {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }

def load_config(model_args: "ModelArguments") -> "PretrainedConfig":
    r"""
    Loads model config.
    """
    init_kwargs = _get_init_kwargs(model_args)
    return AutoConfig.from_pretrained(model_args.model_name_or_path, **init_kwargs)



