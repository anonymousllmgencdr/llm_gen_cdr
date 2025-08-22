from typing import TYPE_CHECKING, List, Optional

from .data_deal_douban import SpecialDataCollator
from .model_deal import load_model_special
from ...extras.constants import IGNORE_INDEX
from .train_utils import CustomDPOTrainer
from .data_deal_douban import make_supervised_data_module
from transformers import AutoTokenizer
from ...hparams import FinetuningArguments, ModelArguments

if TYPE_CHECKING:
    from transformers import TrainerCallback
    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments, Seq2SeqTrainingArguments


def create_ref_model(
    model_args, finetuning_args, add_valuehead: bool = False
):
    
    if finetuning_args.ref_model is not None:
        ref_model_args = ModelArguments.copyfrom(
            model_args,
            model_name_or_path=finetuning_args.ref_model,
            adapter_name_or_path=finetuning_args.ref_model_adapters,
            quantization_bit=finetuning_args.ref_model_quantization_bit,
        )
        ref_finetuning_args = FinetuningArguments()
        tokenizer = load_tokenizer(ref_model_args)["tokenizer"]
        # ref_model = load_model(
        #     tokenizer, ref_model_args, ref_finetuning_args, is_trainable=False, add_valuehead=add_valuehead
        # )
        ref_model = load_model_special(
            tokenizer, ref_model_args, ref_finetuning_args, is_trainable=False, add_valuehead=add_valuehead
        )
        logger.info("Created reference model from {}".format(finetuning_args.ref_model))
    else:
        if finetuning_args.finetuning_type == "lora":
            ref_model = None
        else:
            ref_model_args = ModelArguments.copyfrom(model_args)
            ref_finetuning_args = FinetuningArguments()
            # tokenizer = load_tokenizer(ref_model_args)["tokenizer"]
            tokenizer = AutoTokenizer.from_pretrained(
                ref_model_args.model_name_or_path
            )
            # ref_model = load_model(
            #     tokenizer, ref_model_args, ref_finetuning_args, is_trainable=False, add_valuehead=add_valuehead
            # )
            ref_model = load_model_special(
                tokenizer, ref_model_args, ref_finetuning_args, is_trainable=False, add_valuehead=add_valuehead
            )
            # logger.info("Created reference model from the model itself.")

    return ref_model



def run_dpo(
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    finetuning_args: "FinetuningArguments",
    generating_args: "GeneratingArguments",
    callbacks: Optional[List["TrainerCallback"]] = None,
):

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path
    )

    model = load_model_special(tokenizer, model_args, finetuning_args, training_args.do_train)
    

    dataset_module = make_supervised_data_module(
        tokenizer=tokenizer, data_args=data_args, max_query_len=data_args.cutoff_len
    )


    data_collator = SpecialDataCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if training_args.do_train else None,  # for shift short attention
        label_pad_token_id=IGNORE_INDEX if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )


    # Create reference model
    if finetuning_args.use_ref_model:
        if finetuning_args.ref_model is None and (not training_args.do_train):  # use the model itself
            ref_model = model
        else:
            ref_model = create_ref_model(model_args, finetuning_args)
    else:
        ref_model = None

    # Update arguments
    training_args.remove_unused_columns = False  # important for multimodal and pairwise dataset

    # Initialize our Trainer
    trainer = CustomDPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        finetuning_args=finetuning_args,
        data_collator=data_collator,
        callbacks=callbacks,
        tokenizer=tokenizer,
        processor=None,
        **dataset_module,
        # **tokenizer_module,
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
