from typing import TYPE_CHECKING, List, Optional

from .data_deal_douban import SpecialDataCollator
from .model_deal import load_model_special
from ...extras.constants import IGNORE_INDEX
from transformers import Seq2SeqTrainer
from .data_deal_douban import make_supervised_data_module
from transformers import AutoTokenizer

if TYPE_CHECKING:
    from transformers import TrainerCallback
    from ...hparams import DataArguments, FinetuningArguments, GeneratingArguments, ModelArguments, Seq2SeqTrainingArguments




def run_sft(
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

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset_module["train_dataset"],
    )
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
