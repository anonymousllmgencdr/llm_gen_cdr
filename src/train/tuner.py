import sys
print(sys.path)
import os
import shutil
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from ..extras.logging import get_logger
from ..hparams import get_train_args
from ..train.base_sft import run_baseline
from ..train.de_sft import run_sft
from ..train.de_dpo import run_dpo



from ..train.callbacks import LogCallback

logger = get_logger(__name__)


def run_exp(args: Optional[Dict[str, Any]] = None, callbacks: List["TrainerCallback"] = []) -> None:
    callbacks.append(LogCallback())
    model_args, data_args, training_args, finetuning_args, generating_args = get_train_args(args)

    if finetuning_args.stage == "sft":
        run_sft(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)
    elif finetuning_args.stage == "dpo":
        run_dpo(model_args, data_args, training_args, finetuning_args, generating_args, callbacks)


# if __name__ == '__main__':
#     run_exp()