
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Literal, Optional, Union
from transformers import Seq2SeqTrainingArguments


@dataclass
class WithCallTrainingArguments(Seq2SeqTrainingArguments):

    data_shuffle: bool = field(
        default=False, 
        metadata={"help": "是否需要打乱训练数据"}
    )
    
    stage_count: int = field(
        default=0,
        metadata={"help": "第几阶段训练"},
    )