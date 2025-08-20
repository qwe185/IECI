import json
from .processor_multiarg import MultiargProcessor


_DATASET_DIR = {
    
    "esl":{
        "train_file": './data/ESL/data_esl/train.jsonl',
        "dev_file": './data/ESL/data_esl/dev.jsonl',
        "test_file": './data/ESL/data_esl/test.jsonl',
        "max_span_num_file": "./data/event_typeinfo/role_num_esl.json",
    },
    
}


def build_processor(args, tokenizer):
    if args.dataset_type not in _DATASET_DIR: 
        raise NotImplementedError("Please use valid dataset name")
    args.train_file=_DATASET_DIR[args.dataset_type]['train_file']
    args.dev_file = _DATASET_DIR[args.dataset_type]['dev_file']
    args.test_file = _DATASET_DIR[args.dataset_type]['test_file']

    args.role_name_mapping = None

    processor = MultiargProcessor(args, tokenizer)
    return processor

