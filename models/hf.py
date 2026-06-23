import torch
from transformers import AutoModel, AutoTokenizer


def load_model(args):
    return AutoModel.from_pretrained(
        args.hf_model
    )

def generate_inputs(args, device, dtype):

    # tokenizer = AutoTokenizer.from_pretrained(args.hf_model)
    return {
        "input_ids": torch.randint(
            0,
            # tokenizer.vocab_size,
            1000,
            (args.batch_size, args.seq_len),
            device=device,
            dtype=torch.long
        ),
        "attention_mask": torch.ones(
            args.batch_size,
            args.seq_len,
            device=device,
            dtype=torch.long
        )
    }


def get_runtime_metadata(args):
    return {
        "batch_size": args.batch_size,
        "seq_len": args.seq_len
    }