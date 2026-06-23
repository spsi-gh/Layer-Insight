import torch
from transformers import AutoModel


def load_model(args):
    return AutoModel.from_pretrained(
        args.hf_model
    )


def generate_inputs(args, device, dtype):

    return {
        "input_ids": torch.randint(
            0,
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