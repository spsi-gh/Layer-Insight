import torch
import timm


def load_model(args):

    if args.timm_model is None:
        raise ValueError(
            "--timm-model must be specified"
        )

    model = timm.create_model(
        args.timm_model,
        pretrained=True
    )

    return model


def generate_inputs(
    args,
    device,
    dtype
):
    return (
        torch.randn(
            args.batch_size,
            3,
            224,
            224,
            device=device,
            dtype=dtype
        ),
    )

def get_runtime_metadata(args):
    return {
        "batch_size": args.batch_size,
        "image_size": 224,
        "channels": 3
    }