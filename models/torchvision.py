import torch
import torchvision.models as tv_models


def load_model(args):

    if args.tv_model is None:
        raise ValueError(
            "--tv-model must be specified"
        )

    if not hasattr(tv_models, args.tv_model):
        raise ValueError(
            f"Unknown torchvision model: {args.tv_model}"
        )

    model_fn = getattr(
        tv_models,
        args.tv_model
    )

    try:
        model = model_fn(weights="DEFAULT")
    except TypeError:
        model = model_fn(pretrained=True)

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