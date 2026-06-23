import argparse
import importlib
import os
import json
import platform
from datetime import datetime
import torch

def parse_arguments():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--model",
        required=True,
        help="Python module path. Example: models.mlp"
    )

    p.add_argument(
        "--batch-size",
        type=int,
        default=8
    )

    p.add_argument(
        "--seq-len",
        type=int,
        default=128
    )

    p.add_argument(
        "--dtype",
        choices=["fp32", "bf16"],
        default="bf16"
    )

    p.add_argument(
        "--compile",
        action="store_true"
    )

    p.add_argument(
        "--warmup",
        action="store_true"
    )

    p.add_argument(
        "--trace-dir",
        default="./traces"
    )

    p.add_argument(
    "--hf-model",
    default=None
    )

    p.add_argument(
    "--tv-model",
    default=None
    )

    p.add_argument(
        "--timm-model",
        default=None
    )

    return p.parse_args()


def load_model_module(module_name):
    return importlib.import_module(module_name)


def main():
    args = parse_arguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    module = load_model_module(args.model)

    model = module.load_model(args).to(device=device, dtype=dtype)

    total_params = sum(
    p.numel()
    for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    if args.compile:
        model = torch.compile(model)

    model.eval()

    print(f"Using device: {device}")

    inputs = module.generate_inputs(
        args,
        device,
        dtype
    )
    
    def step():
        with torch.profiler.record_function("forward_pass"):
            with torch.no_grad():
                if isinstance(inputs, dict):
                    return model(**inputs)
                return model(*inputs)

    if args.warmup:
        for _ in range(3):
            step()

        if device.type == "cuda":
            torch.cuda.synchronize()

    os.makedirs(args.trace_dir, exist_ok=True)

    run_id = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
    )

    run_dir = os.path.join(
        args.trace_dir,
        run_id
    )

    os.makedirs(run_dir, exist_ok=True)

    model_name = (
    args.hf_model
    or args.tv_model
    or args.timm_model
    or args.model.split(".")[-1]
    )

    trace_path = os.path.join(
        run_dir,
        "trace.json"
    )

    table_path = os.path.join(
        run_dir,
        "summary.txt"
    )

    metadata_path = os.path.join(
        run_dir,
        "metadata.json"
    )

    schedule = torch.profiler.schedule(
        wait=1,
        warmup=1,
        active=3,
        repeat=1
    )

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    with torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:

        for _ in range(5):
            step()
            prof.step()

    if device.type == "cuda":
        torch.cuda.synchronize()

    print(f"Saving trace: {trace_path}")

    prof.export_chrome_trace(trace_path)

    table = prof.key_averages().table(
        sort_by="cuda_time_total",
        row_limit=20
    )

    with open(table_path, "w") as f:
        f.write(table)

    print(f"Saving table: {table_path}")

    metadata = {
    "model": {
        "adapter": args.model,
        "name": model_name,
        "parameters": total_params,
        "trainable_parameters": trainable_params
        },

    "runtime": (
        module.get_runtime_metadata(args)
        ),

    "hardware": {
        "platform": platform.system(),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda
        }
    }

    if torch.cuda.is_available():
        metadata["hardware"]["gpu"] = (
            torch.cuda.get_device_name(0)
        )

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Saving metadata: {metadata_path}")


if __name__ == "__main__":
    main()