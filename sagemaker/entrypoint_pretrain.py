import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path


def _env(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def _pick_data_path(train_dir: str, data_file: str | None) -> str:
    train_dir_p = Path(train_dir)
    if data_file:
        candidate = train_dir_p / data_file
        if candidate.exists():
            return str(candidate)
        raise FileNotFoundError(f"--data_file={data_file!r} not found under {train_dir!r}")

    # Default convention: put your jsonl at channel root with this name.
    default = train_dir_p / "pretrain.jsonl"
    if default.exists():
        return str(default)

    # Fallback: pick the first jsonl we can find.
    matches = sorted(glob.glob(str(train_dir_p / "*.jsonl")))
    if matches:
        return matches[0]
    raise FileNotFoundError(
        f"No .jsonl found under {train_dir!r}. "
        f"Expected pretrain.jsonl or pass --data_file <name>.jsonl"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="SageMaker entrypoint for MiniMind pretrain (single-node multi-GPU)")

    # SageMaker convention: /opt/ml/input/data/<channel_name>
    parser.add_argument("--train_channel", type=str, default=_env("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train"))
    parser.add_argument("--data_file", type=str, default=None, help="JSONL filename inside the train channel (default: pretrain.jsonl)")

    # Where SageMaker collects model artifacts (uploaded at end).
    parser.add_argument("--model_dir", type=str, default=_env("SM_MODEL_DIR", "/opt/ml/model"))
    # Where you can write intermediate artifacts/checkpoints.
    parser.add_argument("--output_data_dir", type=str, default=_env("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))

    # Your trainer args (a small subset here; everything else can be passed through)
    parser.add_argument("--save_weight", type=str, default="pretrain")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--max_seq_len", type=int, default=340)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_hidden_layers", type=int, default=8)
    parser.add_argument("--use_moe", type=int, default=0, choices=[0, 1])
    parser.add_argument("--accumulation_steps", type=int, default=4)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--from_weight", type=str, default="none")
    parser.add_argument("--from_resume", type=int, default=0, choices=[0, 1])
    parser.add_argument("--use_compile", type=int, default=0, choices=[0, 1])
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MiniMind-Pretrain")

    # Any extra args are forwarded to trainer/train_pretrain.py verbatim.
    args, extra = parser.parse_known_args()

    train_dir = args.train_channel
    data_path = _pick_data_path(train_dir, args.data_file)

    model_dir = args.model_dir
    output_data_dir = args.output_data_dir
    ckpt_dir = str(Path(output_data_dir) / "checkpoints")
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # SageMaker provides SM_NUM_GPUS for GPU instances.
    n_gpus = int(_env("SM_NUM_GPUS", "0") or "0")
    if n_gpus <= 0:
        # Fallback: try to infer via torch if available.
        try:
            import torch

            n_gpus = torch.cuda.device_count()
        except Exception:
            n_gpus = 0

    # Torchrun sets RANK/LOCAL_RANK/WORLD_SIZE which train_pretrain.py expects.
    base_cmd = [
        sys.executable,
        "-u",
        os.path.join("trainer", "train_pretrain.py"),
        "--data_path",
        data_path,
        "--save_dir",
        model_dir,
        "--ckpt_dir",
        ckpt_dir,
        "--save_weight",
        args.save_weight,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--learning_rate",
        str(args.learning_rate),
        "--dtype",
        args.dtype,
        "--max_seq_len",
        str(args.max_seq_len),
        "--hidden_size",
        str(args.hidden_size),
        "--num_hidden_layers",
        str(args.num_hidden_layers),
        "--use_moe",
        str(args.use_moe),
        "--accumulation_steps",
        str(args.accumulation_steps),
        "--grad_clip",
        str(args.grad_clip),
        "--log_interval",
        str(args.log_interval),
        "--save_interval",
        str(args.save_interval),
        "--num_workers",
        str(args.num_workers),
        "--from_weight",
        args.from_weight,
        "--from_resume",
        str(args.from_resume),
        "--use_compile",
        str(args.use_compile),
        "--wandb_project",
        args.wandb_project,
    ]
    if args.use_wandb:
        base_cmd.append("--use_wandb")
    base_cmd.extend(extra)

    if n_gpus >= 2:
        cmd = [
            "torchrun",
            "--standalone",
            "--nproc_per_node",
            str(n_gpus),
            *base_cmd,
        ]
    else:
        cmd = base_cmd

    # Helpful defaults for NCCL (safe for single-node).
    env = os.environ.copy()
    env.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")
    env.setdefault("NCCL_DEBUG", env.get("NCCL_DEBUG", "WARN"))

    print("Launching command:\n" + " ".join(cmd))
    print("SageMaker env summary:\n" + json.dumps(
        {
            "SM_NUM_GPUS": _env("SM_NUM_GPUS", ""),
            "SM_CHANNEL_TRAIN": _env("SM_CHANNEL_TRAIN", ""),
            "SM_MODEL_DIR": _env("SM_MODEL_DIR", ""),
            "SM_OUTPUT_DATA_DIR": _env("SM_OUTPUT_DATA_DIR", ""),
        },
        indent=2,
    ))
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    raise SystemExit(main())

