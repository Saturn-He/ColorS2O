"""Standalone FLOPs + parameter calculator for JiT inference backbone.

Example:
    python FLOPs.py --model JiT-B/8 --img_size 512 --batch_size 1 --use_hint_input
"""

import argparse
from typing import Optional

import torch

from model_jit_mask_guided_embed import JiT_models


def _count_flops_torch_mode(
    model: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    hint_input: Optional[torch.Tensor],
) -> Optional[float]:
    """Count FLOPs with torch's built-in flop counter when available."""
    try:
        from torch.utils.flop_counter import FlopCounterMode
    except Exception:
        return None

    with torch.no_grad():
        with FlopCounterMode(display=False) as counter:
            _ = model(x, t, y, hint_input)
    return float(counter.get_total_flops())


def _count_flops_fvcore(
    model: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    hint_input: Optional[torch.Tensor],
) -> Optional[float]:
    """Count FLOPs via fvcore fallback."""
    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception:
        return None

    with torch.no_grad():
        analysis = FlopCountAnalysis(model, (x, t, y, hint_input))
        return float(analysis.total())


def _count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return total parameters and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def _format_flops(value: float) -> str:
    units = [(1e12, "TFLOPs"), (1e9, "GFLOPs"), (1e6, "MFLOPs")]
    for scale, name in units:
        if value >= scale:
            return f"{value / scale:.4f} {name}"
    return f"{value:.2f} FLOPs"


def _format_count(value: int) -> str:
    units = [(1_000_000_000, "B"), (1_000_000, "M"), (1_000, "K")]
    for scale, suffix in units:
        if value >= scale:
            return f"{value / scale:.4f}{suffix}"
    return str(value)


def _estimate_generate_net_calls(method: str, steps: int) -> int:
    if method == "euler":
        # _forward_sample is called once per step, and each _forward_sample does
        # conditional + unconditional net forward => 2 net forwards.
        return 2 * steps
    if method == "heun":
        # for loop: (steps - 1) heun steps, each does 2x _forward_sample,
        # last step: 1x euler => total _forward_sample = 2*(steps-1)+1
        # net forwards = 2 * _forward_sample
        return 2 * (2 * steps - 1)
    raise ValueError(f"Unsupported sampling method: {method}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute JiT FLOPs for inference.")
    parser.add_argument("--model", default="JiT-B/8", choices=sorted(JiT_models.keys()))
    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--class_num", type=int, default=1)
    parser.add_argument("--attn_dropout", type=float, default=0.0)
    parser.add_argument("--proj_dropout", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0", help="cpu / cuda / cuda:0")
    parser.add_argument("--use_hint_input", action="store_true")

    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--sampling_method", default="heun", choices=["euler", "heun"])
    parser.add_argument(
        "--estimate_generate_flops",
        action="store_true",
        help="Also print estimated FLOPs for full denoiser.generate() based on method/steps.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    model = JiT_models[args.model](
        input_size=args.img_size,
        in_channels=4,
        out_channels=3,
        num_classes=args.class_num,
        attn_drop=args.attn_dropout,
        proj_drop=args.proj_dropout,
    ).to(device)
    model.eval()

    bsz = args.batch_size
    x = torch.randn(bsz, 4, args.img_size, args.img_size, device=device)
    t = torch.rand(bsz, device=device)
    y = torch.zeros(bsz, dtype=torch.long, device=device)

    hint_input = None
    if args.use_hint_input:
        hint_input = torch.zeros(bsz, 4, args.img_size, args.img_size, device=device)

    flops = _count_flops_torch_mode(model, x, t, y, hint_input)
    backend = "torch.utils.flop_counter"
    if flops is None:
        flops = _count_flops_fvcore(model, x, t, y, hint_input)
        backend = "fvcore"

    if flops is None:
        raise RuntimeError(
            "Unable to count FLOPs: neither torch flop counter nor fvcore is available. "
            "Please use PyTorch>=2.1 or install fvcore."
        )

    total_params, trainable_params = _count_parameters(model)

    print(f"FLOPs backend: {backend}")
    print(f"Model: {args.model}, img_size={args.img_size}, batch_size={args.batch_size}")
    print(f"Single net forward FLOPs: {flops:.0f} ({_format_flops(flops)})")
    print(
        f"Model parameters: total={total_params} ({_format_count(total_params)}), "
        f"trainable={trainable_params} ({_format_count(trainable_params)})"
    )

    if args.estimate_generate_flops:
        net_calls = _estimate_generate_net_calls(args.sampling_method, args.num_sampling_steps)
        est_generate_flops = flops * net_calls
        print(
            f"Estimated generate() net-forward FLOPs: {est_generate_flops:.0f} "
            f"({_format_flops(est_generate_flops)})"
        )
        print(f"  - method={args.sampling_method}, steps={args.num_sampling_steps}, net_forward_calls={net_calls}")


if __name__ == "__main__":
    main()