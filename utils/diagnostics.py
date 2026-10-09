"""Model summary, parameter count, approximate FLOPs, gradients, activations, NaN detection."""
from contextlib import contextmanager
import torch
from torch import nn


def assert_finite(value, context):
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"Non-finite tensor at {context}, shape={tuple(value.shape)}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            assert_finite(item, f"{context}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            assert_finite(item, f"{context}.{key}")


def gradient_summary(model):
    squared_norms = []
    missing = []
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            missing.append(name)
        else:
            assert_finite(parameter.grad, f"gradient.{name}")
            squared_norms.append(parameter.grad.detach().float().square().sum())
    norm = torch.stack(squared_norms).sum().sqrt().item() if squared_norms else 0.0
    return {"global_norm": norm, "missing_parameters": missing}


def parameter_summary(model):
    inactive = model.expected_inactive_parameters()
    return {"total": sum(parameter.numel() for parameter in model.parameters()),
            "requires_grad": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "official_inactive": sum(parameter.numel() for name, parameter in model.named_parameters() if name in inactive),
            "parameter_tensors": len(list(model.parameters()))}


@contextmanager
def activation_statistics(model):
    statistics = {}
    def hook(name):
        def capture(module, inputs, output):
            assert_finite(output, f"activation.{name}")
            tensors = output if isinstance(output, (tuple, list)) else (output,)
            statistics[name] = [{"shape": list(tensor.shape), "mean": tensor.detach().mean().item(),
                                "std": tensor.detach().std(unbiased=False).item(),
                                "minimum": tensor.detach().min().item(),
                                "maximum": tensor.detach().max().item()}
                               for tensor in tensors if isinstance(tensor, torch.Tensor)]
        return capture
    monitored = {"feature_extractor", "SC", "fc_t", "fc_s", "fc_l"}
    handles = [module.register_forward_hook(hook(name)) for name, module in model.named_modules()
               if name in monitored]
    try:
        yield statistics
    finally:
        for handle in handles:
            handle.remove()


def estimate_flops(model, images, way, shot):
    """Measured Conv/Linear multiply-add count (2 FLOPs/MAC), plus cosine einsum.

    BN/ELU/interpolation/pooling/norm/mean costs omitted; this is explicitly an estimate.
    Eval forward increments official DropBlock counters; restore counters afterward.
    """
    counts = {"conv_linear_flops": 0}
    def count(module, inputs, output):
        if isinstance(module, nn.Conv2d):
            multiplier = (module.in_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1]
        else:
            multiplier = module.in_features
        counts["conv_linear_flops"] += 2*output.numel()*multiplier
    counters = {name: module.num_batches_tracked for name, module in model.named_modules()
                if hasattr(module, "ramp_batches")}
    was_training = model.training
    handles = [module.register_forward_hook(count) for module in model.modules()
               if isinstance(module, (nn.Conv2d, nn.Linear))]
    try:
        model.eval()
        with torch.no_grad():
            model(images, way=way, shot=shot)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
        for name, module in model.named_modules():
            if name in counters:
                module.num_batches_tracked = counters[name]
    batch = images.shape[0]
    query_count = batch-way*shot
    counts["cosine_flops"] = 2*model.config.feature_dim*(3*query_count*way + batch*model.config.base_classes)
    counts["total_estimated_flops"] = counts["conv_linear_flops"]+counts["cosine_flops"]
    counts["scope"] = "Conv/Linear MACs*2 + cosine inner products; excludes non-MAC operations"
    return counts
