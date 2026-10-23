import torch
import numpy as np
from dataclasses import dataclass


@dataclass
class ModelOutput:
    pref_logits: torch.Tensor | None = None
    success_logits: torch.Tensor | None = None
    progress_logits: torch.Tensor | None = None

    hidden_states: torch.Tensor | None = None


def convert_bins_to_continuous(bin_logits: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Decode progress logits: always softmax over bins, then take their expectation.

    A logit vector summing to one is still a logit vector. Normalized target
    distributions use convert_probabilities_to_continuous instead.
    """
    if isinstance(bin_logits, torch.Tensor):
        bin_probs = torch.softmax(bin_logits, dim=-1)
    else:
        shifted = bin_logits - np.max(bin_logits, axis=-1, keepdims=True)
        exponentials = np.exp(shifted)
        bin_probs = exponentials / exponentials.sum(axis=-1, keepdims=True)
    return convert_probabilities_to_continuous(bin_probs)


def convert_probabilities_to_continuous(bin_probs: torch.Tensor | np.ndarray) -> torch.Tensor | np.ndarray:
    """Take the bin-center expectation of an explicitly normalized distribution."""
    num_bins = bin_probs.shape[-1]
    bin_centers = (
        torch.linspace(
            0.0, 1.0, num_bins, device=bin_probs.device,
            dtype=bin_probs.dtype if bin_probs.is_floating_point() else torch.float32,
        )
        if isinstance(bin_probs, torch.Tensor)
        else np.linspace(0.0, 1.0, num_bins)
    )
    return (
        (bin_probs * bin_centers).sum(dim=-1)
        if isinstance(bin_probs, torch.Tensor)
        else (bin_probs * bin_centers).sum(axis=-1)
    )


def convert_bins_to_continuous_hard(
    bin_logits: torch.Tensor | np.ndarray,
) -> torch.Tensor | np.ndarray:
    """
    Convert discrete bins to a continuous value in [0, 1]
    by selecting the argmax bin and returning its center.
    """
    num_bins = bin_logits.shape[-1]

    if isinstance(bin_logits, torch.Tensor):
        idx = torch.argmax(bin_logits, dim=-1)
        bin_centers = torch.linspace(
            0.0,
            1.0,
            num_bins,
            device=bin_logits.device,
            dtype=bin_logits.dtype,
        )
        return bin_centers[idx]

    else:
        idx = np.argmax(bin_logits, axis=-1)
        bin_centers = np.linspace(0.0, 1.0, num_bins)
        return bin_centers[idx]


def convert_bin_index_to_continuous(bin_index: torch.Tensor | np.ndarray, num_bins: int) -> torch.Tensor | np.ndarray:
    """Convert discrete bin index to continuous progress value in [0, 1] by selecting the argmax bin and returning its center."""
    if isinstance(bin_index, torch.Tensor):
        bin_centers = torch.linspace(0.0, 1.0, num_bins, device=bin_index.device, dtype=torch.float32)
        return bin_centers[bin_index.long()]
    return np.linspace(0.0, 1.0, num_bins)[np.asarray(bin_index, dtype=int)]


def convert_discrete_target_to_continuous(
    target: torch.Tensor | np.ndarray, num_bins: int
) -> torch.Tensor | np.ndarray:
    """Convert discrete target to continuous progress value in [0, 1] by selecting the argmax bin and returning its center."""
    if len(target.shape) == 2:
        return convert_bin_index_to_continuous(target, num_bins)
    else:
        return convert_probabilities_to_continuous(target)
