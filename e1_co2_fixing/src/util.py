from pathlib import Path
import torch


def sigm(t: torch.Tensor, k: float, n: int) -> torch.Tensor:
    """$t^n / (t^n + k^n)$"""
    return t**n / (t**n + k**n)


def rev_sigm(t: torch.Tensor, k: float, n: int) -> torch.Tensor:
    """$k^n / (t^n + k^n)$"""
    return k**n / (t**n + k**n)


class Config:
    """Config container"""

    def __init__(
        self,
        device: str,
        runs_dir: Path | str,
        max_steps: int,
        max_time_m: int,
        n_trials: int,
    ):
        self.device = device
        self.runs_dir = runs_dir if isinstance(runs_dir, Path) else Path(runs_dir)
        self.max_steps = max_steps
        self.max_time_m = max_time_m
        self.n_trials = n_trials

    @classmethod
    def pop_from(cls, kwargs: dict) -> "Config":
        """Pop config keys from kwargs and return Config"""
        return cls(
            device=kwargs.pop("device"),
            runs_dir=kwargs.pop("runs_dir"),
            max_steps=kwargs.pop("max_steps"),
            max_time_m=kwargs.pop("max_time_m"),
            n_trials=kwargs.pop("n_trials"),
        )
