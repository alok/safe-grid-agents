import argparse
import random
from typing import Any, Callable, Dict, Sequence, Union

from ray import tune


def config_from_argparse(
    args: argparse.Namespace, argparse_attr: str, tune_opts: Sequence[float]
) -> Union[float, Callable[[Any], float]]:
    """Helper function to decide whether to use argparse or Ray Tune for a
    hyperparameter.

    Calling `config_from_argparse('lr', ...) would look for `args.lr`,
    and if it exists and is not None, uses it. Otherwise it uses Ray
    Tune.

    This was written to support a specific configuration style and should be
    removed if that style is no longer useful.
    """
    return (
        tune.sample_from(lambda _: random.choice(tune_opts))
        if not hasattr(args, argparse_attr) or getattr(args, argparse_attr) is None
        else getattr(args, argparse_attr)
    )


def tune_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "discount": config_from_argparse(
            args,
            argparse_attr="discount",
            tune_opts=[0.9, 0.99, 0.995, 0.999, 0.9995, 0.9999],
        ),
        "epsilon": config_from_argparse(
            args, argparse_attr="epsilon", tune_opts=[0.01, 0.05, 0.08, 0.1]
        ),
        "lr": config_from_argparse(
            args, argparse_attr="lr", tune_opts=[0.01, 0.05, 0.1, 0.5, 1.0]
        ),
        "batch_size": config_from_argparse(
            args,
            argparse_attr="batch_size",
            tune_opts=[32, 64, 128, 256, 512, 1024, 2048],
        ),
        "clipping": config_from_argparse(
            args, argparse_attr="clipping", tune_opts=[0.1, 0.2, 0.5]
        ),
        "entropy_bonus": config_from_argparse(
            args, argparse_attr="entropy_bonus", tune_opts=[0.0, 0.01, 0.05]
        ),
        "num_samples": 1,
    }
