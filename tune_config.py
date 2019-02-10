import argparse
import random
from copy import deepcopy
from typing import Any, Dict

from ray import tune


TUNE_REQUIRED_CONFIG = {"num_samples": 1}

# The duplication of tune.sample_from(...) is intentional. This makes it easier
# to change sampling strategy in the future for certain parameters.
TUNE_DEFAULT_CONFIG = {
    "discount": tune.sample_from(
        lambda _: random.choice([0.9, 0.99, 0.995, 0.999, 0.9995, 0.9999])
    ),
    "epsilon": tune.sample_from(lambda _: random.choice([0.01, 0.05, 0.08, 0.1])),
    "lr": tune.sample_from(lambda _: random.choice([0.01, 0.05, 0.1, 0.5, 1.0])),
    "batch_size": tune.sample_from(
        lambda _: random.choice([32, 64, 128, 256, 512, 1024, 2048])
    ),
    "clipping": tune.sample_from(lambda _: random.choice([0.1, 0.2, 0.5])),
    "entropy_bonus": tune.sample_from(lambda _: random.choice([0.0, 0.01, 0.05])),
}


def tune_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Helper function to set the `config` for Ray Tune. This has to be passed
    separately from argparse to conform to Tune's API.

    Usage:

    Say the command line is `python3 main.py -t lr -t discount`. Then this will
    extract the `lr` and `discount` keys from `TUNE_DEFAULT_CONFIG`.
    """
    config = deepcopy(TUNE_REQUIRED_CONFIG)

    if args.tune is not None:
        for tunable_param in args.tune:
            config[tunable_param] = TUNE_DEFAULT_CONFIG[tunable_param]
    return config
