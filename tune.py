import random
import ray
import ray.tune as tune

from main import args, train
from typing import Any, Callable, Optional, Sequence, Union


def config_from_argparse(

    argparse_attr: str, tune_opts: Sequence[float]
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

TUNE_CONFIG = {
    "discount": config_from_argparse(
        argparse_attr="discount",
        tune_opts=[0.9, 0.99, 0.995, 0.999, 0.9995, 0.9999],
    ),
    "epsilon": config_from_argparse(
        argparse_attr="epsilon", tune_opts=[0.01, 0.05, 0.08, 0.1]
    ),
    "epsilon_anneal": config_from_argparse(
        argparse_attr="epsilon_anneal", tune_opts=[900000]
    ),
    "lr": config_from_argparse(
        argparse_attr="lr", tune_opts=[0.01, 0.05, 0.1, 0.5, 1.0]
    ),
    "batch_size": config_from_argparse(
        argparse_attr="batch_size", tune_opts=[32, 64, 128, 256, 512, 1024, 2048]
    ),
    "clipping": config_from_argparse(
        argparse_attr="clipping", tune_opts=[0.1, 0.2, 0.5]
    ),
    "entropy_bonus": config_from_argparse(
        argparse_attr="entropy_bonus", tune_opts=[0.01, 0.05]
    ),
    "critic_coeff": config_from_argparse(
        argparse_attr="critic_coeff", tune_opts=[1.0]
    ),
    "sync_every": config_from_argparse(
        argparse_attr="sync_every", tune_opts=[10000]
    ),
    "n_hidden": config_from_argparse(argparse_attr="n_hidden", tune_opts=[100]),
    "n_layers": config_from_argparse(argparse_attr="n_layers", tune_opts=[2]),
    "n_channels": config_from_argparse(argparse_attr="n_channels", tune_opts=[5]),
    "num_samples": 1,
}

if __name__ == "__main__":
    ray.init()

    # This lets us use argparse on top of ray tune while conforming
    # to Tune's requirement that the train function take exactly 2
    # arguments.
    tune.register_trainable(
        "train_curried_fn", lambda config, reporter: train(args, config, reporter)
    )

    # TODO(alok) Integrate Tune reporter with tensorboardX?
    experiment_spec = tune.Experiment(
        name="CRMDP",
        run="train_curried_fn",
        stop={},
        config=TUNE_CONFIG,
        resources_per_trial={"cpu": 4, "gpu": 1},
    )

    tune.run_experiments(experiments=experiment_spec)
