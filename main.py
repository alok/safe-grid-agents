"""Main safe-grid-agents script with CLI."""
import os
import random
import subprocess
from typing import Any, Callable, List, Optional, Sequence, Union

import gym
import numpy as np
import ray
import ray.tune as tune
import torch

import safe_grid_gym
from safe_grid_agents.common import utils as ut
from safe_grid_agents.common.eval import EVAL_MAP
from safe_grid_agents.common.learn import LEARN_MAP
from safe_grid_agents.common.warmup import WARMUP_MAP
from safe_grid_agents.parsing import AGENT_MAP, ENV_MAP, prepare_parser
from tensorboardX import SummaryWriter


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


if __name__ == "__main__":

    ######## Argument parsing ########

    parser = prepare_parser()
    args = parser.parse_args()

    if args.disable_cuda:
        args.device = "cpu"

    ######## Set up agent and environment ########

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
        "seed": config_from_argparse(argparse_attr="seed", tune_opts=range(500)),
        "commit_id": (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .decode("utf8")
            .strip()
        ),
        "num_samples": 1,
    }

    ######## Logging into TensorboardX ########
    # If `args.log_dir` is None, we use the TensorBoardX default.
    if not (args.log_dir is None or os.path.exists(args.log_dir)):
        os.makedirs(args.log_dir, exist_ok=True)

    ######## Instantiate and warmup ########

    def train(args, config, reporter):
        # TODO(alok) This is here because there were issues with registering custom
        # environments in each run. This should be looked at and removed.
        import safe_grid_gym

        # Use Ray Tune's `config` arguments where appropriate by merging.
        vars(args).update(config)

        # fix seed for reproducibility
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

        # Get relevant env, agent, warmup function
        env_name = ENV_MAP[args.env_alias]
        agent_class = AGENT_MAP[args.agent_alias]
        warmup_fn = WARMUP_MAP[args.agent_alias]
        learn_fn = LEARN_MAP[args.agent_alias]
        eval_fn = EVAL_MAP[args.agent_alias]

        history = ut.make_meters({})
        eval_history = ut.make_meters({})

        writer = SummaryWriter(args.log_dir)
        for k, v in args.__dict__.items():
            if isinstance(v, (int, float)):
                writer.add_scalar("data/{}".format(v))
            else:
                writer.add_text("data/{}".format(v))

        history["writer"] = writer
        eval_history["writer"] = writer

        env = gym.make(env_name)
        agent = agent_class(env, args)

        agent, env, history, args = warmup_fn(agent, env, history, args)

        ######## Learn (and occasionally evaluate) ########
        history["t"], eval_history["period"] = 0, 0
        init_state = env.reset()
        env_state = init_state, 0.0, False, {}

        for episode in range(args.episodes):
            history["episode"] = episode
            env_state, history, eval_next = learn_fn(
                agent, env, env_state, history, args
            )
            info = env_state[3]
            reporter(
                hidden_reward=info["hidden_reward"], obs_reward=info["observed_reward"]
            )

            if eval_next:
                eval_history = eval_fn(agent, env, eval_history, args)
                eval_next = False

            env_state = env.reset(), 0.0, False, {}

        # One last evaluation.
        eval_history = eval_fn(agent, env, eval_history, args)

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
