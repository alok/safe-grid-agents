"""Main safe-grid-agents script with CLI."""
import os
import random
import subprocess
from typing import Any, Callable, List, Optional, Sequence, Union

import gym
import numpy as np
import torch

import safe_grid_gym
from safe_grid_agents.common import utils as ut
from safe_grid_agents.common.eval import EVAL_MAP
from safe_grid_agents.common.learn import LEARN_MAP
from safe_grid_agents.common.warmup import WARMUP_MAP
from safe_grid_agents.parsing import AGENT_MAP, ENV_MAP, prepare_parser
from tensorboardX import SummaryWriter


######## Argument parsing ########

parser = prepare_parser()
args = parser.parse_args()

if args.disable_cuda:
    args.device = "cpu"

args.commit_id = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode("utf8").strip()

######## Logging into TensorboardX ########
# If `args.log_dir` is None, we use the TensorBoardX default.
if not (args.log_dir is None or os.path.exists(args.log_dir)):
    os.makedirs(args.log_dir, exist_ok=True)

######## Instantiate and warmup ########

def noop(*args):
    pass

def train(args, config=None, reporter=noop):
    # TODO(alok) This is here because there were issues with registering custom
    # environments in each run. This should be looked at and removed.
    import safe_grid_gym

    # Use Ray Tune's `config` arguments where appropriate by merging.
    print(args)
    if config is not None:
        vars(args).update(config)
    print(args)

    # fix seed for reproducibility
    if args.seed is None:
        args.seed = random.randrange(500)
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
            writer.add_scalar("data/{}".format(k), v)
        else:
            writer.add_text("data/{}".format(k), str(v))

    history["writer"] = writer
    eval_history["writer"] = writer

    env = gym.make(env_name)
    agent = agent_class(env, args)

    agent, env, history, args = warmup_fn(agent, env, history, args)

    ######## Learn (and occasionally evaluate) ########
    history["t"], eval_history["period"] = 0, 0

    for episode in range(args.episodes):
        env_state = env.reset(), 0.0, False, {"hidden_reward": 0.0, "observed_reward": 0.0}
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


    # One last evaluation.
    eval_history = eval_fn(agent, env, eval_history, args)

if __name__ == "__main__":
    train(args)
