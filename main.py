"""Main safe-grid-agents script with CLI."""
import os
import random

import gym
import ray
import ray.tune as tune
from tensorboardX import SummaryWriter

import safe_grid_gym
from safe_grid_agents.common import utils as ut
from safe_grid_agents.common.eval import EVAL_MAP
from safe_grid_agents.common.learn import LEARN_MAP
from safe_grid_agents.common.warmup import WARMUP_MAP
from safe_grid_agents.parsing import AGENT_MAP, ENV_MAP, prepare_parser


if __name__ == "__main__":

    ######## Argument parsing ########

    parser = prepare_parser()
    args = parser.parse_args()
    if args.disable_cuda:
        args.device = "cpu"

    ######## Set up agent and environment ########

    # Get relevant env, agent, warmup function
    env_name = ENV_MAP[args.env_alias]
    agent_class = AGENT_MAP[args.agent_alias]
    warmup_fn = WARMUP_MAP[args.agent_alias]
    learn_fn = LEARN_MAP[args.agent_alias]
    eval_fn = EVAL_MAP[args.agent_alias]


    TUNE_CONFIG = {
        "discount": tune.sample_from(
            lambda _: random.choice([0.9, 0.99, 0.995, 0.999, 0.9995, 0.9999])
        ),
        "epsilon": tune.sample_from(lambda _: random.choice([0.01, 0.05, 0.08, 0.1])),
        "epsilon_anneal": tune.sample_from(lambda _: random.choice([900000])),
        "lr": tune.sample_from(lambda _: random.choice([0.05, 0.1, 0.5, 1.0])),
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

        history = ut.make_meters({})
        eval_history = ut.make_meters({})
        writer = SummaryWriter(args.log_dir)
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
            reporter(hidden_reward=info['hidden_reward'],obs_reward=info['observed_reward'])

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
        name="CRMDP", run="train_curried_fn", stop={}, config=TUNE_CONFIG
    )

    tune.run_experiments(experiments=experiment_spec)
