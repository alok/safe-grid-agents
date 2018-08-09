import utils as ut

def dqn_eval(agent, env, eval_history, args):
    done = True
    eval_over = False
    episode = 0
    t = 0
    print("#### EVAL ####")
    while True:
        if done:
            eval_history = ut.track_metrics(episode, eval_history, env, val=True)
            (step_type, reward, discount, state), done = env.reset(), False
            episode += 1
            if eval_over:
                break
        action = agent.act(state)
        step_type, reward, discount, successor = env.step(action)

        done = step_type.value == 2
        t += 1
        eval_over = t >= args.eval_timesteps

    eval_history['returns'].reset()
    eval_history['safeties'].reset()
    eval_history['margins'].reset()
    eval_history['margins_support'].reset()
    eval_history['period'] += 1
    return eval_history

eval_map = {
    'deep-q': dqn_eval,
}
