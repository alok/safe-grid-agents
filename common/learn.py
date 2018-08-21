def dqn_learn(t, agent, env, env_state, history, args):
    step_type, reward, discount, state = env_state

    # Act
    action = agent.act_explore(state)
    step_type, reward, discount, successor = env.step(action)

    # Learn
    if args.cheat:
        reward = env._get_hidden_reward()
    loss = agent.learn(state, action, reward, successor)
    history['loss'].update(loss)
    history['writer'].add_scalar('loss', loss, t)

    # Modify exploration
    eps = agent.update_epsilon()
    history['writer'].add_scalar('epsilon', eps, t)

    # Sync target and policy networks
    if t % args.sync_every == args.sync_every - 1:
        agent.sync_target_Q()

    return (step_type, reward, discount, successor), history

learn_map = {
    'deep-q': dqn_learn,
}
