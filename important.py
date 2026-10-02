for iteration in range(MAX_ITERATIONS):
    observation = observe(task, history)

    plan = think(task, observation)

    result = act(plan)

    reflection = reflect(task, observation, plan.action, result)

    if reflection.completed:
        return generate_answer(task, history)
