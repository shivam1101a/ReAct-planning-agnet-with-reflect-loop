import os
from openai import OpenAI
from pydantic import BaseModel, Field
from dotenv import load_dotenv


# --------------------------------------------------
# 1. SETUP
# --------------------------------------------------

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MODEL = "gpt-5.6-luna"
MAX_ITERATIONS = 5


# --------------------------------------------------
# 2. STRUCTURED OUTPUT MODELS
# --------------------------------------------------


class Plan(BaseModel):
    thought: str = Field(description="Reasoning about what should be done next")

    action: str = Field(description="The action the agent wants to take")

    tool: str | None = Field(
        default=None, description="Tool to use: calculator or none"
    )

    tool_input: str | None = Field(
        default=None, description="Input to the selected tool"
    )


class Reflection(BaseModel):
    completed: bool = Field(
        description="Whether the task has been successfully completed"
    )

    confidence: float = Field(
        ge=0, le=1, description="Confidence that the task is complete"
    )

    critique: str = Field(description="Critique of the previous action/result")

    next_step: str = Field(
        description="What should happen next if the task is not complete"
    )


# --------------------------------------------------
# 3. TOOL
# --------------------------------------------------


def calculator(expression: str):
    """Simple calculator tool."""

    allowed = set("0123456789+-*/(). ")

    if not expression:
        return "No expression provided."

    if any(char not in allowed for char in expression):
        return "Invalid characters in expression."

    try:
        result = eval(expression, {"__builtins__": {}}, {})

        return str(result)

    except Exception as e:
        return f"Calculator error: {e}"


# --------------------------------------------------
# 4. OBSERVE
# --------------------------------------------------


def observe(task, history):

    if not history:
        return f"Initial observation: The user wants you to solve this task: {task}"

    return (
        "Previous agent history:\n" + "\n".join(history) + "\n\n"
        "Use this information to determine what should happen next."
    )


# --------------------------------------------------
# 5. THINK / PLAN
# --------------------------------------------------


def think(task, observation):

    prompt = f"""
You are the planning component of a ReAct agent.

USER TASK:
{task}

CURRENT OBSERVATION:
{observation}

Available tool:

1. calculator
   - Used for arithmetic calculations.

Your job is to decide the NEXT action.

Rules:
- Do not perform multiple actions at once.
- If calculation is required, use calculator.
- If enough information already exists, prepare to finish.
- Do not invent tool results.
"""

    response = client.responses.parse(model=MODEL, input=prompt, text_format=Plan)

    return response.output_parsed


# --------------------------------------------------
# 6. ACT
# --------------------------------------------------


def act(plan):

    print("\nACTION:", plan.action)

    if plan.tool == "calculator":
        print("TOOL:", plan.tool)
        print("INPUT:", plan.tool_input)

        result = calculator(plan.tool_input)

        print("TOOL RESULT:", result)

        return result

    return "No tool was used."


# --------------------------------------------------
# 7. REFLECT
# --------------------------------------------------


def reflect(task, observation, action, result):

    prompt = f"""
You are the self-critique component of a ReAct agent.

USER TASK:
{task}

OBSERVATION:
{observation}

ACTION:
{action}

RESULT:
{result}

Evaluate whether the task is actually complete.

Do NOT assume that an action succeeded merely because it was attempted.

If the task is incomplete:
- explain what is missing
- identify a useful next step

If it is complete:
- explain why

Return a confidence between 0 and 1.
"""

    response = client.responses.parse(model=MODEL, input=prompt, text_format=Reflection)

    return response.output_parsed


# --------------------------------------------------
# 8. FINAL ANSWER
# --------------------------------------------------


def generate_answer(task, history):

    prompt = f"""
Answer the user's task.

USER TASK:
{task}

AGENT HISTORY:
{history}

Use the useful results from the history.

Do not claim that something happened unless the history supports it.
Give a concise final answer.
"""

    response = client.responses.create(model=MODEL, input=prompt)

    return response.output_text


# --------------------------------------------------
# 9. GRACEFUL DEGRADATION
# --------------------------------------------------


def graceful_fallback(task, history):

    print("\nMAXIMUM ITERATIONS REACHED.")

    prompt = f"""
The agent could not completely finish the task within
the allowed number of iterations.

USER TASK:
{task}

HISTORY:
{history}

Give the best partial answer possible.

Clearly state what was established and what could not
be established.

Do not invent missing information.
"""

    response = client.responses.create(model=MODEL, input=prompt)

    return response.output_text


# --------------------------------------------------
# 10. MAIN REACT LOOP
# --------------------------------------------------


def run_agent(task):

    history = []

    for iteration in range(MAX_ITERATIONS):
        print("\n" + "=" * 60)
        print(f"ITERATION {iteration + 1}")
        print("=" * 60)

        # -----------------------------
        # OBSERVE
        # -----------------------------

        observation = observe(task, history)

        print("\nOBSERVE:")
        print(observation)

        # -----------------------------
        # THINK
        # -----------------------------

        plan = think(task, observation)

        print("\nTHINK:")
        print(plan.thought)

        # -----------------------------
        # ACT
        # -----------------------------

        result = act(plan)

        # -----------------------------
        # STORE EXPERIENCE
        # -----------------------------

        history.append(
            f"""
Iteration {iteration + 1}

Thought:
{plan.thought}

Action:
{plan.action}

Result:
{result}
"""
        )

        # -----------------------------
        # REFLECT
        # -----------------------------

        reflection = reflect(task, observation, plan.action, result)

        print("\nREFLECT:")
        print(reflection.critique)

        print("Confidence:", reflection.confidence)

        # -----------------------------
        # DECIDE
        # -----------------------------

        if reflection.completed:
            print("\nTASK COMPLETED.")

            return generate_answer(task, "\n".join(history))

        print("\nNot complete.", reflection.next_step)

    # --------------------------------------------------
    # MAX ITERATIONS
    # --------------------------------------------------

    return graceful_fallback(task, "\n".join(history))


# --------------------------------------------------
# 11. PROGRAM ENTRY
# --------------------------------------------------

if __name__ == "__main__":
    task = input("What do you want the agent to do? ")

    answer = run_agent(task)

    print("\n" + "=" * 60)
    print("FINAL ANSWER")
    print("=" * 60)

    print(answer)
