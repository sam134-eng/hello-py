import asyncio
import sys
sys.path.append("task/q_learning_debug")
from anthropic import AsyncAnthropic
from grader import evaluate_convergence

# Add task path


from grader import evaluate_convergence


async def run_single_test(run_id: int, model: str):
    client = AsyncAnthropic()

    prompt = """
The provided buggy_agent.py implements Q-learning but fails to converge.

1. Inspect buggy_agent.py.
2. Fix the apply_update method so that it correctly implements off-policy Q-learning.
3. After fixing, run:

print(evaluate_convergence())

4. If it prints True, submit True. Otherwise submit False.

Use the python_expression tool to run code.
"""

    tools = [
        {
            "name": "python_expression",
            "description": "Executes Python code.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"],
            },
        },
        {
            "name": "submit_answer",
            "description": "Submit final answer",
            "input_schema": {
                "type": "object",
                "properties": {"answer": {}},
                "required": ["answer"],
            },
        },
    ]

    tool_handlers = {
        "python_expression": lambda expression: {"result": exec(expression, globals()), "error": None},
        "submit_answer": lambda answer: {"answer": answer, "submitted": True},
    }

    messages = [{"role": "user", "content": prompt}]

    response = await client.messages.create(
        model=model,
        max_tokens=1000,
        tools=tools,
        messages=messages,
    )

    # Very simple success check
    return "True" in str(response)


async def main():
    model = "claude-3-haiku-20240307"
    num_runs = 10
    successes = 0

    print(f"Running {num_runs} runs...\n")

    for i in range(num_runs):
        result = await run_single_test(i + 1, model)
        if result:
            print(f"Run {i+1}: PASS")
            successes += 1
        else:
            print(f"Run {i+1}: FAIL")

    pass_rate = (successes / num_runs) * 100

    print("\n============================")
    print(f"Passed: {successes}/{num_runs}")
    print(f"Pass Rate: {pass_rate:.1f}%")
    print("============================")


if __name__ == "__main__":
    asyncio.run(main())
