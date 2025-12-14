from smolagents.memory import AgentMemory,AgentError,AgentLogger 
from .model import reason_client

logger = AgentLogger()
def support_oppose_check(final_answer, memory: AgentMemory):
    if not isinstance(final_answer, str):
        if isinstance(final_answer, list):
            final_answer = " ".join(str(item) for item in final_answer)
        elif isinstance(final_answer, dict):
            final_answer = " ".join(f"{key}: {value}" for key, value in final_answer.items())
        else:
            raise AgentError("Final answer must be a string,list or dict.",logger)
    final_answer = final_answer.lower()
    if "support" in final_answer or "oppos" in final_answer:
        return True
    else:
        raise AgentError("Your final response must include options, along with analytical information supporting the options or opposing the options.For example:{{option:..., support:..., oppose:...}}",logger)
def abcd_check(final_answer, memory: AgentMemory):
    if not isinstance(final_answer, str):
            raise AgentError(f"The final answer must be one of the letters A, B, C, or D, corresponding to the four options in order, but recieved {final_answer}",logger)
    final_answer = final_answer.lower()
    if final_answer in ["a", "b", "c", "d"]:
        return True
    else:
        raise AgentError(f"The final answer must be one of the letters A, B, C, or D, corresponding to the four options in order,  but recieved {final_answer}",logger)
    
class LLM_checker:
    def __init__(self, check_prompt):
        self.check_prompt = check_prompt
    def check(self, answer,memory: AgentMemory):
        synthetic_prompt = f"The task requirement is: \n{self.check_prompt}\n The answer is: \n{answer}\n Please determine if the answer meets the task requirements. If it does, respond with '#PASS', otherwise respond with 'no' and specific issues that did not meet the requirements."
        completion = reason_client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": synthetic_prompt},
        ],

    )
        res =  completion.choices[0].message.content
        if "#PASS" in res:
            return True
        else:
            raise AgentError(f"LLM checker failed: {res}",logger)

if __name__ == "__main__":
    # Example usage
    try:
        result = support_oppose_check("I like the motion.", None)
        print("Check passed:", result)
    except AgentError as e:
        print("Check failed:", e)

    try:
        result = support_oppose_check("I oppose the motion.", None)
        print("Check passed:", result)
    except AgentError as e:
        print("Check failed:", e)

    try:
        result = abcd_check("A", None)
        print("Check passed:", result)
    except AgentError as e:
        print("Check failed:", e)
        
    try:
        result = abcd_check("E", None)
        print("Check passed:", result)
    except AgentError as e:
        print("Check failed:", e)

