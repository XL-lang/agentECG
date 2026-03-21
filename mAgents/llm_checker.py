from utils import reason_client
import typing

class LLM_checker:
    def __init__(self, check_prompt):
        self.check_prompt = check_prompt
    def check(self, answer, mem):
        if self.check_prompt is None:
            return True, "No check needed"
        synthetic_prompt = f"{self.check_prompt}{answer}"
        completion = reason_client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": synthetic_prompt},
        ],

    )
        res =  completion.choices[0].message.content
        if res is None:
            return False, "No response from LLM"
        if "#PASS" in res:
            return True
        else:
            return False,res
class ECGQA_EMA_checker:
    def __init__(self, choices) -> None:
        self.choices:typing.List[str] = choices
    def check(self, answer: str, mem):
        if not isinstance(answer, str):
            return False, "Answer is not a string"
        answer = answer.lower()
        answer_set = set([choice.strip() for choice in self.choices if choice.strip().lower() in answer])
        if len(answer_set) == 0:
            return False, "No valid answer found, answer must be one or more of the following options: " + ", ".join(self.choices)
        single_select_set = set(["yes", "no", "none"])
        if len(answer_set & single_select_set) > 0:
            if len(answer_set) > 1:
                return False, f"The allowed responses are: {self.choices}. Extract from the response using regular expressions to get your answer: {answer_set}. Note that (yes, no, none) can only appear in single responses. Please revise your answer to avoid ambiguity."
        return True, "Valid  answer"
