from utils import reason_client
class LLM_checker:
    def __init__(self, check_prompt):
        self.check_prompt = check_prompt
    def check(self, answer):
        synthetic_prompt = f"The task requirement is: \n{self.check_prompt}\nYour responses should not include the question, and please verify that the number of answers is correct.  The answer is: \n{answer}\n Please determine if the answer meets the task requirements. If it does, respond with '#PASS', otherwise respond with 'no' and specific issues that did not meet the requirements."
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
            return False,res