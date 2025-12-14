from utils import reason_client
def run_resoning(content:str):
    completion = reason_client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content},
        ],

        extra_body={"enable_thinking": True, 'incremental_output':True},
    )
    return completion.choices[0].message.content

if __name__ == "__main__":
    content = "x*20+2 = 42, solve for x"
    result = run_resoning(content)
    print(result)  