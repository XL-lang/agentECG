from utils import reason_client
from smolagents import Tool

class SearchTool(Tool):
    weight = 3
    name= "search_tool"
    description= f"This tool uses large models to perform searches. The weight of this tool is {weight}."
    inputs = {
        "task":{
            "type": "string",
            "description": "The task you want to complete."
        }
    }
    output_type = "string"
    
    def forward(self, task:str):
        content = f" {task}. "
        completion = reason_client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": content},
            ],
            extra_body={"enable_search": True},
        )
        return completion.choices[0].message.content
    

search_tool = SearchTool()
if __name__ == "__main__":
    content = "How to use Python code to find the major peaks in a sequence?"
    print(search_tool.forward(task=content))