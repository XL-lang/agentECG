from utils import vlm_client,qwen_vl_plus
import base64
from smolagents import Tool
class FigAnaysisTool(Tool):
    name = "fig_anaysis_tools"
    description = "give me the figure, relevant information and direction, and I will analyze the figure and give you the result."
    inputs = {
        "prompt": {
            'type': "string",
            'description': "relevant information and direction",
            },
        "figure": {
            'type': "string",
            'description': "url of the figure,usally 'output.png'",
            },
    }
    output_type = "string"

    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def forward(self, prompt, figure):
        base64_image = FigAnaysisTool.encode_image(figure)
        completion = vlm_client.chat.completions.create(
            model=qwen_vl_plus["model"],
             messages=[
    	{
    	    "role": "system",
            "content": [{"type":"text","text": "You are a helpful assistant."}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}, 
                },
                {"type": "text", "text": prompt},
            ],
        }
    ],
         
        )

     
        
        return completion.choices
if __name__ == "__main__":
    fig_anaysis_tools = FigAnaysisTool()
    prompt = "Analyze the figure and provide insights."
    figure = "output.png"
    result = fig_anaysis_tools(prompt, figure)
    print(result)