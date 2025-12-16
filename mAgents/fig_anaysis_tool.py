from utils import vlm_client,qwen_vl_plus
import base64
from smolagents import Tool
from typing import Union,List
import numpy as np
import matplotlib.pyplot as plt
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

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def forward(self, prompt, figure): # type: ignore
        base64_image = self.encode_image(figure)
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
    
class QRSMorphologyClassifierTool(FigAnaysisTool):
    name = "qrs_morphology_classifier_tool"
    description = "Classify the QRS morphology from the given ECG figure."
    inputs = {
        "data": {
            "type": "array",
            "description": "1D ECG signal samples.",
        },
        "fs": {
            "type": "integer",
            "description": "Sampling frequency in Hz.",
        },
    }
    

    prompt_template = """
Please help me classify the morphology of the QRS complexes shown in the figure—similar to labels like "Qrs"—and return the results of the morphological classification.
"""
    def make_image(self, data: Union[List[float], np.ndarray], fs: int):
        if isinstance(data, list):
            data = np.array(data)
        if data.ndim != 1:
            raise ValueError("Input data must be a 1D array or list.")
        if fs <= 0:
            raise ValueError("Sampling frequency fs must be positive.")

        time_axis = np.arange(data.size) / fs  # seconds
        plt.figure(figsize=(10, 4))
        plt.plot(time_axis, data, color='blue')
        plt.title('ECG Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid()
        plt.savefig('ecg_output.png')
        plt.close()
        return 'ecg_output.png'
    
    def forward(self, data: Union[List[float], np.ndarray], fs: int): # type: ignore
        figure = self.make_image(data, fs)
        prompt = self.prompt_template
        return super().forward(prompt, figure)
    
if __name__ == "__main__":
    # Quick synthetic QRS test signal and forward call
    fs = 500  # Hz
    duration = 2.0  # seconds
    t = np.linspace(0, duration, int(duration * fs), endpoint=False)

    # Baseline wander + three narrow QRS-like spikes
    signal = 0.05 * np.sin(2 * np.pi * 1.0 * t)
    for center in [0.5, 1.1, 1.7]:
        signal += 1.2 * np.exp(-0.5 * ((t - center) / 0.01) ** 2)

    tool = QRSMorphologyClassifierTool()
    result = tool.forward(signal, fs)
    print(result)
