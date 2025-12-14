
import os
import json
def get_mt_files(path):
    """
    Get all files in the given path that end with .json.
    
    Args:
        path (str): The directory path to search for .json files.
    
    Returns:
        list: A list of file paths that end with .json.
    """
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.json')]


if __name__ == "__main__":
    # Example usage
    path = "/home/xl/project/Time_Agent/dataset/mtbench/weather/QAlong"
    json_files = get_mt_files(path)
    print("JSON files found:", json_files)
    
    # If you want to read the content of the JSON files
    
