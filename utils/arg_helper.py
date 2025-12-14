import pandas as pd
def analyze_dict_types(data: dict) -> dict:
    """返回与data同结构的dict，但所有value替换为表示其类型的字符串。"""
    def handle_value(value):
        # 如果是dict，递归处理
        if isinstance(value, dict):
            return {k: handle_value(v) for k, v in value.items()}
        # 如果是list，判断元素类型是否一致
        elif isinstance(value, list):
            if not value:  # 空列表
                return "List[Empty]"
            first_type = type(value[0])
            if all(isinstance(item, first_type) for item in value):
                return f"List[{first_type.__name__}]"
            else:
                return "list"
        # 否则，直接返回类型名
        else:
            return type(value).__name__

    return {key: handle_value(val) for key, val in data.items()}

def tidy_args_to_str(tidy_args: dict) -> str:
    """
    Convert dict to string, but if a dict value is a list longer than 10 elements,
    only keep the first 10, and if the list has float elements, only keep one decimal place.
    Place the ellipsis inside the list bracket before the closing bracket.
    """
    def format_value(value):
        if isinstance(value, list):
            truncated_values = []
            for i, item in enumerate(value):
                if i < 10:
                    if isinstance(item, float):
                        truncated_values.append(f"{item:.1f}")
                    else:
                        truncated_values.append(str(item))
                else:
                    break
            if len(value) > 10:
                return "[" + ", ".join(truncated_values) + ", ...]"
            else:
                return "[" + ", ".join(truncated_values) + "]"
        elif isinstance(value, float):
            return f"{value:.1f}"
        else:
            return str(value)

    items_str = []
    for k, v in tidy_args.items():
        items_str.append(f"'{k}': {format_value(v)}")
    return "{" + ", ".join(items_str) + "}"
def dataframe_to_string(df: pd.DataFrame) -> str:
    row_count = len(df)
    if row_count <= 5:
        return df.to_string()
    else:
        omitted_count = row_count - 5
        head_str = df.head().to_string()
        return f"{head_str}\n......Display only the first five lines, with the remaining {omitted_count} lines omitted."

if __name__ == "__main__":
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6], "y": [10, 20, 30, 40, 50, 60]})
    print(dataframe_to_string(df))