from typing import List, Union, Optional
ecg_qa_types = [
    "single-verify",
    "single-choose",
    "single-query",
    "comparison_consecutive-verify",
    "comparison_consecutive-query",
    "comparison_irrelevant-verify",
    "comparison_irrelevant-query"
]
def get_ecgqa_answer_check_prompt(question_type: str, question, additional_info) -> Optional[str]:
    if "verify" in question_type:
        prompt = f"""the response must be "yes" or "no" in lowercase, with no other content."""
    elif "choose" in question_type:
        prompt = f"""the response must be one of the options mentioned in the question or "null", with no other content. the question is:{question}"""
    else:
        return None
    return prompt
    
    