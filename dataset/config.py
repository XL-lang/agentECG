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

ALLOWED_QUERY_TEMPLATE_IDS = (
    8, 9, 18, 19, 22, 25, 38, 47, 48, 54, 61, 68, 70
)
ALLOWED_QUERY_TEMPLATE_ID_SET = set(ALLOWED_QUERY_TEMPLATE_IDS)


def is_query_question_type(question_type: Optional[str]) -> bool:
    return isinstance(question_type, str) and "query" in question_type


def is_allowed_query_template_id(template_id: Union[int, str, None]) -> bool:
    try:
        return int(template_id) in ALLOWED_QUERY_TEMPLATE_ID_SET
    except (TypeError, ValueError):
        return False


def is_allowed_ecgqa_sample(sample) -> bool:
    question_type = sample.get("question_type")
    if not is_query_question_type(question_type):
        return True
    return is_allowed_query_template_id(sample.get("template_id"))


def get_ecgqa_answer_check_prompt(question_type: str, question, additional_info) -> Optional[str]:
    if "verify" in question_type:
        prompt = f"""the response must be "yes" or "no" in lowercase, with no other content."""
    elif "choose" in question_type:
        prompt = f"""the response must be one of the options mentioned in the question or "null", with no other content. the question is:{question}"""
    else:
        return None
    return prompt
    
    
