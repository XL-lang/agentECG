from smolagents.agents import ActionStep,CodeAgent
import os
import datetime 

def process_string(text: str) -> str:
    import re
    
    def format_list(lst):
        formatted = []
        for i, item in enumerate(lst):
            if i == 10:
                break
            try:
                val = float(item)
                formatted.append(f"{val:.1f}")
            except ValueError:
                formatted.append(item)
            if i == 9 and len(lst) > 10:
                formatted[-1] += "..."
        return f"[{','.join(formatted)}]"


    matches = re.findall(r'\[[^\]]*\]', text)
    result = text
    for m in matches:
        content = m.strip("[]").split(",")
        new_list = format_list([c.strip() for c in content])
        if new_list:
            result = result.replace(m, new_list, 1)
    return result







def clean_memory(memory_step: ActionStep, agent: CodeAgent, log=True) -> None:
    if not isinstance(memory_step, ActionStep):
        return
    last_step = memory_step
    if last_step.error:
        last_step.error.message = process_string(last_step.error.message)
    else:
        cleaned_steps = []
        for s in agent.memory.steps:
            
            if isinstance(s, ActionStep) and s.error:
                continue
            cleaned_steps.append(s)
        agent.memory.steps = cleaned_steps


            
            
