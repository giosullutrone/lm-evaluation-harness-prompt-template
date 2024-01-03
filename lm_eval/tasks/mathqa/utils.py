import re


def doc_to_choice(doc):
    choices = [
        c[4:].rstrip(" ,")
        for c in re.findall(r"[abcd] \) .*?, |e \) .*?$", doc["options"])
    ]
    return choices

def text_to_prompt(text: str) -> str:
    return f'[INST] \n{text} [/INST]'