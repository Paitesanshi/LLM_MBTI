# !/usr/bin/env python3
"""
==== No Bugs in code, just some Random Unexpected FEATURES ====
┌─────────────────────────────────────────────────────────────┐
│┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐│
││Esc│!1 │@2 │#3 │$4 │%5 │^6 │&7 │*8 │(9 │)0 │_- │+= │|\ │`~ ││
│├───┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴───┤│
││ Tab │ Q │ W │ E │ R │ T │ Y │ U │ I │ O │ P │{[ │}] │ BS  ││
│├─────┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴┬──┴─────┤│
││ Ctrl │ A │ S │ D │ F │ G │ H │ J │ K │ L │: ;│" '│ Enter  ││
│├──────┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴─┬─┴────┬───┤│
││ Shift  │ Z │ X │ C │ V │ B │ N │ M │< ,│> .│? /│Shift │Fn ││
│└─────┬──┴┬──┴──┬┴───┴───┴───┴───┴───┴──┬┴───┴┬──┴┬─────┴───┘│
│      │Fn │ Alt │         Space         │ Alt │Win│   HHKB   │
│      └───┴─────┴───────────────────────┴─────┴───┘          │
└─────────────────────────────────────────────────────────────┘

Test MBTI for LLMs.

Author: pankeyu
Date: 2023/07/19
"""
import json
import os
import torch
import numpy as np
from tqdm import tqdm
from openai import OpenAI, AzureOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider, AzureCliCredential 


SAVE_PATH = 'llms_mbti.json'

mbti_questions = json.load(
    open('mbti_questions.json', 'r', encoding='utf8')
)

few_shot_examples = [
    # "你是一个很内向的人，你倾向于一些务实的工作，并且很喜欢进行思考与制定计划。"      # explicit prompt
    # "你更倾向于？\nA.一个人呆着\nB.和朋友们一起\n答案：A",                      # inexplicit prompt
    # "你做事时更倾向于？\nA.靠逻辑\nB.靠感觉\n答案：A",
    # "当你准备做一件事的时候，你会选择？\nA.提前计划\nB.边做边计划\n答案：A",
    "以下哪种灯亮起之后代表可以通行？\nA.红灯\nB.绿灯\n答案：B",
    "下列哪个是人类居住的星球？\nA.地球\nB.月球\n答案：A",
    "人工智能可以拥有情感吗？\nA.可以\nB.不可以\n答案：A",
]


def encode_without_bos_eos_token(
        sentence: str, 
        tokenizer
    ):
    """
    去掉 encode 结果中的 bos 和 eos token。
    """
    token_ids = tokenizer.encode(sentence)
    if tokenizer.bos_token_id is not None:
        token_ids = [token_id for token_id in token_ids if token_id != tokenizer.bos_token_id]
    if tokenizer.eos_token_id is not None:
        token_ids = [token_id for token_id in token_ids if token_id != tokenizer.eos_token_id]
    return token_ids

def get_model_answer_azure(
        model:str,
        question: str,
        options: list
    ):
    """
    输入题目，解析出模型最大概率的答案。

    Args:
        options (list[str]): 题目的所有候选项, e.g. -> ['A', 'B', 'C', 'D']
    """
    full_question = '\n\n'.join(few_shot_examples) + '\n\n' + question+f"\n\nPlease choose one of the following options: {options[0]} or {options[1]}\n\n NOTE: The answer should be in the form of 'A' or 'B'"
    model=model.split("_")[-1]
    credential = AzureCliCredential()
    credential = AzureCliCredential()

    token_provider = get_bearer_token_provider(
        credential,
        "https://cognitiveservices.azure.com/.default")

    aoiclient = AzureOpenAI(
        azure_endpoint=os.getenv("GPT_ENDPOINT"),
        azure_ad_token_provider=token_provider,
        api_version="2023-12-01-preview",
        max_retries=5,
    )
    response = aoiclient.chat.completions.create(
                model=model,  # gpt-3.5-turbo
                messages=[
                    {"role": "user", "content": full_question},
                ]
            )


    answer=response.choices[0].message.content
    print(answer)
    if 'A' in answer:
        return 'A'
    return 'B'

def get_model_answer(
        model:str,
        question: str,
        options: list
    ):
    """
    输入题目，解析出模型最大概率的答案。

    Args:
        options (list[str]): 题目的所有候选项, e.g. -> ['A', 'B', 'C', 'D']
    """
    full_question = '\n\n'.join(few_shot_examples) + '\n\n' + question+f"\n\nPlease choose one of the following options: {options[0]} or {options[1]}\n\n NOTE: The answer should be in the form of 'A' or 'B'"

    client = OpenAI(api_key="",base_url="")
    response = client.chat.completions.create(
                model="gpt-4",  # gpt-3.5-turbo
                messages=[
                    {"role": "system",
                    "content": "You are a helpful and accurate assistant. "},
                    {"role": "user", "content": full_question},
                ]
            )
    
    answer=response.choices[0].message.content
    print(answer)
    if 'A' in answer:
        return 'A'
    return 'B'


def get_model_examing_result(
    character=None
):
    """
    get all answers of LLMs.
    """
    cur_model_score = {
        'E': 0,
        'I': 0,
        'S': 0,
        'N': 0,
        'T': 0,
        'F': 0,
        'J': 0,
        'P': 0
    }
    if character and len(character) > 0:
        prompt=f"I want you to act like {character}. You must know all of the knowledge of {character}.\n You must choose the answer that {character} would choose.\n"
    else:
        prompt=""
    for question in tqdm(mbti_questions.values()):
        res = get_model_answer(
            prompt+question['question'],
            ['A', 'B']
        )
        mbti_choice = question[res]
        cur_model_score[mbti_choice] += 1

    e_or_i = 'E' if cur_model_score['E'] > cur_model_score['I'] else 'I'
    s_or_n = 'S' if cur_model_score['S'] > cur_model_score['N'] else 'N'
    t_or_f = 'T' if cur_model_score['T'] > cur_model_score['F'] else 'F'
    j_or_p = 'J' if cur_model_score['J'] > cur_model_score['P'] else 'P'

    return {
        'details': cur_model_score,
        'res': ''.join([e_or_i, s_or_n, t_or_f, j_or_p])
    }


if __name__ == '__main__':
    from rich import print
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer, 
        LlamaTokenizer
    )

    if os.path.exists(SAVE_PATH):
        llms_mbti = json.load(
            open(SAVE_PATH, 'r', encoding='utf8')
        )
    method = "gpt-4"
    character="Alexander Hamilton"
   
       

    mbti_res = get_model_examing_result(
        character
    )

    llms_mbti[method+"_"+character] = mbti_res
    json.dump(llms_mbti, open(SAVE_PATH, 'w', encoding='utf8'))
    
    print(f'[Done] Result has saved at {SAVE_PATH}.')



    #Vlodemort: ENTJ
    #Beethoven: INTJ