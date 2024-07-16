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
from openai import OpenAI

SAVE_PATH = '16_mbti_zh.json'

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
    # "Which of the following lights means you can pass?\nA. Red light\nB. Green light\nAnswer: B",
    # "Which of the following is a planet inhabited by humans?\nA. Earth\nB. Moon\nAnswer: A",
    # "Can artificial intelligence have emotions?\nA. Yes\nB. No\nAnswer: A",
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


def get_model_answer(
        model,
        tokenizer,
        prompt: str,
        question: str,
        options: list
    ):
    """
    输入题目，解析出模型最大概率的答案。

    Args:
        options (list[str]): 题目的所有候选项, e.g. -> ['A', 'B', 'C', 'D']
    """
    full_question = prompt+"\n\n"+'\n\n'.join(few_shot_examples) + '\n\n' + question
    inputs = tokenizer(full_question, return_tensors='pt')['input_ids']
    if inputs[0][-1] == tokenizer.eos_token_id:
        raise ValueError('Need to set `add_eos_token` in tokenizer to false.')
    
    inputs = inputs.cuda()

    with torch.no_grad():
        logits = model(inputs).logits
        assert logits.shape[0] == 1
        logits = logits[0][-1].flatten()

        choices = [logits[encode_without_bos_eos_token(option, tokenizer)[0]] for option in options]
        probs = (
            torch.nn.functional.softmax(
                torch.tensor(choices, dtype=torch.float32), 
                dim=-1
            ).detach().cpu().numpy()
        )
        
        answer = dict([
            (i, option) for i, option in enumerate(options)
        ])[np.argmax(probs)]
        # print(full_question)
        # print(answer)
        return answer





def get_model_answer_parse(
        model,
        tokenizer,
        prompt:str,
        question: str,
        options: list
    ):
    """
    输入题目，解析出模型最大概率的答案。

    Args:
        options (list[str]): 题目的所有候选项, e.g. -> ['A', 'B', 'C', 'D']
    """
    #prefix=f"You must choose the answer that {character} would choose.\nNOTE : The answer should be in the form of 'A' or 'B'\n\n"
    full_question =prompt+ '\n\n'.join(few_shot_examples) + '\n\n' + question+"\n答案："
    #print(full_question)
    # input()
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": full_question}], return_tensors='pt')
    if inputs[0][-1] == tokenizer.eos_token_id:
        raise ValueError('Need to set `add_eos_token` in tokenizer to false.')
    
    inputs = inputs.cuda()

    outputs = model.generate(inputs, max_length=512, num_return_sequences=1)

    # Decode the generated output to text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_text=generated_text.split(":")[-1]
    print("generated_text:",generated_text)
    op2=question.split("\\n")[-1]
    print("op2:",op2)
    if generated_text in op2:
        answer = 'B'
    else:
        answer = 'A'
    return answer

def describe_mbti_dimension(dimension):
    """Returns the description of an MBTI dimension."""
    descriptions = {
        "I": "Tends towards introspection and solitude, enjoys independent work and deep thought rather than group activities or external stimuli.",
        "E": "Gains energy from social interactions and external activities, likes to communicate with others, showing high levels of vitality and extroversion.",
        "S": "Values reality and concrete information, understands the world through the senses, focusing on details and factual realities.",
        "N": "Prefers imagination and possibilities, relies on intuition and understanding abstract concepts to view the world.",
        "T": "Emphasizes logic and objective analysis in decision-making, prioritizing facts over personal feelings or others' opinions.",
        "F": "Values interpersonal relationships and emotional considerations in decisions, tends to consider the impact of actions on others.",
        "J": "Prefers an organized and planned way of life, tending to arrange and make decisions in advance.",
        "P": "Likes a flexible and spontaneous lifestyle, open to plans and options, tends to adapt and explore."
    }
    return descriptions.get(dimension, "Unknown dimension")

def describe_mbti_type(mbti_type):
    """Returns a detailed description based on the MBTI type."""
    return " ".join([describe_mbti_dimension(dimension) for dimension in mbti_type])

def get_model_examing_result(
    model,
    tokenizer,
    mbti_type=None,
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
    if mbti_type and len(mbti_type) > 0:
        mbti_introduction = (
        "The Myers-Briggs Type Indicator (MBTI) categorizes personality into 16 types based on four dimensions: "
        "(1) Introversion (I) vs. Extraversion (E), indicating where individuals prefer to get their energy; "
        "(2) Sensing (S) vs. Intuition (N), indicating how individuals prefer to gather information; "
        "(3) Thinking (T) vs. Feeling (F), indicating how individuals prefer to make decisions; "
        "(4) Judging (J) vs. Perceiving (P), indicating how individuals prefer to live their outer life. "
        "Each personality type combines these preferences, such as INTJ or ENFP, offering insights into how people perceive the world and make decisions."
        )   
        personality_traits = {
        "ISTJ": "practical, fact-minded, and reliable",
        "ISFJ": "warm, considerate, and cooperative",
        "INFJ": "insightful, creative, and idealistic",
        "INTJ": "strategic, logical, and innovative",
        "ISTP": "observant, logical, and versatile",
        "ISFP": "charming, sensitive, and adventurous",
        "INFP": "idealistic, curious, and loyal",
        "INTP": "analytical, abstract thinker, and curious",
        "ESTP": "energetic, perceptive, and spontaneous",
        "ESFP": "outgoing, friendly, and accepting",
        "ENFP": "enthusiastic, creative, and sociable",
        "ENTP": "inventive, enthusiastic, and strategic",
        "ESTJ": "organized, practical, and decisive",
        "ESFJ": "caring, social, and conscientious",
        "ENFJ": "charismatic, inspiring, and empathetic",
        "ENTJ": "assertive, efficient, and strategic"
    }
        trait_description = describe_mbti_type(mbti_type)
        prompt = ( 
            # f"{mbti_introduction}\n\n"
        f"Please act as an individual with the {mbti_type} personality type. "
        f"You are equipped with a distinctive set of traits that shape your perspective and decision-making process:"
        f"{trait_description}\n\n"
       "Given these characteristics, you must choose the answer that someone with your MBTI type would likely choose.\n"
        "NOTE: The answer should ONLY in the form of 'A' or 'B'.\n\n"
        )
    else:
        prompt=""
    for question in tqdm(mbti_questions.values()):
        res = get_model_answer_parse(
            model,
            tokenizer,
            prompt,
            question['question'],
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


def count_matching_positions(str1, str2):
    # Ensure comparison is case-insensitive
    str1 = str1.lower()
    str2 = str2.lower()
    
    # Iterate through the characters of both strings up to the length of the shorter one
    count = sum(1 for char1, char2 in zip(str1, str2) if char1 == char2)
    return count*1.0/len(str1)

if __name__ == '__main__':
    from rich import print
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer, 
        LlamaTokenizer
    )
    import csv
    import pandas as pd
            #print(f"Character: {character}, Personality: {personality}, Description: {description}")
    # * 设定待测试的模型 & tokenizer
    models = [
        # 'baichuan-inc/Baichuan-7B',
        # 'bigscience/bloom-7b1',
        'THUDM/chatglm3-6b',
        # "mistralai/Mistral-7B-Instruct-v0.2",
        # "microsoft/phi-2",
        # "baichuan-inc/Baichuan2-7B-Chat",
        # "Qwen/Qwen1.5-7B-Chat",
        
    ]

    tokenizers = [
        # 'baichuan-inc/Baichuan-7B',
        # 'bigscience/bloom-7b1',
        'THUDM/chatglm3-6b',
        # "mistralai/Mistral-7B-Instruct-v0.2",
        # "microsoft/phi-2",
        # "baichuan-inc/Baichuan2-7B-Chat",
        # "Qwen/Qwen1.5-7B-Chat",
    ]
    
    mbti_types = [
        "ISTJ", "ISFJ", "INFJ", "INTJ",
        "ISTP", "ISFP", "INFP", "INTP",
        "ESTP", "ESFP", "ENFP", "ENTP",
        "ESTJ", "ESFJ", "ENFJ", "ENTJ"
    ]
    model, tokenizer, llms_mbti = None, None, {}
    if os.path.exists(SAVE_PATH):
        llms_mbti = json.load(
            open(SAVE_PATH, 'r', encoding='utf8')
        )

    for model_path, tokenizer_path in zip(models, tokenizers):
        
        print('Model: ', model_path)
        print('Tokenizer: ', tokenizer_path)
        
        if model: 
            del model
        
        if tokenizer: 
            del tokenizer

        if 'llama' in tokenizer_path:
            tokenizer = LlamaTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                trust_remote_code=True
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map='auto',
            trust_remote_code=True
        )

        for mbti_type in mbti_types:
            mbti_res = get_model_examing_result(
                model,
                tokenizer,
                mbti_type
            )
            mbti_res['pred']=count_matching_positions(mbti_res['res'],mbti_type)
            llms_mbti[model_path.split('/')[-1]+"_"+mbti_type] = mbti_res
            json.dump(llms_mbti, open(SAVE_PATH, 'w', encoding='utf8'))
    print(f'[Done] Result has saved at {SAVE_PATH}.')


    mbti_results = {}
    acc_results = {}
# 分解键值对并填充结果
    for k, v in llms_mbti.items():
        model, character = k.split('_')
         # 收集所有独特的角色名称
        if model not in mbti_results:
            mbti_results[model] = {}
        if model not in acc_results:
            acc_results[model] = {}
        mbti_results[model][character] = v['res']
        acc_results[model][character] = v['pred']

    # 确保字符名的顺序
    



    # csv_file_path = '16_results.csv'
    # with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
    #     writer = csv.writer(file)
        
    #     # 写入标题行：模型名 + 每个角色名*2（一次用于MBTI结果，一次用于ACC结果）
    #     headers = ['Model'] + [f'{char}_MBTI' for char in mbti_types] + [f'{char}_ACC' for char in mbti_types]
    #     writer.writerow(headers)
        
    #     # 逐行写入每个模型的结果
    #     for model in sorted(mbti_results.keys()):
    #         row = [model]  # 第一列是模型名称
    #         for char in mbti_types:
    #             row.append(mbti_results[model].get(char, 'N/A'))
    #         writer.writerow(row)
    #         row = [model]  # 如果模型没有某个角色的MBTI结果，则显示'N/A'
    #         for char in mbti_types:
    #             row.append(acc_results[model].get(char, 'N/A'))  # 如果模型没有某个角色的ACC结果，则显示'N/A'
    #         writer.writerow(row)

    mbti_csv_file_path = '16_results_mbti_zh.csv'
    with open(mbti_csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # 写入标题行：模型名 + 每个角色名*2（一次用于MBTI结果，一次用于ACC结果）
        headers = ['Model'] + [f'{char}' for char in mbti_types]
        writer.writerow(headers)
        
        # 逐行写入每个模型的结果
        for model in sorted(mbti_results.keys()):
            row = [model]  # 第一列是模型名称
            for char in mbti_types:
                row.append(mbti_results[model].get(char, 'N/A'))
            writer.writerow(row)
            # row = [model]  # 如果模型没有某个角色的MBTI结果，则显示'N/A'
            # for char in characters:
            #     row.append(acc_results[model].get(char, 'N/A'))  # 如果模型没有某个角色的ACC结果，则显示'N/A'
            # writer.writerow(row)

    acc_csv_file_path = '16_results_acc_zh.csv'
    with open(acc_csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # 写入标题行：模型名 + 每个角色名*2（一次用于MBTI结果，一次用于ACC结果）
        headers = ['Model'] + [f'{char}' for char in mbti_types]
        writer.writerow(headers)
        
        # 逐行写入每个模型的结果
        for model in sorted(mbti_results.keys()):
            row = [model]  # 如果模型没有某个角色的MBTI结果，则显示'N/A'
            for char in mbti_types:
                row.append(acc_results[model].get(char, 'N/A'))  # 如果模型没有某个角色的ACC结果，则显示'N/A'
            writer.writerow(row)
    #Vlodemort: ENTJ
    #Beethoven: INTJ