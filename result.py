# import csv
# import os
# import json
# import csv
# def read_csv_to_dict(file_path):
#     """
#     读取CSV文件到字典。
#     """
#     if not os.path.exists(file_path):
#         return {}
#     with open(file_path, mode='r', encoding='utf-8') as file:
#         reader = csv.DictReader(file)
#         return {row["Model"]: row for row in reader}

# def write_dict_to_csv(file_path, data_dict, headers):
#     """
#     将字典写回CSV文件。
#     """
#     with open(file_path, mode='w', newline='', encoding='utf-8') as file:
#         writer = csv.DictWriter(file, fieldnames=headers)
#         writer.writeheader()
#         for row in data_dict.values():
#             writer.writerow(row)

# def update_or_append_results(file_path, new_data, characters):
#     """
#     更新或追加新的MBTI和ACC结果到CSV文件。
#     """
#     data_dict = read_csv_to_dict(file_path)
    
#     # 更新或追加数据
#     for model, results in new_data.items():
#         if model in data_dict:
#             # 更新现有模型的数据
#             for char, res in results["MBTI"].items():
#                 data_dict[model][f'{char}_MBTI'] = res
#             for char, acc in results["ACC"].items():
#                 data_dict[model][f'{char}_ACC'] = acc
#         else:
#             # 添加新模型的数据
#             data_dict[model] = {"Model": model}
#             for char in characters:
#                 data_dict[model][f'{char}_MBTI'] = results["MBTI"].get(char, 'N/A')
#                 data_dict[model][f'{char}_ACC'] = results["ACC"].get(char, 'N/A')
    
#     # 确保列标题包括所有可能的角色
#     headers = ['Model'] + [f'{char}_MBTI' for char in characters] + [f'{char}_ACC' for char in characters]
#     write_dict_to_csv(file_path, data_dict, headers)

# # 示例数据
# characters = characters = [
#     "Abraham Lincoln",
#     "Albert Einstein",
#     "Bilbo Baggins",
#     "Charlize Theron",
#     "Clarice Starling",
#     "Dr. Ana Stelline",
#     "Forrest Gump",
#     "Frodo Baggins",
#     "Harry Potter",
#     "Joe Biden",
#     "Logan",
#     "Melanie Hamilton",
#     "Michael Jordan",
#     "Natasha Romanoff",
#     "Officer K",
#     "Peter Parker",
#     "Rhett Butler",
#     "Shaquille O'Neal",
#     "Steve Rogers",
#     "Tony Stark",
#     "Wade Wilson"
# ]  # 需要包括所有可能的角色名

# # 指定的CSV文件路径
# file_path = 'pop20_results.csv'
# llms_mbti = json.load(open('pop20_mbti.json', 'r', encoding='utf8'))
# # 更新或追加数据到CSV
# update_or_append_results(file_path, llms_mbti, characters)


import json
import csv


# 加载数据
llms_mbti = json.load(open('16_mbti_en.json', 'r', encoding='utf8'))

# 初始化结果字典
mbti_results = {}
acc_results = {}
characters = set()

# 分解键值对并填充结果
for k, v in llms_mbti.items():
    model, character = k.split('_')
    characters.add(character)  # 收集所有独特的角色名称
    if model not in mbti_results:
        mbti_results[model] = {}
    if model not in acc_results:
        acc_results[model] = {}
    mbti_results[model][character] = v['res']
    acc_results[model][character] = v['pred']

# 确保字符名的顺序
characters = sorted(list(characters))

# 打印每个模型的结果，先MBTI，再ACC
for model in sorted(mbti_results.keys()):  # 模型名按字典序排序
    mbti_row = [model]  # 第一列是模型名称
    acc_row = [model]
    for character in characters:
        mbti_row.append(mbti_results[model].get(character, 'N/A'))  # 如果模型没有某个角色的结果，则显示'N/A'
        acc_row.append(acc_results[model].get(character, 'N/A'))
    print(mbti_row)  # MBTI结果
    print(acc_row)   # ACC结果


# csv_file_path = 'pop20_results.csv'
# with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file)
    
#     # 写入标题行：模型名 + 每个角色名*2（一次用于MBTI结果，一次用于ACC结果）
#     headers = ['Model'] + [f'{char}_MBTI' for char in characters] + [f'{char}_ACC' for char in characters]
#     writer.writerow(headers)
    
#     # 逐行写入每个模型的结果
#     for model in sorted(mbti_results.keys()):
#         row = [model]  # 第一列是模型名称
#         for char in characters:
#             row.append(mbti_results[model].get(char, 'N/A'))  # 如果模型没有某个角色的MBTI结果，则显示'N/A'
#         for char in characters:
#             row.append(acc_results[model].get(char, 'N/A'))  # 如果模型没有某个角色的ACC结果，则显示'N/A'
#         writer.writerow(row)
