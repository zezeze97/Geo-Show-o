import json

# JSON文件路径
json_file_path = '/lustre/home/2201210053/Geo-Show-o/data/formalgeo7k/formalgeo7k_v2/custom_json/t2i/test.json'

# 目标文本文件路径
txt_file_path = 'output.txt'

# 打开JSON文件并加载内容
with open(json_file_path, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# 打开文本文件以写入数据
with open(txt_file_path, 'w', encoding='utf-8') as txt_file:
    for entry in data:
        for conversation in entry['conversations']:
            if 'value' in conversation and conversation['value']:  # 只提取非空的'value'
                txt_file.write(conversation['value'] + '\n')

print("提取完成，已保存至 'output.txt'")
