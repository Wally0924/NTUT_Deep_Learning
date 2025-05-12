import json
import random

# 讀取原始 JSONL 檔案
input_file = '/home/rvl/mingwei/NTUT_Deep_Learning/tw_drone_pro_license_test_shuffled_en.jsonl'
with open(input_file, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

# 隨機抽取 10 題
sampled_data = random.sample(data, 10)

# 存成新的 JSON 檔案
output_file = 'sampled_10_questions.json'
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(sampled_data, f, ensure_ascii=False, indent=2)
