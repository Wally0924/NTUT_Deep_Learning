import pandas as pd

# 輸入你的兩份答案檔案路徑
file1 = "/home/rvl/mingwei/NTUT_Deep_Learning/prompt_only_answers.csv"
file2 = "/home/rvl/mingwei/NTUT_Deep_Learning/gemma_27_answers.csv"

# 讀取CSV
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# 以Id為主鍵合併
merged = pd.merge(df1, df2, on="Id", suffixes=('_1', '_2'))

# 比對答案
merged['is_same'] = merged['Answer_1'] == merged['Answer_2']

# 輸出不同的題號與答案
diff = merged[~merged['is_same']]

print(f"總題數：{len(merged)}")
print(f"答案完全一致的題數：{merged['is_same'].sum()}")
print(f"答案不同的題數：{len(diff)}")
print("\n不同的題目：")
print(diff[['Id', 'Answer_1', 'Answer_2']])

# 若要存成檔案
diff.to_csv("answer_diff.csv", index=False)
