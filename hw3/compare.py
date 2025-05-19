import pandas as pd
import json

# --- 設定檔案路徑 ---
correct_answers_file = "/home/rvl/mingwei/NTUT_Deep_Learning/wen_correct.csv"  # 標準答案檔案
your_answers_file = "/home/rvl/mingwei/NTUT_Deep_Learning/qwen_qwen3-32b_rag_answers.csv" # 你的模型答案檔案
questions_file = "/home/rvl/mingwei/NTUT_Deep_Learning/tw_drone_pro_license_test_shuffled_en.jsonl" # 原始題庫檔案
output_diff_file = "answer_differences_with_questions.csv" # 差異結果輸出檔案
output_diff_txt_file = "answer_differences_with_questions.txt" # 差異結果純文字輸出檔案

# --- 步驟 1: 讀取原始題庫的題目文本 ---
def load_question_texts(jsonl_file_path):
    """
    從 JSONL 檔案載入問題文本。
    JSONL 行號從 1 開始，將其映射到從 0 開始的 Id。
    """
    question_map = {}
    try:
        with open(jsonl_file_path, "r", encoding="utf-8") as f:
            # enumerate(f, 1) 使 line_number 從 1 開始，符合 JSONL 的自然行號
            for line_number, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    # 假設題目文本在 'prompt' 鍵中
                    # 將 JSONL 的行號 (line_number, 從1開始) 轉換為 Id (從0開始)
                    question_id = line_number - 1
                    question_map[question_id] = data.get("prompt", f"題目文本未找到於 JSONL 第 {line_number} 行").strip()
                except json.JSONDecodeError:
                    question_id = line_number - 1
                    question_map[question_id] = f"錯誤：JSONL 第 {line_number} 行 JSON 解析失敗"
                    print(f"警告：JSONL 第 {line_number} 行 JSON 解析失敗於題庫檔案 '{jsonl_file_path}'")
    except FileNotFoundError:
        print(f"錯誤：題庫檔案 '{jsonl_file_path}' 未找到。將無法顯示題目內容。")
        return None
    print(f"題庫檔案 '{jsonl_file_path}' 已載入，共 {len(question_map)} 條題目。Id 範圍從 0 到 {len(question_map)-1}。")
    return question_map

question_texts_map = load_question_texts(questions_file)

# --- 步驟 2: 讀取並合併答案 CSV ---
try:
    df_correct = pd.read_csv(correct_answers_file)
    df_your = pd.read_csv(your_answers_file)
except FileNotFoundError as e:
    print(f"錯誤：無法讀取答案檔案：{e}")
    exit()

# 確保 'Id' 欄位存在且類型一致
if 'Id' not in df_correct.columns or 'Id' not in df_your.columns:
    print("錯誤：兩個答案檔案中都必須包含 'Id' 欄位。")
    exit()
if 'Answer' not in df_correct.columns:
    print(f"錯誤：標準答案檔案 '{correct_answers_file}' 中缺少 'Answer' 欄位。請將答案欄位命名為 'Answer'。")
    exit()
if 'Answer' not in df_your.columns:
    print(f"錯誤：您的答案檔案 '{your_answers_file}' 中缺少 'Answer' 欄位。請將答案欄位命名為 'Answer'。")
    exit()

# 為了避免與原始欄位名衝突，在合併前重命名答案欄位
df_correct = df_correct.rename(columns={'Answer': 'Correct_Answer'})
df_your = df_your.rename(columns={'Answer': 'Your_Answer'})

# 以Id為主鍵合併
# 確保 Id 是整數類型以進行正確合併
try:
    df_correct['Id'] = df_correct['Id'].astype(int)
    df_your['Id'] = df_your['Id'].astype(int)
except ValueError as e:
    print(f"錯誤：'Id' 欄位轉換為整數失敗：{e}。請檢查 'Id' 欄位是否都為有效的數字。")
    exit()

merged_df = pd.merge(df_correct[['Id', 'Correct_Answer']], df_your[['Id', 'Your_Answer']], on="Id", how="inner")

if merged_df.empty:
    print("錯誤：合併後的 DataFrame 為空。請檢查 'Id' 是否匹配或檔案是否正確。")
    print(f"  標準答案 Id 範圍: {df_correct['Id'].min()}-{df_correct['Id'].max()}, 數量: {len(df_correct)}")
    print(f"  你的答案 Id 範圍: {df_your['Id'].min()}-{df_your['Id'].max()}, 數量: {len(df_your)}")
    exit()

# --- 步驟 3: 比對答案 ---
# 確保答案欄位是字符串類型再進行比較，避免因類型不同導致的錯誤
merged_df['is_same'] = merged_df['Correct_Answer'].astype(str).str.strip().str.upper() == \
                       merged_df['Your_Answer'].astype(str).str.strip().str.upper()

# --- 步驟 4: 提取並顯示不同的題目 ---
diff_df = merged_df[~merged_df['is_same']].copy() # 使用 .copy() 避免 SettingWithCopyWarning

# 如果成功載入題庫，則加入題目文本
if question_texts_map:
    diff_df['Question_Text'] = diff_df['Id'].map(question_texts_map).fillna("題目文本未找到或Id不匹配")
    columns_to_display_in_csv = ['Id', 'Question_Text', 'Correct_Answer', 'Your_Answer']
    columns_to_print = ['Id', 'Question_Text', 'Correct_Answer', 'Your_Answer']
else:
    columns_to_display_in_csv = ['Id', 'Correct_Answer', 'Your_Answer']
    columns_to_print = ['Id', 'Correct_Answer', 'Your_Answer']


# --- 步驟 5: 輸出統計與差異結果 ---
total_questions = len(merged_df)
correct_count = merged_df['is_same'].sum()
diff_count = len(diff_df)
accuracy = (correct_count / total_questions) * 100 if total_questions > 0 else 0

print("--- 答案比對結果 ---")
print(f"標準答案檔案：{correct_answers_file}")
print(f"您的答案檔案：{your_answers_file}")
print(f"原始題庫檔案：{questions_file if question_texts_map else '未載入或載入失敗'}")
print("----------------------")
print(f"總比對題數：{total_questions}")
print(f"答案完全一致的題數：{correct_count}")
print(f"答案不同的題數：{diff_count}")
print(f"準確率：{accuracy:.2f}%")
print("----------------------")

if diff_count > 0:
    print("\n以下是答案不同的題目詳情：")
    # 為了更好的終端輸出格式，逐條打印
    # 也保存到純文字檔案
    try:
        with open(output_diff_txt_file, "w", encoding="utf-8") as f_txt:
            f_txt.write("--- 答案不同的題目詳情 ---\n")
            f_txt.write(f"標準答案檔案：{correct_answers_file}\n")
            f_txt.write(f"您的答案檔案：{your_answers_file}\n")
            f_txt.write(f"原始題庫檔案：{questions_file if question_texts_map else '未載入或載入失敗'}\n")
            f_txt.write("--------------------------\n")
            f_txt.write(f"總比對題數：{total_questions}\n")
            f_txt.write(f"答案完全一致的題數：{correct_count}\n")
            f_txt.write(f"答案不同的題數：{diff_count}\n")
            f_txt.write(f"準確率：{accuracy:.2f}%\n")
            f_txt.write("--------------------------\n\n")

            for index, row in diff_df[columns_to_print].iterrows():
                print(f"\n題目 Id: {row['Id']}")
                f_txt.write(f"題目 Id: {row['Id']}\n")
                if 'Question_Text' in row and pd.notna(row['Question_Text']):
                    # 簡單清理題目文本中的換行符，使其在終端更易讀
                    q_text_display = row['Question_Text'].replace('\n', ' ').replace('\r', '')
                    print(f"  題目內容 (預覽): {q_text_display[:200]}...") # 顯示前200個字符預覽
                    f_txt.write(f"  題目內容:\n{row['Question_Text']}\n") # TXT 檔案保存完整題目
                elif 'Question_Text' in row: # 即使是 "題目文本未找到..." 也要打印
                    print(f"  題目內容: {row['Question_Text']}")
                    f_txt.write(f"  題目內容: {row['Question_Text']}\n")

                print(f"  標準答案: {row['Correct_Answer']}")
                print(f"  你的答案: {row['Your_Answer']}")
                f_txt.write(f"  標準答案: {row['Correct_Answer']}\n")
                f_txt.write(f"  你的答案: {row['Your_Answer']}\n\n")
        print(f"\n詳細的差異也已儲存到純文字檔案：{output_diff_txt_file}")
    except Exception as e:
        print(f"\n錯誤：寫入純文字差異檔案失敗：{e}")


    # 將差異結果儲存到 CSV 檔案
    try:
        diff_df[columns_to_display_in_csv].to_csv(output_diff_file, index=False, encoding="utf-8-sig")
        print(f"\n詳細的差異已儲存到 CSV 檔案：{output_diff_file}")
    except Exception as e:
        print(f"\n錯誤：儲存 CSV 差異檔案失敗：{e}")
else:
    print("\n太棒了！所有題目的答案都一致！")

print("--- 比對完成 ---")
