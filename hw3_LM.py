import os
import json
import re
import pandas as pd
from tqdm import tqdm
from collections import Counter
from openai import OpenAI # 導入 OpenAI 套件

# ROCm/AMD GPU專用環境變數 (如果需要，請保留，否則可以移除)
os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collection_threshold:0.9,max_split_size_mb:512"
# torch.ones(1).to('cuda') # 預熱GPU (如果需要 PyTorch，請保留，否則可以移除)
# os.environ["HF_TOKEN"] = "hf_YOUR_HUGGINGFACE_TOKEN" # 替換成你的 Hugging Face Token (如果需要，請保留，否則可以移除)

# === LM Studio API 設定 ===
LM_STUDIO_BASE_URL = "http://localhost:1234/v1" # LM Studio API 的基本 URL
LM_STUDIO_API_KEY = "lm-studio" # API 金鑰 (LM Studio 通常不需要，但 openai 套件需要，可任意填寫)

# === 模型名稱 (請務必替換成您在 LM Studio 中實際載入並運行的模型名稱) ===
# 例如："lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF"
# 或 "TheBloke/Mistral-7B-Instruct-v0.2-GGUF/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_NAME = "qwen_qwen3-32b" # 這是您提供的範例，請確保它與您LM Studio中的模型ID一致

# 創建 OpenAI 客戶端 (用於與 LM Studio API 互動)
client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_API_KEY)

DOMAIN_KEYWORDS = {
    "法規": [
        "regulation", "law", "penalty", "registration", "civil aviation", "CAA", "license", "permit", "certificate", "authority", "compliance", "insurance"
    ],
    "氣象": [
        "weather", "fog", "dew point", "cloud", "wind", "wind shear", "visibility", "front", "TAF", "METAR", "humidity", "temperature", "pressure", "thunderstorm", "icing", "hail"
    ],
    "物理": [
        "bernoulli", "lift", "damping", "frequency", "oscillation", "fluid", "moment", "force", "aerodynamics", "drag", "thrust", "torque", "center of gravity"
    ],
    "操作": [
        "attitude", "throttle", "emergency", "pre-flight", "signal link", "operation", "takeoff", "landing", "hover", "manual", "autonomous", "control", "routine check"
    ],
    "技術": [
        "stability", "amplitude", "navigation", "gyroscope", "sensor", "system", "multirotor", "helicopter", "UAV", "battery", "propeller", "engine", "payload"
    ]
}

PROMPT_TEMPLATES = {
    "法規": """You are a Taiwan CAA certified drone instructor and written exam expert. Answer the following single-choice question strictly and accurately according to the latest Taiwan Civil Aviation Act, CAA official guidelines, and written exam standards.
/no_think
**Rules for answering:**
- Do NOT guess or use general knowledge; only use information from the law, official rules, and exam materials.
- Read the question and all options carefully. Identify legal keywords (such as "registration", "penalty", "responsibility", "application", "authority", etc.).
- Eliminate any option that is clearly inconsistent with the law or official regulations.
- If "All of the above" or "None of the above" appears, carefully check each option for full correctness before choosing.
- If there is any doubt, always choose the answer that best fits the strictest safety and legal requirements.
- Output ONLY the final answer as a single uppercase letter (A/B/C/D). Do NOT explain, justify, or add any extra text.

Example:
Question: What is the validity period of a remote-controlled drone registration?
A) 1 year.
B) 2 years.
C) 3 years.
D) 5 years.
Answer: B

Please answer:
Question: {question}
A) {A}
B) {B}
C) {C}
D) {D}
Answer:
""",
    "物理": """You are a Taiwan CAA certified drone physics instructor and written exam expert. Answer the following single-choice question strictly and accurately according to physical laws, official study materials, and written exam standards.
/no_think
**Rules for answering:**
- Do NOT guess or use common sense; only use physics principles and official exam materials.
- Identify all physics keywords (such as "lift", "thrust", "frequency", "Bernoulli", etc.).
- Apply the correct physics formulas and principles to each option.
- Eliminate any answer that is physically impossible or contradicts the laws of physics.
- If "All of the above" or "None of the above" appears, check each option for accuracy before choosing.
- If uncertain, always choose the answer that is most consistent with strict physics and safety standards.
- Output ONLY the final answer as a single uppercase letter (A/B/C/D). Do NOT explain, justify, or add any extra text.

Example:
Question: What is the relationship between main rotor speed and lift during drone hovering?
A) Lift is proportional to the square of the rotor speed.
B) Lift is proportional to rotor speed.
C) No direct relationship.
D) It depends on air density.
Answer: A

Please answer:
Question: {question}
A) {A}
B) {B}
C) {C}
D) {D}
Answer:
""",
    "氣象": """You are a certified aviation meteorology instructor for the Taiwan drone license exam. Answer the following single-choice question strictly and accurately according to international aviation meteorology standards, Taiwan CWA guidelines, and official exam materials.
/no_think
**Rules for answering:**
- Do NOT guess or use general knowledge; only use official meteorology principles and exam materials.
- Identify all meteorological keywords (such as "wind shear", "cloud", "dew point", "visibility", etc.).
- Apply correct meteorological definitions and principles to each option.
- Eliminate any answer that contradicts official meteorological science.
- If "All of the above" or "None of the above" appears, check each option for accuracy before choosing.
- If uncertain, always choose the answer that is safest and most consistent with official guidelines.
- Output ONLY the final answer as a single uppercase letter (A/B/C/D). Do NOT explain, justify, or add any extra text.

Example:
Question: What is low-level wind shear?
A) Sudden change in wind direction or speed below 1600 feet.
B) Only occurs at high altitude.
C) Unrelated to temperature.
D) Only related to terrain.
Answer: A

Please answer:
Question: {question}
A) {A}
B) {B}
C) {C}
D) {D}
Answer:
""",
    "操作": """You are a certified drone operation instructor and written exam expert for Taiwan's drone license. Answer the following single-choice question strictly and accurately according to official standard operating procedures, safety principles, and written exam materials.
/no_think
**Rules for answering:**
- Do NOT guess or use general knowledge; only use official procedures and exam standards.
- Identify operational keywords (such as "pre-flight check", "emergency", "manual control", "autonomous", etc.).
- Eliminate any answer that violates safety or standard procedures.
- If "All of the above" or "None of the above" appears, check each option for accuracy before choosing.
- If uncertain, always choose the answer that is safest and most compliant with official procedures.
- Output ONLY the final answer as a single uppercase letter (A/B/C/D). Do NOT explain, justify, or add any extra text.

Example:
Question: When should a drone operator complete a pre-flight check?
A) Before every flight.
B) When abnormalities are found.
C) When weather changes.
D) When the operator feels something is wrong.
Answer: A

Please answer:
Question: {question}
A) {A}
B) {B}
C) {C}
D) {D}
Answer:
""",
    "默認": """You are a certified expert in the Taiwan drone license written exam. Answer the following single-choice question strictly and accurately according to the latest regulations, official study materials, and safety principles.
/no_think
**Rules for answering:**
- Do NOT guess or use common sense; only use official exam materials and standards.
- Identify important keywords in the question and options.
- Eliminate any answer that is unreasonable or contradicts official standards.
- If "All of the above" or "None of the above" appears, check each option for accuracy before choosing.
- If uncertain, always choose the answer that is safest and most compliant with official standards.
- Output ONLY the final answer as a single uppercase letter (A/B/C/D). Do NOT explain, justify, or add any extra text.

Example:
Question: When should a drone operator complete a pre-flight check?
A) Before every flight.
B) When abnormalities are found.
C) When weather changes.
D) When the operator feels something is wrong.
Answer: A

Please answer:
Question: {question}
A) {A}
B) {B}
C) {C}
D) {D}
Answer:
"""
}

def load_test_questions(file_path='tw_drone_pro_license_test_shuffled_en.jsonl'):
    questions = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            raw_data = json.loads(line)
            # 確保 prompt 鍵存在
            if "prompt" in raw_data:
                raw = raw_data["prompt"]
                # 預處理全形字符
                text = raw.replace("（","(").replace("）",")").replace("Ａ","A").replace("Ｂ","B").replace("Ｃ","C").replace("Ｄ","D")
                questions.append(text)
            else:
                print(f"警告: 找不到 'prompt' 鍵於行: {line.strip()}")
    return questions

def parse_question(question_text):
    cleaned = re.sub(r'\s+', ' ', question_text).strip()
    patterns = [
        r"^(.*?)\s+A[\)\.]\s*(.*?)\s+B[\)\.]\s*(.*?)\s+C[\)\.]\s*(.*?)\s+D[\)\.]\s*(.*?)(?:\s+Answer:\s*[A-D]?)?$",
        r"^(.*?)\nA[\)\.]\s*(.*?)\nB[\)\.]\s*(.*?)\nC[\)\.]\s*(.*?)\nD[\)\.]\s*(.*?)(?:\nAnswer:\s*[A-D]?)?$",
        r"^(.*?)\s+A\.\s*(.*?)\s+B\.\s*(.*?)\s+C\.\s*(.*?)\s+D\.\s*(.*?)(?:\s+Answer:\s*[A-D]?)?$"
    ]
    for pattern in patterns:
        match = re.match(pattern, cleaned, re.DOTALL)
        if match:
            return {
                "question": match.group(1).strip(),
                "options": {
                    'A': match.group(2).strip(),
                    'B': match.group(3).strip(),
                    'C': match.group(4).strip(),
                    'D': match.group(5).strip()
                }
            }
    parts = re.split(r'\s+[A-D][\)\.]\s*', cleaned)
    if len(parts) >= 5:
        return {
            "question": parts[0].strip(),
            "options": {
                'A': parts[1].strip(),
                'B': parts[2].strip(),
                'C': parts[3].strip(),
                'D': parts[4].strip()
            }
        }
    options = {}
    for idx, key in enumerate(['A', 'B', 'C', 'D']):
        try:
            # 嘗試從 parts 中獲取，如果 parts 不夠長，則標記為解析失敗
            if idx + 1 < len(parts):
                options[key] = parts[idx+1].strip()
            else: # 如果 parts 不夠長，則認為該選項缺失
                options[key] = f"[解析失敗{key}]"
        except Exception:
            options[key] = f"[解析失敗{key}]"

    # 如果 question_text 本身就沒有選項的模式，例如直接是 A) B) C) D) 開頭
    # 這時 parts[0] 可能會是空的，或者不是真正的題幹
    # 我們需要一個策略來處理這種情況，例如，如果 parts[0] 很短或空白，而 cleaned 本身包含選項標識
    if not parts[0].strip() and any(opt_marker in cleaned for opt_marker in ["A)", "B)", "C)", "D)"]):
         # 這種情況下，我們認定沒有題幹，將 cleaned 設為題幹，並標記所有選項解析失敗
        final_question = cleaned # 或者 "題幹缺失"
        final_options = {k: f"[題幹缺失或格式錯誤-{k}]" for k in ['A', 'B', 'C', 'D']}
    else:
        final_question = parts[0].strip() if parts else cleaned
        final_options = options

    # 如果題幹解析後仍然是空的，並且原始文本不為空，將原始文本作為題幹
    if not final_question and cleaned:
        final_question = cleaned # 或者標記為 "無法解析題幹"
        # 確保選項也被標記為可能的問題
        if all("[解析失敗" in v for v in final_options.values()): # 如果所有選項都已標記失敗
            pass # 維持原樣
        elif all(f"[題幹缺失或格式錯誤-{k}]" in v for k, v in final_options.items()):
            pass
        else: # 否則，也將選項標記為與題幹相關的問題
            final_options = {k: f"[題幹解析問題-{k}]" for k in ['A', 'B', 'C', 'D']}


    return {"question": final_question, "options": final_options}


def detect_domain(question, options):
    # 確保 options.values() 中的所有元素都是字符串
    option_texts = [str(opt) for opt in options.values()]
    text = (str(question) + ' ' + ' '.join(option_texts)).lower()
    domain_weights = {
        "法規": sum(3 for kw in DOMAIN_KEYWORDS["法規"] if kw in text),
        "物理": sum(2 for kw in DOMAIN_KEYWORDS["物理"] if kw in text),
        "氣象": sum(2 for kw in DOMAIN_KEYWORDS["氣象"] if kw in text),
        "操作": sum(1 for kw in DOMAIN_KEYWORDS["操作"] if kw in text)
    }
    # 處理 domain_weights 為空的情況
    if not domain_weights:
        return "默認"
        
    max_domain = max(domain_weights, key=domain_weights.get, default="默認")
    # 如果所有權重都是0，也返回 "默認"
    return max_domain if domain_weights.get(max_domain, 0) > 0 else "默認"


def build_advanced_prompt(question_data):
    options = question_data["options"]
    for key in ['A','B','C','D']:
        options[key] = str(options.get(key, f"[解析失敗{key}]")).replace('"','').strip() # 確保是字符串
    
    question_text = str(question_data.get("question", "")).replace('"', '') # 確保是字符串
    domain = detect_domain(question_text, options)
    
    return PROMPT_TEMPLATES[domain].format(
        question=question_text,
        **options
    )

def extract_answer(text):
    if not isinstance(text, str): # 確保輸入是字符串
        return "C" # 或者其他錯誤處理

    patterns = [
        r"Answer:\s*([A-D])",       # 優先匹配 "Answer: X"
        r"\b([A-D])\b(?!\s*:)",     # 匹配獨立的 A, B, C, D (不後面接冒號)
                                    # 例如，如果模型直接回答 "A"
        r"答案[\：:]\s*([A-D])",
        r"正確答案\s*([A-D])",
        r"\b([A-D])[\)\.]",         # 匹配 "A)", "A." 等
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    
    # 如果以上都沒有匹配到，嘗試找文本中第一個出現的 A, B, C, D 字母
    # 這是一個比較寬鬆的匹配，作為最後手段
    first_letter_match = re.search(r"([A-D])", text)
    if first_letter_match:
        return first_letter_match.group(1).upper()
        
    return "C" # 預設答案

def get_llm_response(prompt_text, model_name=MODEL_NAME):
    """
    向 LM Studio API 發送請求並獲取 LLM 的回應。
    Args:
        prompt_text (str): 要發送給 LLM 的完整提示文字 (包含角色設定、規則、範例和問題)。
        model_name (str): 要使用的模型名稱。
    Returns:
        str: LLM 的回應內容，如果出錯則返回 None。
    """
    try:
        # prompt_text 已經包含了所有指令、角色扮演和問題
        # 我們將這個完整的文本作為 'user' 消息的內容
        messages_for_llm = [
            {"role": "user", "content": "/no_think\n" + prompt_text}
        ]
        # 注意：某些模型可能對 'system' 消息有特定偏好。
        # 如果模型表現不佳，可以嘗試添加一個非常簡短且不衝突的 'system' 消息，
        # 例如：{"role": "system", "content": "You are an expert assistant answering multiple-choice questions."}
        # 但通常對於 instruct-tuned 模型，將所有內容放在 user message 中是可行的。

        completion = client.chat.completions.create(
            model=model_name,
            messages=messages_for_llm,
            temperature=0.0, # 保持確定性輸出以應對考試題目
            max_tokens=10,   # 預期答案為單個字母 (A, B, C, D)，加上少量可能的周圍文本
                             # 原本的 -1 (LM Studio 特有，表示無限制) 可能導致模型在未嚴格遵守指令時過於冗長
            stream=False # 非流式輸出
        )
        response_content = completion.choices[0].message.content
        return response_content
    except Exception as e:
        print(f"API 請求錯誤: {e}")
        return None

def main():
    print("🚀 使用 LM Studio API 進行推理...")
    print("📥 載入題庫...")
    questions = load_test_questions()
    if not questions:
        print("錯誤：題庫為空或載入失敗，程式終止。")
        return

    print(f"🔍 開始推理 {len(questions)} 題...")

    results = []
    failed_parse_questions = [] # 用於記錄解析本身失敗的原始題目文本

    batch_size = 1 # 對於 API 調用，通常逐條發送，除非 API 支持批處理
    print(f"🎯 批次大小（API調用時通常為1）：{batch_size}")

    question_id_counter = 0 # 手動維護題目ID

    for i in tqdm(range(0, len(questions), batch_size)):
        batch_raw_texts = questions[i:i+batch_size]
        
        current_batch_answers = []

        for raw_text in batch_raw_texts:
            question_data = parse_question(raw_text)
            
            # 檢查解析是否成功，特別是題幹是否有效
            if not question_data["question"] or "[解析失敗" in question_data["question"] or \
               all("[解析失敗" in opt_val for opt_val in question_data["options"].values()) or \
               all("[題幹缺失或格式錯誤" in opt_val for opt_val in question_data["options"].values()):
                print(f"\n警告: 題目 ID {question_id_counter} 解析失敗或題幹無效。原始文本: '{raw_text[:100]}...'")
                failed_parse_questions.append({"Id": question_id_counter, "OriginalQuestion": raw_text, "ParsedData": question_data})
                current_batch_answers.append("C") # 或其他錯誤標記
                results.append({"Id": question_id_counter, "Answer": "C"}) #記錄預設答案
                question_id_counter +=1
                continue # 跳過此題的 API 調用

            prompt_text = build_advanced_prompt(question_data)
            llm_response = get_llm_response(prompt_text)
            
            if llm_response:
                extracted_ans = extract_answer(llm_response)
                current_batch_answers.append(extracted_ans)
                results.append({"Id": question_id_counter, "Answer": extracted_ans})
            else:
                current_batch_answers.append("C") # API 出錯，使用預設答案
                results.append({"Id": question_id_counter, "Answer": "C"})
            
            question_id_counter += 1


    if not results:
        print("錯誤：沒有成功處理任何題目，無法生成答案文件。")
    else:
        pd.DataFrame(results)[["Id", "Answer"]].to_csv("gemma_27_answers.csv", index=False)
        print("✅ 完成！答案已儲存至 gemma_27_answers.csv")

        # 統計分佈
        counter = Counter([r["Answer"] for r in results])
        print("\n📊 答案分布統計:")
        for opt_char in ['A','B','C','D', 'E', 'F']: # 包含可能的錯誤/預設答案
            if counter.get(opt_char,0) > 0 :
                 print(f"{opt_char}: {counter.get(opt_char,0)}")


    if failed_parse_questions:
        print(f"\n⚠️ 題目解析失敗數量：{len(failed_parse_questions)}，已記錄於 failed_parse_questions.json")
        with open("failed_parse_questions.json", "w", encoding="utf-8") as f:
            json.dump(failed_parse_questions, f, ensure_ascii=False, indent=4) # indent=4 讓格式更易讀

if __name__ == "__main__":
    main()
