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
# 或是您在 LM Studio 中下載並執行的任何模型的 ID
MODEL_NAME = "qwen_qwen3-32b" # 範例模型，請替換為您LM Studio中實際運行的模型ID

# 創建 OpenAI 客戶端 (用於與 LM Studio API 互動)
client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_API_KEY)

DOMAIN_KEYWORDS = {
    "法規": [
        "regulation", "law", "penalty", "registration", "civil aviation", "CAA", "license", "permit", "certificate", "authority", "compliance", "insurance", "act", "article", "enforcement", "provision", "rule", "guideline", "restriction", "operator", "owner", "liability"
    ],
    "氣象": [
        "weather", "fog", "dew point", "cloud", "wind", "wind shear", "visibility", "front", "TAF", "METAR", "humidity", "temperature", "pressure", "thunderstorm", "icing", "hail", "atmosphere", "altitude", "density", "forecast", "report", "convection", "inversion"
    ],
    "物理": [
        "bernoulli", "lift", "damping", "frequency", "oscillation", "fluid", "moment", "force", "aerodynamics", "drag", "thrust", "torque", "center of gravity", "newton's law", "inertia", "velocity", "acceleration", "mass", "density", "pressure", "energy", "power", "rotor", "propeller", " airfoil"
    ],
    "操作": [
        "attitude", "throttle", "emergency", "pre-flight", "signal link", "operation", "takeoff", "landing", "hover", "manual", "autonomous", "control", "routine check", "checklist", "procedure", "maneuver", "pilot", "operator", "risk", "safety", "payload", "battery", "navigation", "GPS", "visual line of sight"
    ],
    "技術": [ # 技術可能與操作、物理部分重疊，此處放更偏硬體/系統的詞
        "stability", "amplitude", "navigation", "gyroscope", "sensor", "system", "multirotor", "helicopter", "UAV", "battery", "propeller", "engine", "payload", "frequency band", "data link", "transmitter", "receiver", "ground control station", "GCS", "firmware", "software"
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
    "技術": """You are a certified drone technology and systems expert for Taiwan's drone license exam. Answer the following single-choice question strictly and accurately according to official documentation, engineering principles, and written exam materials related to drone hardware and software systems.
/no_think
**Rules for answering:**
- Do NOT guess or use general knowledge; only use established technical facts and official exam standards.
- Identify technical keywords (such as "sensor," "GPS," "battery," "multirotor," "frequency band," "data link," etc.).
- Evaluate options based on technical feasibility, system design, and common drone technologies.
- Eliminate any answer that is technically incorrect or impractical.
- If "All of the above" or "None of the above" appears, check each option for technical accuracy before choosing.
- If uncertain, always choose the answer that aligns best with standard drone technology and safety.
- Output ONLY the final answer as a single uppercase letter (A/B/C/D). Do NOT explain, justify, or add any extra text.

Example:
Question: Which frequency bands are commonly used for remote drone control and data transmission?
A) 300MHz or 500MHz.
B) 500MHz or 600MHz.
C) 2.4GHz or 5.8GHz.
D) 100GHz or 220GHz.
Answer: C

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
- If "All of ઉthe above" or "None of the above" appears, check each option for accuracy before choosing.
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
            try:
                raw_data = json.loads(line)
                if "prompt" in raw_data and isinstance(raw_data["prompt"], str):
                    # 預處理全形字符及常見標點統一
                    text = raw_data["prompt"].replace("（","(").replace("）",")")
                    text = text.replace("Ａ","A").replace("Ｂ","B").replace("Ｃ","C").replace("Ｄ","D")
                    text = text.replace("．",".").replace("：",":")
                    questions.append(text)
                else:
                    print(f"警告: 'prompt' 鍵缺失或格式不正確於行: {line.strip()}")
            except json.JSONDecodeError:
                print(f"警告: JSON 解析錯誤於行: {line.strip()}")
    return questions

def parse_question(question_text):
    cleaned = re.sub(r'\s+', ' ', question_text).strip()
    # 移除結尾可能存在的 "Answer: [A-D]" (英文題目中可能出現)
    cleaned = re.sub(r'\s*Answer:\s*[A-D]$', '', cleaned).strip()
    # 移除結尾可能存在的 "答案：[A-D]" (中文情境可能出現)
    cleaned = re.sub(r'\s*答案：[A-D]$', '', cleaned).strip()


    # 優先匹配包含換行符的模式，更常見於題目格式
    patterns = [
        # 模式1: Question...\nA) OptA\nB) OptB\nC) OptC\nD) OptD
        r"^(.*?)\s*\n\s*A[\)\.]\s*(.*?)\s*\n\s*B[\)\.]\s*(.*?)\s*\n\s*C[\)\.]\s*(.*?)\s*\n\s*D[\)\.]\s*(.*?)$",
        # 模式2: Question... A) OptA B) OptB C) OptC D) OptD (選項間可能無換行)
        r"^(.*?)\s+A[\)\.]\s*(.*?)\s+B[\)\.]\s*(.*?)\s+C[\)\.]\s*(.*?)\s+D[\)\.]\s*(.*?)$",
        # 模式3: 針對選項標點後可能緊跟非空格字符的情況
        r"^(.*?)\s*A[\)\.](.*?)\s*B[\)\.](.*?)\s*C[\)\.](.*?)\s*D[\)\.](.*?)$",
    ]

    for i, pattern_str in enumerate(patterns):
        match = re.match(pattern_str, cleaned, re.DOTALL) # re.DOTALL 讓 . 可以匹配換行符
        if match:
            q_text = match.group(1).strip()
            options_dict = {
                'A': match.group(2).strip(),
                'B': match.group(3).strip(),
                'C': match.group(4).strip(),
                'D': match.group(5).strip()
            }
            # 確保題幹和選項不為空
            if q_text and all(opt_text for opt_text in options_dict.values()):
                return {"question": q_text, "options": options_dict}

    # 若上述正規表達式均不匹配，使用基於分割的後備方案
    # 這個後備方案更通用，但可能不夠精確
    parts = re.split(r'\s+(?=[A-D][\)\.])', cleaned) # 在選項標示符前分割，保留標示符
    
    final_question = ""
    final_options = {k: f"[解析失敗-{k}]" for k in ['A', 'B', 'C', 'D']}

    if parts:
        final_question = parts[0].strip()
        current_opt_key = None
        option_texts = {}

        for part in parts[1:]:
            part_stripped = part.strip()
            match_opt = re.match(r'^([A-D])[\)\.]\s*(.*)$', part_stripped)
            if match_opt:
                opt_char = match_opt.group(1)
                opt_text = match_opt.group(2).strip()
                if opt_char in ['A', 'B', 'C', 'D']:
                    option_texts[opt_char] = opt_text
        
        # 檢查是否成功解析出 A, B, C, D 選項
        parsed_all_options = True
        for key in ['A', 'B', 'C', 'D']:
            if key in option_texts and option_texts[key]:
                final_options[key] = option_texts[key]
            else:
                final_options[key] = f"[選項{key}解析失敗]" # 更明確的失敗訊息
                parsed_all_options = False
        
        if not final_question or not parsed_all_options: # 如果題幹為空或選項不完整
             # 如果題幹看起來像選項的一部分，或者選項解析不完整，認定為整體解析失敗
            if any(final_question.startswith(f"{opt_char})" ) for opt_char in ['A','B','C','D']) or not parsed_all_options:
                # print(f"Fallback parse insufficient for: {cleaned[:100]}...")
                final_question = cleaned # 將整個文本視為問題，讓模型自行判斷
                final_options = {k: f"[格式特殊-{k}]" for k in ['A', 'B', 'C', 'D']} # 標記選項讓模板知道
            # else:
                # print(f"Fallback parse produced: Q='{final_question}', Opts='{final_options}'")


    if not final_question.strip(): # 如果最終題幹還是空的
        # print(f"Final fallback, treating whole text as question: {cleaned[:100]}...")
        final_question = cleaned # 最後手段，整個文本作為問題
        final_options = {k: f"[題幹選項難分-{k}]" for k in ['A', 'B', 'C', 'D']}

    return {"question": final_question, "options": final_options}


def detect_domain(question, options):
    # 確保 options.values() 中的所有元素都是字符串
    option_texts = [str(opt) for opt in options.values() if isinstance(opt, str)] # 過濾掉非字符串
    text_to_search = (str(question) + ' ' + ' '.join(option_texts)).lower()

    domain_scores = {domain: 0 for domain in DOMAIN_KEYWORDS}

    for domain, keywords in DOMAIN_KEYWORDS.items():
        score = 0
        for kw in keywords:
            if kw in text_to_search:
                if domain == "法規": score += 3 # 法規關鍵詞權重較高
                elif domain in ["物理", "氣象"]: score += 2 # 物理、氣象次之
                else: score += 1 # 其他（操作、技術）
        domain_scores[domain] = score
    
    # 如果所有分數都是0，返回 "默認"
    if all(score == 0 for score in domain_scores.values()):
        return "默認"

    # 找到最高分的領域
    # 如果有多個領域同分，可以依照預設優先級 (例如法規 > 物理 > 氣象 > 操作 > 技術)
    # 或簡單取第一個最高分的
    # sorted_domains = sorted(domain_scores.items(), key=lambda item: item[1], reverse=True)
    # max_domain = sorted_domains[0][0]
    
    # 為了更穩定的結果，如果最高分有多個，可以定義一個優先級列表
    priority_order = ["法規", "物理", "氣象", "操作", "技術", "默認"]
    
    max_score = 0
    if domain_scores: # 確保 domain_scores 不是空的
        max_score = max(domain_scores.values())

    if max_score == 0: # 如果最高分是0 (前面已處理，雙重保險)
        return "默認"

    best_domains = [domain for domain, score in domain_scores.items() if score == max_score]
    
    for domain_name in priority_order:
        if domain_name in best_domains:
            return domain_name
            
    return "默認" # 理論上不會執行到這裡，因為 priority_order 包含 "默認"


def build_advanced_prompt(question_data):
    options = question_data["options"]
    # 清理選項，確保是字符串並移除不必要的引號
    for key in ['A','B','C','D']:
        opt_val = options.get(key, f"[選項 {key} 缺失]")
        if not isinstance(opt_val, str):
            opt_val = str(opt_val)
        options[key] = opt_val.replace('"','').strip()

    question_text = str(question_data.get("question", "[題目缺失]")).replace('"', '')
    
    # 如果選項看起來像解析失敗的標記，可能表示題目本身格式有問題
    # 在這種情況下，將整個原始題目內容作為題幹傳給 "默認" 模板
    if all(f"[{tag}" in opt_val for opt_val in options.values() for tag in ["解析失敗", "格式特殊", "題幹選項難分", "選項缺失"]):
        domain = "默認"
        # 這種情況下，我們不填充選項到模板，讓模型直接從題幹(原始文本)中理解
        # 模板中的選項部分將會是空的或包含我們的標記
        # 為了讓模型有機會正確回答，我們把原始未解析的問題塞進去，讓選項部分為空
        # 這樣可以讓模型在某些情況下，如果它能理解原始的A/B/C/D格式，仍然能回答
        return PROMPT_TEMPLATES[domain].format(
            question=question_text, # question_text 此時可能是原始未分割的題目
            A=options.get('A',''), B=options.get('B',''), C=options.get('C',''), D=options.get('D','')
        )
    else:
        domain = detect_domain(question_text, options)
        return PROMPT_TEMPLATES[domain].format(
            question=question_text,
            A=options['A'], B=options['B'], C=options['C'], D=options['D']
        )


def extract_answer(text):
    if not isinstance(text, str): # 確保輸入是字符串
        return "C" # 預設或其他錯誤處理

    # 轉換為大寫，方便匹配
    text_upper = text.upper()

    # 優先級高的精確匹配
    # 1. "Answer: X" 或 "答案：X" (X為A-D)
    match = re.search(r"(?:ANSWER|答案)\s*[:：]\s*([A-D])", text_upper)
    if match:
        return match.group(1)

    # 2. 尋找單獨的字母 A, B, C, 或 D，作為一個詞 (word boundary)
    #    並且後面不能緊跟 ":" (避免匹配 "A:")
    #    前面可以是一些提示詞如 "THE CORRECT OPTION IS "
    match = re.search(r"\b([A-D])\b(?!\s*:)", text_upper)
    if match:
        # 檢查這個字母是否是句子中的主要答案指示
        # 例如，避免從 "OPTION A AND OPTION B ARE..." 中提取 "A"
        # 這比較困難，先信任 \b 邊界
        # 考慮模型可能直接輸出 "A" 或 "The answer is A."
        # 如果文本很短 (如少於5個字符)，且包含A-D，則可能是直接答案
        if len(text_upper.strip()) <= 5: # 例如 "A", "A.", "A\n"
             return match.group(1)
        # 檢查常見的答案句式
        if re.search(r"(?:IS|WAS|BE)\s+(?:THE\s+)?(?:CORRECT\s+)?(?:ANSWER\s+)?\b" + match.group(1) + r"\b", text_upper):
            return match.group(1)
        if re.search(r"\b" + match.group(1) + r"\b\s+IS\s+(?:THE\s+)?CORRECT", text_upper):
            return match.group(1)


    # 3. 匹配 "A)", "B.", "C)", etc.
    match = re.search(r"\b([A-D])[\)\.]", text_upper)
    if match:
        return match.group(1)

    # 4. 作為最後手段，尋找文本中第一個出現的 A, B, C, D 字母
    #    這比較寬鬆，適用於模型輸出非常不規範的情況
    first_letter_match = re.search(r"([A-D])", text_upper) # 不使用 \b，允許 "ANSWERA" -> A
    if first_letter_match:
        return first_letter_match.group(1)

    return "C" # 預設答案，如果都找不到


def get_llm_response(prompt_text, model_name=MODEL_NAME):
    """
    向 LM Studio API 發送請求並獲取 LLM 的回應。
    """
    try:
        messages_for_llm = [
            # 可以考慮加入一個非常簡短的 system prompt，但通常 instruct 模型在 user prompt 中表現良好
            # {"role": "system", "content": "You are an AI assistant specialized in answering multiple-choice questions from drone certification exams."},
            {"role": "user", "content": prompt_text}
        ]

        completion = client.chat.completions.create(
            model=model_name,
            messages=messages_for_llm,
            temperature=0.0, # 保持確定性輸出以應對考試題目
            max_tokens=15,   # 稍微增加一點 token 預算以防模型輸出如 "Answer: A" 或類似的短語
            stream=False
        )
        response_content = completion.choices[0].message.content
        return response_content
    except Exception as e:
        print(f"API 請求錯誤: {e}")
        # 可以考慮在這裡返回一個特定的錯誤標記，而不僅僅是None
        return f"API_ERROR: {str(e)}"


def main():
    print("🚀 使用 LM Studio API 進行推理...")
    print(f"💡 使用模型: {MODEL_NAME}")
    print("📥 載入題庫...")
    questions_raw_text = load_test_questions()

    if not questions_raw_text:
        print("錯誤：題庫為空或載入失敗，程式終止。")
        return

    print(f"🔍 開始推理 {len(questions_raw_text)} 題...")
    results = []
    failed_parse_log = [] # 用於記錄解析本身失敗的原始題目文本
    api_error_log = []   # 用於記錄API調用失敗的題目

    # 注意: 這裡的 question_id_counter 應該與 tw_drone_test.jsonl 中的 Id 對應 (如果有的話)
    # 如果 tw_drone_test.jsonl 沒有 Id, 我們就從 0 開始計數
    # 根據 output format 要求，Id 從 0 開始
    
    for idx, raw_text in enumerate(tqdm(questions_raw_text)):
        question_id = idx # CSV Id 從 0 開始
        
        question_data = parse_question(raw_text)

        # 檢查解析是否充分，題幹和選項是否有效
        # 如果題幹包含解析失敗的標記，或者所有選項都包含解析失敗的標記，則認為解析不佳
        is_parse_failed = "[解析失敗" in question_data["question"] or \
                          all("[解析失敗" in opt_val for opt_val in question_data["options"].values()) or \
                          all("[格式特殊" in opt_val for opt_val in question_data["options"].values()) or \
                          all("[題幹選項難分" in opt_val for opt_val in question_data["options"].values()) or \
                          not question_data["question"].strip() # 題幹為空

        if is_parse_failed:
            # print(f"\n警告: 題目 ID {question_id} 解析不佳。原始文本: '{raw_text[:100]}...'")
            # print(f"解析結果: Q: {question_data['question']}, Opts: {question_data['options']}")
            failed_parse_log.append({
                "Id": question_id, 
                "OriginalQuestion": raw_text, 
                "ParsedQuestion": question_data["question"],
                "ParsedOptions": question_data["options"]
            })
            # 即使解析不佳，也嘗試讓模型處理，因為 build_advanced_prompt 有針對此情況的處理
            # 但如果題幹為空，則直接給預設答案
            if not question_data["question"].strip():
                results.append({"Id": question_id, "Answer": "C"}) # 記錄預設答案
                continue


        prompt_text = build_advanced_prompt(question_data)
        # print(f"\nID: {question_id}\nGenerated Prompt:\n{prompt_text[:500]}...") # DEBUG: 打印部分提示

        llm_response_content = get_llm_response(prompt_text)

        if llm_response_content and llm_response_content.startswith("API_ERROR:"):
            print(f"\nAPI錯誤於題目 ID {question_id}: {llm_response_content}")
            api_error_log.append({"Id": question_id, "Error": llm_response_content, "Prompt": prompt_text})
            extracted_ans = "C" # API 出錯，使用預設答案
        elif llm_response_content:
            # print(f"LLM Raw Response for ID {question_id}: '{llm_response_content.strip()}'") # DEBUG
            extracted_ans = extract_answer(llm_response_content)
            # print(f"Extracted Answer for ID {question_id}: {extracted_ans}") # DEBUG
        else: # llm_response_content is None (理論上被 API_ERROR 情況覆蓋)
            print(f"\n無回應或未知API錯誤於題目 ID {question_id}")
            api_error_log.append({"Id": question_id, "Error": "No response or unknown API error", "Prompt": prompt_text})
            extracted_ans = "C" # 出錯，使用預設答案
            
        results.append({"Id": question_id, "Answer": extracted_ans})

    if not results:
        print("錯誤：沒有成功處理任何題目，無法生成答案文件。")
    else:
        # 確保 Id 是整數且排序正確
        df_results = pd.DataFrame(results)
        df_results["Id"] = df_results["Id"].astype(int)
        df_results = df_results.sort_values(by="Id").reset_index(drop=True)

        # 輸出檔名格式根據競賽要求，例如 MODEL_NAME_1_answer.csv
        # 這裡的 "1" 可能是版本號或特定標識符，先寫死為1
        output_filename = f"{MODEL_NAME}_2_answer.csv"
        df_results[["Id", "Answer"]].to_csv(output_filename, index=False)
        print(f"✅ 完成！答案已儲存至 {output_filename}")

    # 統計分佈
    counter = Counter([r["Answer"] for r in results])
    print("\n📊 答案分布統計:")
    # 確保 A, B, C, D 都有打印，即使數量為0
    for opt_char in ['A','B','C','D']:
        print(f"{opt_char}: {counter.get(opt_char,0)}")
    # 打印其他可能的非標準答案（例如錯誤提取或預設值）
    other_answers = {k:v for k,v in counter.items() if k not in ['A','B','C','D']}
    if other_answers:
        print("其他答案:")
        for ans, num in other_answers.items():
            print(f"  {ans}: {num}")


    if failed_parse_log:
        print(f"\n⚠️ 題目解析不佳數量：{len(failed_parse_log)}，詳情已記錄於 failed_parse_log.json")
        with open("failed_parse_log.json", "w", encoding="utf-8") as f:
            json.dump(failed_parse_log, f, ensure_ascii=False, indent=4)
    
    if api_error_log:
        print(f"\n🔥 API調用失敗數量：{len(api_error_log)}，詳情已記錄於 api_error_log.json")
        with open("api_error_log.json", "w", encoding="utf-8") as f:
            json.dump(api_error_log, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
