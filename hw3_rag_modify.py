import os
import json
import re
import torch
import pandas as pd
from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM

# ROCm/AMD GPU專用環境變數
os.environ["TORCH_BLAS_PREFER_HIPBLASLT"] = "0"
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"
os.environ["PYTORCH_HIP_ALLOC_CONF"] = "garbage_collection_threshold:0.9,max_split_size_mb:512"
torch.ones(1).to('cuda')  # 預熱GPU
os.environ["HF_TOKEN"] = "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"

MODEL_NAME = "google/gemma-3-4b-it"

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
"""
,

"物理": """You are a Taiwan CAA certified drone physics instructor and written exam expert. Answer the following single-choice question strictly and accurately according to physical laws, official study materials, and written exam standards.

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
"""
,

"氣象": """You are a certified aviation meteorology instructor for the Taiwan drone license exam. Answer the following single-choice question strictly and accurately according to international aviation meteorology standards, Taiwan CWA guidelines, and official exam materials.

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
"""
,

"操作": """You are a certified drone operation instructor and written exam expert for Taiwan's drone license. Answer the following single-choice question strictly and accurately according to official standard operating procedures, safety principles, and written exam materials.

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
"""
,

 "默認": """You are a certified expert in the Taiwan drone license written exam. Answer the following single-choice question strictly and accurately according to the latest regulations, official study materials, and safety principles.

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
            raw = json.loads(line)["prompt"]
            # 預處理全形字符
            text = raw.replace("（","(").replace("）",")").replace("Ａ","A").replace("Ｂ","B").replace("Ｃ","C").replace("Ｄ","D")
            questions.append(text)
    return questions

def parse_question(question_text):
    """
    強化多格式支援與自動補齊選項，確保每題都能產生四個選項
    """
    cleaned = re.sub(r'\s+', ' ', question_text).strip()
    # 支援多種常見格式
    patterns = [
        r"^(.*?)\s+A[\)\.]\s*(.*?)\s+B[\)\.]\s*(.*?)\s+C[\)\.]\s*(.*?)\s+D[\)\.]\s*(.*?)$",
        r"^(.*?)\nA[\)\.]\s*(.*?)\nB[\)\.]\s*(.*?)\nC[\)\.]\s*(.*?)\nD[\)\.]\s*(.*?)$",
        r"^(.*?)\s+A\.\s*(.*?)\s+B\.\s*(.*?)\s+C\.\s*(.*?)\s+D\.\s*(.*?)$"
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
    # 萬一沒 match，嘗試用分割法
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
    # 若仍失敗，自動補齊缺失選項
    options = {}
    for idx, key in enumerate(['A', 'B', 'C', 'D']):
        try:
            options[key] = parts[idx+1].strip()
        except Exception:
            options[key] = f"[解析失敗{key}]"
    return {"question": cleaned, "options": options}

def detect_domain(question, options):
    text = (question + ' ' + ' '.join(options.values())).lower()
    domain_weights = {
        "法規": sum(3 for kw in DOMAIN_KEYWORDS["法規"] if kw in text),
        "物理": sum(2 for kw in DOMAIN_KEYWORDS["物理"] if kw in text),
        "氣象": sum(2 for kw in DOMAIN_KEYWORDS["氣象"] if kw in text),
        "操作": sum(1 for kw in DOMAIN_KEYWORDS["操作"] if kw in text)
    }
    max_domain = max(domain_weights, key=domain_weights.get)
    return max_domain if domain_weights[max_domain] > 2 else "默認"

def build_advanced_prompt(question_data):
    options = question_data["options"]
    # 自動補齊缺失選項
    for key in ['A','B','C','D']:
        options[key] = options.get(key, f"[解析失敗{key}]").replace('"','').strip()
    domain = detect_domain(question_data["question"], options)
    return PROMPT_TEMPLATES[domain].format(
        question=question_data["question"].replace('"',''),
        **options
    )

def initialize_model():
    print(f"偵測到AMD GPU：{torch.cuda.get_device_name(0)}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"
    )
    return model, tokenizer

def extract_answer(text):
    """
    強化答案抽取規則，優先找獨立大寫字母，減少預設為C的情形
    """
    patterns = [
        r"答案[\：:]\s*([A-D])",
        r"\b([A-D])\b(?=\s*$)",
        r"正確答案\s*([A-D])",
        r"Answer[\：:]\s*([A-D])",
        r"([A-D])[\)\.]?\s*$"
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    # 最後手段：找最早出現的單一大寫字母A-D
    match = re.search(r"\b([A-D])\b", text)
    if match:
        return match.group(1).upper()
    return "C"

def main():
    print("🚀 初始化模型中...")
    model, tokenizer = initialize_model()
    print("📥 載入題庫...")
    questions = load_test_questions()
    print("🔍 開始推理...")

    results = []
    failed_questions = []
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    batch_size = max(1, int(total_mem // 3.5))
    print(f"💻 總顯存：{total_mem:.1f}GB | 🎯 批次大小：{batch_size}")

    idx = 0
    for i in tqdm(range(0, len(questions), batch_size)):
        batch = questions[i:i+batch_size]
        batch_data = [parse_question(q) for q in batch]
        # 確保每題都產生 prompt
        prompts = [build_advanced_prompt(d) for d in batch_data]
        inputs = tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        ).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=5,
                do_sample=False,
                top_k=1,
                top_p=0.7,
                temperature=0.1,
                pad_token_id=tokenizer.eos_token_id
            )
        answers = [extract_answer(tokenizer.decode(o, skip_special_tokens=True)) for o in outputs]
        for j, ans in enumerate(answers):
            # 若有選項明顯解析失敗，記錄下來
            if "[解析失敗" in str(batch_data[j]["options"].values()):
                failed_questions.append({"Id": idx, "Question": batch[j]})
            results.append({"Id": idx, "Answer": ans})
            idx += 1

    pd.DataFrame(results)[["Id", "Answer"]].to_csv("prompt_only_answers.csv", index=False)
    print("✅ 完成！答案已儲存至 prompt_only_answers.csv")

    # 統計分佈
    counter = Counter([r["Answer"] for r in results])
    print(f"\n答案數量:{len(counter)}")
    print("\n📊 答案分布統計:")
    for opt in ['A','B','C','D']:
        print(f"{opt}: {counter.get(opt,0)}")

    # 額外輸出解析失敗題目
    if failed_questions:
        print(f"\n⚠️ 解析失敗題目數量：{len(failed_questions)}，已記錄於 failed_questions.json")
        with open("failed_questions.json", "w", encoding="utf-8") as f:
            json.dump(failed_questions, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
