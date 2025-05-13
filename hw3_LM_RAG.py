import os
import json
import re
import pandas as pd
from tqdm import tqdm
from collections import Counter
from openai import OpenAI
from pathlib import Path

# Langchain RAG related imports
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# === RAG Settings ===
DOCUMENTS_PATHS = [
    Path(__file__).parent / "1131114.txt",
    Path(__file__).parent / "Taiwan_UAV_Regulations_20241114.txt"
]
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # Lightweight embedding model
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
TOP_K_DOCS = 3  # Retrieve the most relevant document chunks

# File paths
CORRECT_ANSWERS_FILE = "wen_correct.csv"  # Path to the correct answers CSV file

# Global variables for RAG pipeline
vector_store = None
retriever = None

# === LM Studio API Settings ===
LM_STUDIO_BASE_URL = "http://localhost:1234/v1"
LM_STUDIO_API_KEY = "lm-studio"  # Can be any value

# === Model name (please replace with the actual model name loaded in your LM Studio) ===
MODEL_NAME = "qwen_qwen3-32b"  # Make sure this matches your model ID in LM Studio

client = OpenAI(base_url=LM_STUDIO_BASE_URL, api_key=LM_STUDIO_API_KEY)

DOMAIN_KEYWORDS = {
    "Regulations": ["regulation", "law", "penalty", "registration", "civil aviation", "CAA", "license", "permit",
             "certificate", "authority", "compliance", "insurance", "article", "act", "rule", "regulation"],
    "Meteorology": ["weather", "fog", "dew point", "cloud", "wind", "wind shear", "visibility", "front", "TAF", "METAR",
             "humidity", "temperature", "pressure", "thunderstorm", "icing", "hail", "aviation meteorology"],
    "Physics": ["bernoulli", "lift", "damping", "frequency", "oscillation", "fluid", "moment", "force", "aerodynamics",
             "drag", "thrust", "torque", "center of gravity", "physics"],
    "Operations": ["attitude", "throttle", "emergency", "pre-flight", "signal link", "operation", "takeoff", "landing",
             "hover", "manual", "autonomous", "control", "routine check", "procedure", "aviation safety"],
    "Technology": ["stability", "amplitude", "navigation", "gyroscope", "sensor", "system", "multirotor", "helicopter", "UAV",
             "battery", "propeller", "engine", "payload", "telemetry", "electronics"]
}

PROMPT_TEMPLATES = {
    "Regulations": """You are an expert on Taiwan CAA drone regulations. Your ONLY task is to choose the single correct answer (A, B, C, or D) for the question, using the provided context if helpful. Output ONLY the letter.

/no_think

**Instructions:**
1.  Read the question CAREFULLY.
2.  If the **Retrieved Context** below helps, use it to answer.
3.  If the context is missing or unhelpful, use your expert knowledge of Taiwan drone regulations.
4.  Choose ONE answer.
5.  Answer with ONLY the uppercase letter (A, B, C, or D). Nothing else!

**Retrieved Context:**
{retrieved_context}

**Question:**
{question}
A) {A}
B) {B}
C) {C}
D) {D}

Your Answer (A, B, C, or D):""",

    "Physics": """You are an expert on drone physics. Your ONLY task is to choose the single correct answer (A, B, C, or D) for the question, using the provided context if helpful. Output ONLY the letter.

/no_think

**Instructions:**
1. Read the question CAREFULLY.
2. If the **Retrieved Context** below helps, use it to answer.
3. If the context is missing or unhelpful, use your expert knowledge of drone physics.
4. Choose ONE answer.
5. Answer with ONLY the uppercase letter (A, B, C, or D). Nothing else!

**Retrieved Context:**
{retrieved_context}

**Question:**
{question}
A) {A}
B) {B}
C) {C}
D) {D}

Your Answer (A, B, C, or D):""",

    "Meteorology": """You are an expert on aviation meteorology. Your ONLY task is to choose the single correct answer (A, B, C, or D) for the question, using the provided context if helpful. Output ONLY the letter.

/no_think

**Instructions:**
1. Read the question CAREFULLY.
2. If the **Retrieved Context** below helps, use it to answer.
3. If the context is missing or unhelpful, use your expert knowledge of aviation meteorology.
4. Choose ONE answer.
5. Answer with ONLY the uppercase letter (A, B, C, or D). Nothing else!

**Retrieved Context:**
{retrieved_context}

**Question:**
{question}
A) {A}
B) {B}
C) {C}
D) {D}

Your Answer (A, B, C, or D):""",

    "Operations": """You are an expert on drone operation. Your ONLY task is to choose the single correct answer (A, B, C, or D) for the question, using the provided context if helpful. Output ONLY the letter.

/no_think

**Instructions:**
1. Read the question CAREFULLY.
2. If the **Retrieved Context** below helps, use it to answer.
3. If the context is missing or unhelpful, use your expert knowledge of drone operation.
4. Choose ONE answer.
5. Answer with ONLY the uppercase letter (A, B, C, or D). Nothing else!

**Retrieved Context:**
{retrieved_context}

**Question:**
{question}
A) {A}
B) {B}
C) {C}
D) {D}

Your Answer (A, B, C, or D):""",

    "Default": """You are an expert on drone knowledge. Your ONLY task is to choose the single correct answer (A, B, C, or D) for the question, using the provided context if helpful. Output ONLY the letter.

/no_think

**Instructions:**
1. Read the question CAREFULLY.
2. If the **Retrieved Context** below helps, use it to answer.
3. If the context is missing or unhelpful, use your expert knowledge.
4. Choose ONE answer.
5. Answer with ONLY the uppercase letter (A, B, C, or D). Nothing else!

**Retrieved Context:**
{retrieved_context}

**Question:**
{question}
A) {A}
B) {B}
C) {C}
D) {D}

Your Answer (A, B, C, or D):"""
}


def setup_rag_pipeline():
    global vector_store, retriever

    print("📚 Setting up RAG pipeline...")
    docs = []
    for doc_path in DOCUMENTS_PATHS:
        if doc_path.exists():
            try:
                loader = TextLoader(str(doc_path), encoding='utf-8')
                docs.extend(loader.load())
                print(f"📄 Loaded document: {doc_path.name}")
            except Exception as e:
                print(f"⚠️ Failed to load document {doc_path.name}: {e}")
        else:
            print(f"⚠️ Document not found: {doc_path}")

    if not docs:
        print("❌ RAG document loading failed or no documents provided, RAG functionality will be limited.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    splits = text_splitter.split_documents(docs)
    print(f"🧩 Documents split into {len(splits)} chunks.")

    print(f"🧠 Initializing embedding model: {EMBEDDING_MODEL_NAME}...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    except Exception as e:
        print(f"❌ Failed to initialize embedding model: {e}. Please ensure 'sentence-transformers' is installed and the model name is correct.")
        print(" Try running: pip install sentence-transformers")
        return

    print(f"💾 Creating vector database (FAISS)...")
    try:
        vector_store = FAISS.from_documents(splits, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K_DOCS})
        print("✅ RAG pipeline setup complete!")
    except Exception as e:
        print(f"❌ Failed to create vector database: {e}. Please ensure 'faiss-cpu' (or 'faiss-gpu') is installed.")
        print(" Try running: pip install faiss-cpu")


def get_retrieved_context(query_text):
    if retriever:
        try:
            relevant_docs = retriever.invoke(query_text)
            if relevant_docs:
                context_parts = [f"--- Document Snippet {i + 1} ---\n{doc.page_content}\n" for i, doc in
                                 enumerate(relevant_docs)]
                return "\n".join(context_parts)
            else:
                return "No relevant context found in the provided documents."
        except Exception as e:
            print(f"⚠️ Error retrieving documents: {e}")
            return "Error retrieving context from documents."
    return "RAG retriever not initialized or no documents found."


def load_test_questions(file_path='tw_drone_pro_license_test_shuffled_en.jsonl'):
    questions = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                raw_data = json.loads(line)
                if "prompt" in raw_data:
                    raw = raw_data["prompt"]
                    text = raw.replace("（", "(").replace("）", ")").replace("Ａ", "A").replace("Ｂ", "B").replace("Ｃ",
                                                                                                                    "C").replace(
                        "Ｄ", "D")
                    questions.append(text)
                else:
                    print(f"Warning: 'prompt' key not found in line: {line.strip()}")
            except json.JSONDecodeError as e:
                print(f"JSON decoding failed: {e}, line: {line.strip()}")  # Log JSON errors
                continue
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
            if idx + 1 < len(parts):
                options[key] = parts[idx + 1].strip()
            else:
                options[key] = f"[Parse_Failed_{key}]"
        except Exception:
            options[key] = f"[Parse_Failed_{key}]"

    if not parts[0].strip() and any(opt_marker in cleaned for opt_marker in ["A)", "B)", "C)", "D)"]):
        final_question = cleaned
        final_options = {k: f"[Missing_Stem_Or_Format_Error_{k}]" for k in ['A', 'B', 'C', 'D']}
    else:
        final_question = parts[0].strip() if parts else cleaned
        final_options = options

    if not final_question and cleaned:
        final_question = cleaned

    if all("[Parse_Failed" in v for v in final_options.values()):
        pass
    elif all(f"[Missing_Stem_Or_Format_Error_{k}]" in v for k, v in final_options.items()):
        pass
    else:
        final_options = {k: f"[Stem_Parse_Issue_{k}]" for k in ['A', 'B', 'C', 'D']}

    return {"question": final_question, "options": final_options}


def detect_domain(question, options):
    option_texts = [str(opt) for opt in options.values()]
    text = (str(question) + ' ' + ' '.join(option_texts)).lower()
    domain_weights = {
        "Regulations": sum(3 for kw in DOMAIN_KEYWORDS["Regulations"] if kw in text),
        "Physics": sum(2 for kw in DOMAIN_KEYWORDS["Physics"] if kw in text),
        "Meteorology": sum(2 for kw in DOMAIN_KEYWORDS["Meteorology"] if kw in text),
        "Operations": sum(1 for kw in DOMAIN_KEYWORDS["Operations"] if kw in text)
    }

    if not domain_weights:
        return "Default"

    max_domain = max(domain_weights, key=domain_weights.get, default="Default")
    return max_domain if domain_weights.get(max_domain, 0) > 0 else "Default"


def build_rag_prompt(question_data, retrieved_context):
    options = question_data["options"]
    for key in ['A', 'B', 'C', 'D']:
        options[key] = str(options.get(key, f"[Parse_Failed_{key}]")).replace('"', '').strip()

    question_text = str(question_data.get("question", "")).replace('"', '')
    domain = detect_domain(question_text, options)

    return PROMPT_TEMPLATES[domain].format(
        retrieved_context=retrieved_context,  # Added context parameter
        question=question_text,
        A=options['A'],
        B=options['B'],
        C=options['C'],
        D=options['D']
    )


def extract_answer(text):
    if not isinstance(text, str):
        return "C"  # Default if not a string
    
    # Simply strip whitespace instead of using problematic regex
    cleaned_text = text.strip()
    
    # Highest priority match: If model output format is "Your Answer (A, B, C, or D):X"
    match_with_prefix = re.search(r"Your Answer \(A, B, C, or D\):\s*([A-D])", cleaned_text, re.IGNORECASE)
    if match_with_prefix:
        return match_with_prefix.group(1).upper()
    
    # Check for "Answer: X" format
    match_answer_format = re.search(r"Answer:\s*([A-D])", cleaned_text, re.IGNORECASE)
    if match_answer_format:
        return match_answer_format.group(1).upper()
    
    # Check for standalone letter at the end
    standalone_letter_match = re.search(r"\b([A-D])\s*$", cleaned_text, re.IGNORECASE)
    if standalone_letter_match:
        return standalone_letter_match.group(1).upper()
    
    # If the cleaned text is just a single letter
    if len(cleaned_text) == 1 and cleaned_text.upper() in ['A', 'B', 'C', 'D']:
        return cleaned_text.upper()
    
    # Find all letters A-D in the text and take the last one
    all_options = re.findall(r"([A-D])", cleaned_text, re.IGNORECASE)
    if all_options:
        return all_options[-1].upper()
    
    # Default fallback
    return "C"


def get_llm_response_via_api(full_prompt_text, model_name=MODEL_NAME):
    try:
        messages_for_llm = [
            {"role": "user", "content": full_prompt_text}
        ]
        completion = client.chat.completions.create(
            model=model_name,
            messages=messages_for_llm,
            temperature=0.0,
            max_tokens=10,  # Expecting a single letter answer
            stream=False
        )
        response_content = completion.choices[0].message.content
        return response_content
    except Exception as e:
        print(f"API request error: {e}")
        return None


def compare_with_correct_answers(model_results_df, correct_answers_file=CORRECT_ANSWERS_FILE):
    """
    Compare model answers with correct answers and generate a report.
    
    Args:
        model_results_df: DataFrame with model answers
        correct_answers_file: Path to CSV file with correct answers
    
    Returns:
        DataFrame with comparison results
    """
    print(f"\n📊 Comparing model answers with correct answers from {correct_answers_file}...")
    
    try:
        # Load correct answers
        correct_df = pd.read_csv(correct_answers_file)
        
        # Ensure both DataFrames have the same ID format
        model_results_df['Id'] = model_results_df['Id'].astype(int)
        correct_df['Id'] = correct_df['Id'].astype(int)
        
        # Merge on Id
        merged_df = pd.merge(correct_df, model_results_df, on='Id', suffixes=('_correct', '_model'))
        
        # Compare answers (case-insensitive)
        merged_df['is_correct'] = merged_df.apply(
            lambda row: str(row['Answer_correct']).strip().upper() == str(row['Answer_model']).strip().upper(), 
            axis=1
        )
        
        # Calculate statistics
        total = len(merged_df)
        correct_count = merged_df['is_correct'].sum()
        incorrect_count = total - correct_count
        accuracy = (correct_count / total) * 100 if total > 0 else 0
        
        # Print summary
        print(f"Total questions: {total}")
        print(f"Correct answers: {correct_count}")
        print(f"Incorrect answers: {incorrect_count}")
        print(f"Accuracy: {accuracy:.2f}%")
        
        # Get differences
        differences_df = merged_df[~merged_df['is_correct']].copy()
        
        # Save differences to CSV
        differences_file = f"{MODEL_NAME}_answer_differences.csv"
        differences_df.to_csv(differences_file, index=False)
        print(f"Differences saved to {differences_file}")
        
        # Create a more detailed differences file with question text
        if 'question_texts_map_global' in globals() and question_texts_map_global:
            differences_df['Question_Text'] = differences_df['Id'].map(question_texts_map_global)
            detailed_diff_file = f"{MODEL_NAME}_answer_differences_with_questions.txt"
            
            with open(detailed_diff_file, "w", encoding="utf-8") as f:
                f.write(f"Answer Differences Report for {MODEL_NAME}\n")
                f.write(f"Total questions: {total}\n")
                f.write(f"Correct answers: {correct_count}\n")
                f.write(f"Incorrect answers: {incorrect_count}\n")
                f.write(f"Accuracy: {accuracy:.2f}%\n\n")
                
                for _, row in differences_df.iterrows():
                    f.write(f"ID: {row['Id']}\n")
                    f.write(f"Question: {row['Question_Text']}\n")
                    f.write(f"Correct Answer: {row['Answer_correct']}\n")
                    f.write(f"Model Answer: {row['Answer_model']}\n")
                    f.write("-" * 80 + "\n\n")
                    
            print(f"Detailed differences with questions saved to {detailed_diff_file}")
        
        return differences_df, accuracy
        
    except FileNotFoundError:
        print(f"❌ Correct answers file '{correct_answers_file}' not found.")
        return None, 0
    except Exception as e:
        print(f"❌ Error comparing answers: {e}")
        return None, 0


def main():
    print("🚀 Starting RAG language model inference (using LM Studio API)...")

    setup_rag_pipeline()
    if not retriever:
        print("❌ RAG retriever not successfully initialized, program will run in non-RAG mode (relying only on model's internal knowledge).")

    print("📥 Loading question bank...")
    questions_raw_text = load_test_questions()

    if not questions_raw_text:
        print("Error: Question bank is empty or failed to load, program terminated.")
        return

    print(f"🔍 Starting inference on {len(questions_raw_text)} questions...")
    results = []
    failed_parse_questions = []
    question_id_counter = 0

    for raw_text in tqdm(questions_raw_text):
        question_data = parse_question(raw_text)

        if not question_data["question"] or "[Parse_Failed" in question_data["question"] or \
                all("[Parse_Failed" in opt_val for opt_val in question_data["options"].values()) or \
                all("[Missing_Stem_Or_Format_Error" in opt_val for opt_val in question_data["options"].values()):
            print(f"\nWarning: Question ID {question_id_counter} failed to parse or has invalid stem. Original text: '{raw_text[:100]}...'")
            failed_parse_questions.append(
                {"Id": question_id_counter, "OriginalQuestion": raw_text, "ParsedData": question_data})
            results.append({"Id": question_id_counter, "Answer": "C", "RetrievedContext": "Parsing Failed"})
            question_id_counter += 1
            continue

        # 1. Retrieve relevant context
        query_for_retrieval = question_data["question"]
        retrieved_context = get_retrieved_context(query_for_retrieval)

        # 2. Build RAG prompt
        rag_prompt_text = build_rag_prompt(question_data, retrieved_context)

        # 3. Get LLM response via API
        llm_response = get_llm_response_via_api(rag_prompt_text)

        current_answer = "C"  # Default answer
        if llm_response:
            current_answer = extract_answer(llm_response)

        results.append({
            "Id": question_id_counter,
            "Answer": current_answer,
            # "RetrievedContext": retrieved_context  # Optional: record retrieved context for analysis
        })
        question_id_counter += 1

    if not results:
        print("Error: No questions were successfully processed, cannot generate answer file.")
    else:
        output_filename = f"{MODEL_NAME}_rag_answers.csv"
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_filename, index=False)
        print(f"✅ Complete! Answers saved to {output_filename}")
        
        # Add comparison with correct answers if file exists
        if os.path.exists(CORRECT_ANSWERS_FILE):
            differences_df, accuracy = compare_with_correct_answers(results_df)
        
        counter = Counter([r["Answer"] for r in results])
        print("\n📊 Answer distribution statistics:")
        for opt_char in sorted(counter.keys()):
            if counter[opt_char] > 0:
                print(f"{opt_char}: {counter[opt_char]}")

    if failed_parse_questions:
        print(f"\n⚠️ Number of failed question parses: {len(failed_parse_questions)}, logged in failed_parse_questions_rag.json")
        with open("failed_parse_questions_rag.json", "w", encoding="utf-8") as f:
            json.dump(failed_parse_questions, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()
