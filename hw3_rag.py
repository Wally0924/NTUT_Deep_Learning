import numpy as np
import pandas as pd
import os
import json
from tqdm import tqdm
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from sentence_transformers import CrossEncoder
import keras
import keras_hub

# 設定環境變數
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.00"
os.environ["KERAS_BACKEND"] = "jax"  # 或 "tensorflow" 或 "torch"

# 檔案路徑設定
DRONE_TEST_FILE = '/home/rvl1421/NTUT_Deep_Learning/tw_drone_pro_license_test_shuffled_en.jsonl'
DRONE_REGULATION_EN = '/home/rvl1421/NTUT_Deep_Learning/Taiwan_UAV_Regulations_20241114.txt'
DRONE_REGULATION_ZH = '/home/rvl1421/NTUT_Deep_Learning/_1131114.txt'

# 載入 LLM 模型
gemma_lm = keras_hub.models.Gemma3CausalLM.from_preset(
    "gemma3_instruct_4b",
    dtype="bfloat16",
)

# 設定模型參數
gemma_lm.preprocessor.max_images_per_prompt = 2
gemma_lm.preprocessor.sequence_length = 768

# 載入法規文件
def load_regulations():
    with open(DRONE_REGULATION_EN, 'r', encoding='utf-8') as f:
        en_regulations = f.read()
    
    with open(DRONE_REGULATION_ZH, 'r', encoding='utf-8') as f:
        zh_regulations = f.read()
    
    return en_regulations, zh_regulations

# 文本分塊
def split_documents(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return text_splitter.split_text(text)

# 建立向量資料庫
def create_vector_store(chunks):
    # 使用多語言模型以支援中英文
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

# 重排序機制
def rerank_documents(query, docs, top_k=3):
    reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    # 準備重排序
    pairs = [(query, doc) for doc in docs]
    scores = reranker.predict(pairs)
    
    # 根據分數排序
    scored_docs = list(zip(docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # 返回重排序後的前k個文檔
    return [doc for doc, score in scored_docs[:top_k]]

# 使用RAG回答問題
def answer_question_with_rag(question, options, vector_store, model):
    # 構建查詢
    query = f"{question} Options: {', '.join(options)}"
    
    # 檢索相關文件 (檢索更多文件以提高覆蓋率)
    docs = vector_store.similarity_search(query, k=7)
    docs_content = [doc.page_content for doc in docs]
    
    # 重排序文件
    reranked_docs = rerank_documents(query, docs_content)
    context = "\n\n".join(reranked_docs)
    
    # 構建增強提示
    prompt = f"""You are an expert on UAV regulations and drone operations. Based on the following information, answer the multiple-choice question. Choose only one option: A, B, C, or D.

Context information:
{context}

Question: {question}
A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}

Your answer should be just the letter (A, B, C, or D) without explanation:"""
    
    # 使用模型生成答案
    response = model.generate(prompt, max_tokens=10)
    
    # 提取答案字母
    answer = response.strip()
    # 確保只返回單一字母
    if len(answer) > 0 and answer[0] in "ABCD":
        return answer[0]
    
    # 如果無法提取單一字母，嘗試從回答中找出最可能的選項
    for letter in "ABCD":
        if letter in answer:
            return letter
    
    # 如果還是找不到，返回None
    return None

# 自反思RAG (增強版)
def self_reflective_rag(question, options, vector_store, model):
    # 初始檢索
    query = f"{question} Options: {', '.join(options)}"
    docs = vector_store.similarity_search(query, k=5)
    docs_content = [doc.page_content for doc in docs]
    
    # 重排序文件
    reranked_docs = rerank_documents(query, docs_content)
    context = "\n\n".join(reranked_docs)
    
    # 第一階段：評估檢索內容是否足夠
    evaluation_prompt = f"""Question: {question}
Options: A) {options[0]}, B) {options[1]}, C) {options[2]}, D) {options[3]}

Retrieved information:
{context}

Is the retrieved information sufficient to answer the question? Answer YES or NO:"""
    
    eval_response = model.generate(evaluation_prompt, max_tokens=50).strip()
    
    # 如果檢索不足，擴大檢索範圍
    if "NO" in eval_response.upper():
        # 擴大檢索範圍並嘗試不同的查詢方式
        expanded_query = f"drone UAV regulations {question}"
        additional_docs = vector_store.similarity_search(expanded_query, k=5)
        additional_content = [doc.page_content for doc in additional_docs]
        
        # 合併並重排序
        all_docs = list(set(docs_content + additional_content))
        reranked_docs = rerank_documents(query, all_docs, top_k=5)
        context = "\n\n".join(reranked_docs)
    
    # 最終回答
    answer_prompt = f"""You are an expert on UAV regulations and drone operations. Based on the following information, answer the multiple-choice question.
Choose only one option: A, B, C, or D.

Context information:
{context}

Question: {question}
A) {options[0]}
B) {options[1]}
C) {options[2]}
D) {options[3]}

Your answer should be just the letter (A, B, C, or D) without explanation:"""
    
    final_answer = model.generate(answer_prompt, max_tokens=10).strip()
    
    # 提取答案字母
    if len(final_answer) > 0 and final_answer[0] in "ABCD":
        return final_answer[0]
    
    # 如果無法提取單一字母，嘗試從回答中找出最可能的選項
    for letter in "ABCD":
        if letter in final_answer:
            return letter
    
    # 如果還是找不到，返回None
    return None

# 主程式
def main():
    # 載入法規文件
    en_regulations, zh_regulations = load_regulations()
    
    # 分塊處理
    en_chunks = split_documents(en_regulations)
    zh_chunks = split_documents(zh_regulations)
    
    # 合併中英文資料 (可選，視需求決定是否只使用英文)
    all_chunks = en_chunks + zh_chunks
    
    # 建立向量資料庫
    vector_store = create_vector_store(all_chunks)
    
    # 載入測試資料
    with open(DRONE_TEST_FILE, 'r') as f:
        test_data = [json.loads(line) for line in f]
    
    # 回答問題
    answers = []
    for i, item in tqdm(enumerate(test_data)):
        question = item['prompt']
        
        # 解析選項 (假設選項格式為 "A. 選項內容")
        options_pattern = r'([A-D])\.\s+(.*?)(?=\s+[A-D]\.|$)'
        options_matches = re.findall(options_pattern, question, re.DOTALL)
        
        if options_matches:
            # 提取問題和選項
            question_text = question.split('A.')[0].strip()
            options = [match[1].strip() for match in options_matches]
            
            # 確保有四個選項
            while len(options) < 4:
                options.append("")
            
            # 使用自反思RAG回答問題
            answer = self_reflective_rag(question_text, options, vector_store, gemma_lm)
            
            # 如果無法得到答案，使用基本RAG方法
            if not answer:
                answer = answer_question_with_rag(question_text, options, vector_store, gemma_lm)
            
            # 如果還是無法得到答案，預設為A
            if not answer:
                answer = "A"
        else:
            # 如果無法解析選項，預設為A
            answer = "A"
        
        answers.append({'Id': i, 'Answer': answer})
    
    # 保存結果
    pd.DataFrame(answers).to_csv('submission.csv', index=False)
    print("完成！答案已保存至 submission.csv")

if __name__ == "__main__":
    main()
