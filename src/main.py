Python
import os
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

# 載入 .env 檔裡面的 API 金鑰
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

# --- 1. 定義 Pydantic 結構化輸出模型 ---
class EmailClassification(BaseModel):
    category: str = Field(description="分類：選課問題、勞作教育、英文門檻、其他")
    urgency: int = Field(description="緊急程度 1 至 5 分")
    summary: str = Field(description="信件主旨摘要")
    suggested_reply: str = Field(description="建議助教回覆之內容草稿")

# --- 2. FAQ 資料庫與 Embedding 向量比對 ---
faqs = [
    {"q": "選課衝堂怎麼辦？", "a": "請於加退選期間透過校務系統進行線上退選，或洽系辦填寫人工加退選單。"},
    {"q": "勞作教育請假規定為何？", "a": "勞作教育請假須於事發三天內附證明至系統或勞作教育組辦公室登記。"},
    {"q": "畢業門檻的英文檢定分數是多少？", "a": "本系畢業門檻須通過多益 550 分或同等英語能力測驗。"}
]

def get_embedding(text: str):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def search_faq(query: str):
    import numpy as np
    query_vec = np.array(get_embedding(query))
    best_match = None
    max_sim = -1
    
    for faq in faqs:
        faq_vec = np.array(get_embedding(faq["q"]))
        # 計算餘弦相似度 (Cosine Similarity)
        sim = np.dot(query_vec, faq_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(faq_vec))
        if sim > max_sim:
            max_sim = sim
            best_match = faq
    return best_match, max_sim

# --- 3. 執行測試主程式 ---
if __name__ == "__main__":
    print("=== 1. FAQ 語意檢索測試 (Embedding) ===")
    user_query = "請問多益要幾分才能畢業？"
    match, score = search_faq(user_query)
    print(f"使用者問題: {user_query}")
    print(f"最佳配對 FAQ: {match['q']} (相似度: {score:.4f})")
    print(f"解答: {match['a']}\n")

    print("=== 2. 客服信件結構化分類測試 (Structured Output) ===")
    incoming_email = "助教好，我這學期有一門必修跟選修衝堂了，系統一直擋我選課，請問該怎麼辦？"
    
    completion = client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "你是一個專業的系辦助教 AI，請將來信進行結構化分析。"},
            {"role": "user", "content": incoming_email}
        ],
        response_format=EmailClassification,
    )
    
    result = completion.choices.message.parsed
    print(result.model_dump_json(indent=2, ensure_ascii=False))
