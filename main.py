import os
import time
import pandas as pd
import requests
from openai import OpenAI
from ddgs import DDGS  # 已经同步改成新版包名 ddgs

# ========== 配置区 ==========
CSV_FILE = "data.csv"  # 如果你的 CSV 文件叫别的名字（比如 players.csv），改这里
LARK_WEBHOOK = os.getenv("LARK_WEBHOOK")
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")
MODEL = os.getenv("MODEL", "gpt-3.5-turbo")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

print(f"🔧 LARK_WEBHOOK 是否配置: {'是' if LARK_WEBHOOK else '否'}")
print(f"🔧 API_KEY 是否配置: {'是' if API_KEY else '否'}")
print(f"🔧 BASE_URL: {BASE_URL}")
print(f"🔧 MODEL: {MODEL}")

# ========== 搜索模块 ==========
def search_weibo(name):
    """微博搜索（保留占位，实际容易失败）"""
    try:
        print(f"  🔍 微博搜索: {name}")
        # 微博搜索接口经常变，这里保留原逻辑
        url = f"https://m.weibo.cn/api/container/getIndex?containerid=100103type%3D1%26q%3D{name}"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        data = r.json()
        items = []
        for card in data.get("data", {}).get("cards", [])[:5]:
            mblog = card.get("mblog", {})
            text = mblog.get("text", "").replace("<br>", "\n")
            if text:
                items.append({"platform": "微博", "text": text[:200]})
        print(f"  ✅ 微博搜到 {len(items)} 条")
        return items
    except Exception as e:
        print(f"  ❌ 微博搜索失败: {e}")
        return []

def search_web(query):
    """网页搜索（使用新版 ddgs）"""
    try:
        print(f"  🔍 网页搜索: {query}")
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=5)
        items = [{"platform": "网页", "text": r.get("body", "")[:200]} for r in results if r.get("body")]
        print(f"  ✅ 网页搜到 {len(items)} 条")
        return items
    except Exception as e:
        print(f"  ❌ 网页搜索失败: {e}")
        return []

def collect_items(search_names, identity):
    items, seen = [], set()
    for nm in search_names[:2]:
        items += search_web(f"{nm} 荷兰队 足球")
        time.sleep(1.5)
    dedup = []
    for it in items:
        k = it["text"][:50]
        if k not in seen: 
            seen.add(k)
            dedup.append(it)
    return dedup

# ========== LLM 过滤 ==========
def llm_filter(identity, items):
    if not items:
        print("  ⚠️ 没有搜索结果，跳过 LLM")
        return []
    
    name_hints = "、".join(filter(None, [identity.get("full"), identity.get("common"), identity.get("orig"), *identity.get("alias", []), *identity.get("nick", [])]))
    ctx = []
    if identity.get("position"): ctx.append(f"位置：{identity['position']}")
    if identity.get("wc"): ctx.append(f"世界杯：{identity['wc']}")
    if identity.get("ec"): ctx.append(f"欧洲杯：{identity['ec']}")
    ctx_line = "；".join(ctx) or "荷兰足球人物"
    
    numbered = "\n".join(f"{i+1}. [{it['platform']}] {it['text']}" for i, it in enumerate(items))
    prompt = f"目标人物称呼：{name_hints}\n背景：{ctx_line}\n\n只保留确实在讲此人的动态/新闻。同名不同人、非荷兰足球、纯广告抽奖不要。\n\n输出：\n序号 | 平台 | 一句话摘要\n\n{numbered}"
    
    try:
        print(f"  🤖 调用 LLM ({MODEL})...")
        resp = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0)
        
        # 打印 LLM 原始返回，方便排查
        raw_content = resp.choices[0].message.content
        print(f"  📝 LLM 原始返回:\n{raw_content}")
        
        # 放宽过滤条件：只要不是空行就保留
        result = [l.strip() for l in raw_content.splitlines() if l.strip()]
        print(f"  ✅ LLM 返回 {len(result)} 条结果")
        return result
    except Exception as e:
        print(f"  ❌ LLM 调用失败: {e}")
        return []

# ========== 飞书推送 ==========
def send_lark(title, lines):
    text = f"球员周报 | {title}\n\n" + "\n".join(lines)
    print(f"📤 发送到飞书: {title}, {len(lines)} 条")
    try:
        r = requests.post(LARK_WEBHOOK, json={"msg_type": "text", "content": {"text": text[:18000]}}, timeout=10)
        print(f"📤 飞书返回: {r.status_code} {r.text}")
    except Exception as e:
        print(f"📤 飞书发送失败: {e}")

# ========== 人员解析 ==========
def get_person(row):
    """从 CSV 行解析人员信息（根据你的 CSV 列名调整）"""
    identity = {
        "full": row.get("full_name", ""),
        "common": row.get("name", ""),
        "orig": row.get("original_name", ""),
        "alias": [],
        "nick": [],
        "position": row.get("position", ""),
        "wc": row.get("world_cup", ""),
        "ec": row.get("euro", "")
    }
    
    # 搜索词
    search_names = [identity["common"], identity["full"], identity["orig"]]
    search_names = [s for s in search_names if s]
    
    return search_names, identity

# ========== 主函数 ==========
def main():
    print("🚀 开始运行...")
    df = pd.read_csv(CSV_FILE, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    print(f"📋 共读取 {len(df)} 行数据")
    
    # 测试模式：只跑前 3 人
    df = df.head(3)
    print(f"📋 测试模式：只跑前 {len(df)} 人")
    
    for i, (_, row) in enumerate(df.iterrows(), 1):
        search_names, identity = get_person(row)
        name = identity["full"] or identity["common"]
        print(f"\n--- [{i}/{len(df)}] {name} ---")
        print(f"  搜索词: {search_names}")
        
        if not search_names:
            print("  ⚠️ 没有搜索词，跳过")
            continue
            
        items = collect_items(search_names, identity)
        print(f"  汇总 {len(items)} 条候选")
        
        res = llm_filter(identity, items)
        if res:
            send_lark(name, res)
            print(f"  ✅ 已发送飞书")
        else:
            print(f"  ⚠️ 没有相关内容，未发送")
            
        time.sleep(3)
        
    print("\n🏁 全部完成")

if __name__ == "__main__":
    main()
