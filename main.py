import os
import time
import requests
import pandas as pd
from openai import OpenAI
from ddgs import DDGS

# ========== 配置区（已修正）==========
CSV_FILE = "Dutch-National-Football-Team-Coach&Player-since-1994.csv"

LARK_WEBHOOK = os.environ.get("LARK_WEBHOOK")
API_KEY = os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL = os.environ.get("MODEL_NAME", "gpt-4o-mini")

print(f"🔧 LARK_WEBHOOK 是否配置: {'是' if LARK_WEBHOOK else '否'}")
print(f"🔧 API_KEY 是否配置: {'是' if API_KEY else '否'}")
print(f"🔧 BASE_URL: {BASE_URL}")
print(f"🔧 MODEL: {MODEL}")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
HEADERS = {"User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X)"}

def cell_to_list(v):
    if not isinstance(v, str): return []
    return [p.strip() for p in v.replace("；","/").replace(";","/").split("/") if p.strip()]

def get_person(row):
    full = row.get("全名")
    common = row.get("常用名")
    orig = row.get("原名")
    search_names = []
    for n in [common, full]:
        if isinstance(n, str) and n.strip():
            search_names.append(n.strip())
    if isinstance(orig, str) and orig.strip():
        search_names.append(orig.strip().split()[-1])
    search_names = list(dict.fromkeys(search_names))
    identity = {
        "full": full if isinstance(full, str) else "",
        "common": common if isinstance(common, str) else "",
        "orig": orig if isinstance(orig, str) else "",
        "alias": cell_to_list(row.get("别名")),
        "nick": cell_to_list(row.get("外号")),
        "position": row.get("位置",""),
        "wc": row.get("世界杯参加年",""),
        "ec": row.get("欧洲杯参加年",""),
    }
    return search_names, identity

def search_web(query):
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
        if k not in seen: seen.add(k); dedup.append(it)
    return dedup

def llm_filter(identity, items):
    if not items:
        print("  ⚠️ 没有搜索结果，跳过 LLM")
        return []
    name_hints = "、".join(filter(None,[identity["full"],identity["common"],identity["orig"],*identity["alias"],*identity["nick"]]))
    ctx = []
    if identity["position"]: ctx.append(f"位置：{identity['position']}")
    if identity["wc"]: ctx.append(f"世界杯：{identity['wc']}")
    if identity["ec"]: ctx.append(f"欧洲杯：{identity['ec']}")
    ctx_line = "；".join(ctx) or "荷兰足球人物"
    numbered = "\n".join(f"{i+1}. [{it['platform']}] {it['text']}" for i,it in enumerate(items))
    prompt = f"目标人物称呼：{name_hints}\n背景：{ctx_line}\n\n只保留确实在讲此人的动态/新闻。同名不同人、非荷兰足球、纯广告抽奖不要。\n\n输出：\n序号 | 平台 | 一句话摘要\n\n{numbered}"
    try:
        print(f"  🤖 调用 LLM ({MODEL})...")
        resp = client.chat.completions.create(model=MODEL, messages=[{"role":"user","content":prompt}], temperature=0)
        raw_content = resp.choices[0].message.content
        print(f"  📝 LLM 原始返回:\n{raw_content}")
        result = [l.strip() for l in raw_content.splitlines() if l.strip()]
        print(f"  ✅ LLM 返回 {len(result)} 条结果")
        return result
    except Exception as e:
        print(f"  ❌ LLM 调用失败: {e}")
        return []

def send_lark(title, lines):
    text = f"球员周报 | {title}\n\n" + "\n".join(lines)
    print(f"📤 发送到飞书: {title}, {len(lines)} 条")
    try:
        r = requests.post(LARK_WEBHOOK, json={"msg_type":"text","content":{"text":text[:18000]}}, timeout=10)
        print(f"📤 飞书返回: {r.status_code} {r.text}")
    except Exception as e:
        print(f"📤 飞书发送失败: {e}")

def main():
    print("🚀 开始运行...")
    df = pd.read_csv(CSV_FILE, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    print(f"📋 共读取 {len(df)} 行数据")
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
