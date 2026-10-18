import os
import time
import requests
import pandas as pd
from openai import OpenAI
from ddgs import DDGS

CSV_FILE = "Dutch-National-Football-Team-Coach&Player-since-1994.csv"

LARK_WEBHOOK = os.environ.get("LARK_WEBHOOK")
API_KEY = os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL = os.environ.get("MODEL_NAME", "gpt-4o-mini")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def cell_to_list(v):
    if not isinstance(v, str):
        return []
    return [p.strip() for p in v.replace("；", "/").replace(";", "/").split("/") if p.strip()]

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
        "position": row.get("位置", ""),
        "wc": row.get("世界杯参加年", ""),
        "ec": row.get("欧洲杯参加年", ""),
    }
    return search_names, identity

def search_web(query):
    out = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, timelimit="w", max_results=5):
                if r.get("body"):
                    out.append({
                        "title": r.get("title", ""),
                        "url": r.get("href", ""),
                        "text": r.get("body", "")[:300]
                    })
        print(f"  ✅ 搜到 {len(out)} 条")
    except Exception as e:
        print(f"  ❌ 搜索失败: {e}")
    return out

def collect_items(search_names):
    items = []
    seen = set()
    for nm in search_names[:2]:
        for it in search_web(f"{nm} 荷兰队 足球"):
            k = it["url"] or it["text"][:40]
            if k not in seen:
                seen.add(k)
                items.append(it)
        time.sleep(1.5)
    return items

def llm_filter(identity, items):
    if not items:
        print("  ⚠️ 无搜索结果")
        return []

    name_hints = "、".join(filter(None, [
        identity["full"], identity["common"], identity["orig"],
        *identity["alias"], *identity["nick"]
    ]))
    ctx = []
    if identity["position"]:
        ctx.append(f"位置：{identity['position']}")
    if identity["wc"]:
        ctx.append(f"世界杯：{identity['wc']}")
    if identity["ec"]:
        ctx.append(f"欧洲杯：{identity['ec']}")
    ctx_line = "；".join(ctx) or "荷兰足球人物"

    blocks = []
    for i, it in enumerate(items, 1):
        blocks.append(f"[{i}] 标题：{it['title']}\n链接：{it['url']}\n摘要：{it['text']}")
    joined = "\n\n".join(blocks)

    prompt = f"""目标人物：{name_hints}
背景：{ctx_line}
现在是2026年，只保留「近7天内」关于此人的动态/新闻/转会/采访/伤情。

对每条输入，按下面格式输出，不要解释、不要空行：
序号 | 是否近期(是/否) | 中文翻译标题
---
{joined}"""

    try:
        print(f"  🤖 调用 LLM...")
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        lines = [l.strip() for l in resp.choices[0].message.content.splitlines()
                 if l.strip() and "|" in l]

        results = []
        for line in lines:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                continue
            idx, recent, zh = parts[0], parts[1], parts[2]
            if recent != "是":
                continue
            try:
                i = int(idx) - 1
            except:
                continue
            if 0 <= i < len(items):
                it = items[i]
                results.append({
                    "title": it["title"],
                    "url": it["url"],
                    "zh": zh
                })
        print(f"  ✅ 过滤后 {len(results)} 条本周动态")
        return results
    except Exception as e:
        print(f"  ❌ LLM 失败: {e}")
        return []

def send_lark(name, results):
    content = {
        "msg_type": "post",
        "content": {
            "post": {
                "zh_cn": {
                    "title": f"球员周报 | {name}",
                    "content": []
                }
            }
        }
    }
    for r in results:
        content["content"]["post"]["zh_cn"]["content"].append([
            {"tag": "text", "text": f"• {r['zh']}\n"},
            {"tag": "a", "text": "原文链接", "href": r["url"]},
        ])
    if not results:
        content["content"]["post"]["zh_cn"]["content"].append([
            {"tag": "text", "text": "本周无新动态"}
        ])
    try:
        r = requests.post(LARK_WEBHOOK, json=content, timeout=10)
        print(f"  📤 飞书发送: {r.status_code}")
    except Exception as e:
        print(f"  📤 飞书失败: {e}")

def main():
    print("🚀 开始运行...")
    df = pd.read_csv(CSV_FILE, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    print(f"📋 共 {len(df)} 行")

    df = df.head(3)
    print(f"📋 测试模式：前 {len(df)} 人\n")

    for i, (_, row) in enumerate(df.iterrows(), 1):
        search_names, identity = get_person(row)
        name = identity["full"] or identity["common"]
        print(f"--- [{i}/{len(df)}] {name} ---")
        print(f"  搜索词: {search_names}")

        if not search_names:
            print("  ⚠️ 无搜索词，跳过\n")
            continue

        items = collect_items(search_names)
        print(f"  汇总 {len(items)} 条候选")
        res = llm_filter(identity, items)

        if res:
            send_lark(name, res)
            print(f"  ✅ 已发送飞书")
        else:
            print(f"  ⚠️ 本周无相关动态")
        print()
        time.sleep(3)

    print("🏁 全部完成")

if __name__ == "__main__":
    main()
