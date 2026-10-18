import os
import time
import requests
import pandas as pd
from openai import OpenAI
from ddgs import DDGS

# ========== 配置 ==========
CSV_FILE = "Dutch-National-Football-Team-Coach&Player-since-1994.csv"

LARK_WEBHOOK = os.environ.get("LARK_WEBHOOK")
API_KEY = os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL = os.environ.get("MODEL_NAME", "gpt-4o-mini")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# ========== 工具函数 ==========
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
    eng_name = ""
    if isinstance(orig, str) and orig.strip():
        eng_name = orig.strip()
    elif isinstance(full, str) and full.strip():
        eng_name = full.strip()
    identity = {
        "full": full if isinstance(full, str) else "",
        "common": common if isinstance(common, str) else "",
        "orig": orig if isinstance(orig, str) else "",
        "eng_name": eng_name,
        "alias": cell_to_list(row.get("别名")),
        "nick": cell_to_list(row.get("外号")),
        "position": row.get("位置", ""),
        "wc": row.get("世界杯参加年", ""),
        "ec": row.get("欧洲杯参加年", ""),
    }
    return search_names, identity

# ========== 搜索：新闻频道（主力） ==========
def search_news(query, region="nl-nl"):
    out = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.news(query, region=region, max_results=5):
                if r.get("body") or r.get("title"):
                    out.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", r.get("href", "")),
                        "text": (r.get("body") or r.get("excerpt") or "")[:300]
                    })
        print(f"  ✅ 新闻频道搜到 {len(out)} 条")
    except Exception as e:
        print(f"  ❌ 新闻搜索失败: {e}")
    return out

# ========== 搜索：网页频道（兜底） ==========
def search_text(query):
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
        print(f"  ✅ 网页频道搜到 {len(out)} 条")
    except Exception as e:
        print(f"  ❌ 网页搜索失败: {e}")
    return out

def collect_items(search_names, eng_name=""):
    items = []
    seen = set()
    if eng_name:
        items += search_news(f"{eng_name} Netherlands football")
        time.sleep(1.5)
        items += search_news(f"{eng_name} Dutch national team")
        time.sleep(1.5)
    for nm in search_names[:2]:
        items += search_text(f"{nm} 荷兰队 足球")
        time.sleep(1.5)
    deduped = []
    for it in items:
        k = it["url"] or it["text"][:40]
        if k not in seen:
            seen.add(k)
            deduped.append(it)
    return deduped

# ========== LLM 过滤 + 翻译 ==========
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
当前日期：2026年10月。请判断每条是否是此人的近期动态/新闻/转会/采访/伤情。

对每条输入，按下面格式输出，不要解释、不要空行：
序号 | 是否近期(是/否) | 中文翻译标题
---
{joined}"""

    try:
        print(f"  🤖 调用 LLM ({MODEL})...")
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        raw = resp.choices[0].message.content
        print(f"  📝 LLM 原始返回:\n{raw}")
        lines = [l.strip() for l in raw.splitlines() if l.strip() and "|" in l]

        results = []
        for line in lines:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                continue
            # 修复：去掉方括号再转整数
            idx_raw = parts[0].strip("[]").strip()
            recent = parts[1].strip()
            zh = parts[2].strip()
            if recent != "是":
                continue
            try:
                i = int(idx_raw) - 1
            except:
                continue
            if 0 <= i < len(items):
                it = items[i]
                results.append({
                    "title": it["title"],
                    "url": it["url"],
                    "zh": zh
                })
        print(f"  ✅ 过滤后 {len(results)} 条")
        return results
    except Exception as e:
        print(f"  ❌ LLM 失败: {e}")
        return []

# ========== 飞书发送（富文本） ==========
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
            {"tag": "text", "text": "本周暂无新动态"}
        ])
    try:
        r = requests.post(LARK_WEBHOOK, json=content, timeout=10)
        print(f"  📤 飞书发送: {r.status_code}")
    except Exception as e:
        print(f"  📤 飞书失败: {e}")

# ========== 主函数 ==========
def main():
    print("🚀 开始运行...")
    print(f"🔧 LARK_WEBHOOK: {'已配置' if LARK_WEBHOOK else '未配置'}")
    print(f"🔧 API_KEY: {'已配置' if API_KEY else '未配置'}")
    print(f"🔧 MODEL: {MODEL}")

    df = pd.read_csv(CSV_FILE, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    print(f"📋 共读取 {len(df)} 行")

    # 测试模式：跑现役球员
    test_keywords = ["范戴克", "加克波", "德佩", "邓弗里斯", "哈维·西蒙斯"]
    df_test = df[df["全名"].astype(str).str.contains("|".join(test_keywords), na=False)]
    if len(df_test) > 0:
        df = df_test
        print(f"📋 测试模式：跑 {len(df)} 名现役球员")
    else:
        df = df.head(3)
        print(f"📋 测试模式：前 3 人")

    for i, (_, row) in enumerate(df.iterrows(), 1):
        search_names, identity = get_person(row)
        name = identity["full"] or identity["common"]
        print(f"\n--- [{i}/{len(df)}] {name} ---")
        print(f"  英文名: {identity['eng_name']}")
        print(f"  搜索词: {search_names}")

        if not search_names and not identity["eng_name"]:
            print("  ⚠️ 无搜索词，跳过")
            continue

        items = collect_items(search_names, identity["eng_name"])
        print(f"  汇总 {len(items)} 条候选")
        res = llm_filter(identity, items)

        if res:
            send_lark(name, res)
            print(f"  ✅ 已发送飞书 ({len(res)} 条)")
        else:
            print(f"  ⚠️ 本周无相关动态")
        time.sleep(3)

    print("\n🏁 全部完成")

if __name__ == "__main__":
    main()
