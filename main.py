import os
import re
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

# 每周时间区间（测试写死，正式可改成自动算）
WEEK_START = "20261012"
WEEK_END = "20261018"
REPORT_TITLE = f"{WEEK_START}-{WEEK_END} 荷兰国家队历史球员资讯"

# 中文体育源优先级（数字越大越优先）
CN_SOURCE_RANK = {
    "dongqiudi.com": 100,
    "hupu.com": 90,
    "zhibo8.com": 80,
    "sports.sina.com.cn": 75,
    "sina.com.cn": 70,
    "qq.com": 65,
    "sohu.com": 60,
    "163.com": 60,
    "toutiao.com": 55,
}

# 外文权威源优先级
EN_SOURCE_RANK = {
    "ad.nl": 90, "telegraaf.nl": 88, "vi.nl": 86, "nos.nl": 85,
    "sportnieuws.nl": 80,
    "espn.com": 85, "bbc.com": 84, "skysports.com": 83, "theguardian.com": 82,
    "reuters.com": 85, "apnews.com": 84,
}

BLOCK_DOMAINS = ["yahoo.com", "msn.com", "consent.", "privacy"]

MAX_PER_PERSON = 3

# ========== 工具 ==========
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
    eng_name = orig.strip() if isinstance(orig, str) and orig.strip() else (
        full.strip() if isinstance(full, str) else "")
    return search_names, {
        "full": full if isinstance(full, str) else "",
        "common": common if isinstance(common, str) else "",
        "orig": orig if isinstance(orig, str) else "",
        "eng_name": eng_name,
        "alias": cell_to_list(row.get("别名")),
        "position": row.get("位置", ""),
    }

def host_of(url):
    if not url:
        return ""
    m = re.search(r"https?://([^/]+)/?", url)
    return m.group(1).lower() if m else ""

def is_blocked(url):
    h = host_of(url)
    return any(b in h for b in BLOCK_DOMAINS)

def is_chinese(url):
    h = host_of(url)
    return any(cn in h for cn in CN_SOURCE_RANK)

# ========== 搜索 ==========
def search_news(query):
    out = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.news(query, region="nl-nl", timelimit="w", max_results=6):
                url = r.get("url") or ""
                if not url or is_blocked(url):
                    continue
                out.append({
                    "title": r.get("title", ""),
                    "url": url,
                    "text": (r.get("body") or "")[:300],
                    "source": (r.get("source") or "").lower(),
                    "date": r.get("date", ""),
                    "lang": "auto",
                })
        print(f"  ✅ news {len(out)} 条")
    except Exception as e:
        print(f"  ❌ news 失败: {e}")
    return out

def search_text(query):
    # 兜底，只取中文源；外文 text 结果无日期不信
    out = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=6):
                url = r.get("href") or ""
                if not url or is_blocked(url):
                    continue
                if not is_chinese(url):
                    continue
                out.append({
                    "title": r.get("title", ""),
                    "url": url,
                    "text": (r.get("body") or "")[:300],
                    "source": host_of(url),
                    "date": "",
                    "lang": "cn",
                })
        print(f"  ✅ text 中文 {len(out)} 条")
    except Exception as e:
        print(f"  ❌ text 失败: {e}")
    return out

def collect_items(search_names, eng_name):
    items = []
    seen = set()
    if eng_name:
        items += search_news(f"{eng_name} Netherlands footballer")
        time.sleep(1.5)
        items += search_news(f"{eng_name} Oranje news")
        time.sleep(1.5)
    for nm in search_names[:2]:
        items += search_text(f"{nm} 荷兰队")
        time.sleep(1.5)
    dedup = []
    for it in items:
        k = it["url"]
        if k not in seen:
            seen.add(k)
            dedup.append(it)
    return dedup

# ========== LLM：只做分组 + 外文翻译 ==========
def llm_group_and_translate(identity, items):
    if not items:
        return []
    name_hints = "、".join(filter(None, [identity["full"], identity["common"], identity["orig"], *identity["alias"]]))
    blocks = []
    for i, it in enumerate(items, 1):
        blocks.append(f"[{i}] ({'中文' if it['lang']=='cn' else '外文'}) {it['title']}\n{it['text']}")
    joined = "\n\n".join(blocks)

    prompt = f"""目标人物：{name_hints}
下面是候选新闻，可能中英文混着来。

请做两件事：
1. 把「同一件事」归到一组，每组只留最该保留的那 1 条（中文优先；多条中文选最权威；无中文选最权威外文）
2. 对被选中的「外文」条目，给出：
   中文标题 | 150-200字中文摘要（包含事件经过、当事人角色、背景、后续）
   对被选中的「中文」条目，只给：
   中文标题

输出格式（每行一条，最多 {MAX_PER_PERSON} 条）：
序号 | 中文标题 | 摘要（中文条摘要写“-”）
---
{joined}"""
    try:
        resp = client.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0)
        raw = resp.choices[0].message.content
        print("  📝 LLM 返回:\n" + raw)
        parsed = []
        for line in raw.splitlines():
            line = line.strip()
            if "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                continue
            idx = parts[0].strip("[]").strip()
            try:
                i = int(re.findall(r"\d+", idx)[0]) - 1
            except:
                continue
            if 0 <= i < len(items):
                parsed.append({
                    "item": items[i],
                    "title": parts[1],
                    "summary": parts[2],
                })
        return parsed[:MAX_PER_PERSON]
    except Exception as e:
        print(f"  ❌ LLM 失败: {e}")
        return []

# ========== 飞书拼接 ==========
def build_post(all_people):
    content = []
    content.append([{"tag": "text", "text": f"🇳🇱 {REPORT_TITLE}", "style": ["bold"]}])
    content.append([{"tag": "text", "text": "\n"}])

    any_news = False
    for name, news_list in all_people:
        if not news_list:
            continue
        any_news = True
        content.append([{"tag": "text", "text": f"{name}", "style": ["bold"]}])
        for n in news_list:
            it = n["item"]
            if it["lang"] == "cn":
                # 中文：只标题+链接
                content.append([
                    {"tag": "text", "text": f"• {n['title']}\n"},
                    {"tag": "a", "text": "原文链接", "href": it["url"]},
                ])
            else:
                # 外文：标题+摘要+链接
                summary = n["summary"] if n["summary"] not in ("-", "") else ""
                content.append([
                    {"tag": "text", "text": f"• {n['title']}\n{summary}\n"},
                    {"tag": "a", "text": "原文链接", "href": it["url"]},
                ])
        content.append([{"tag": "text", "text": "\n"}])

    if not any_news:
        content.append([{"tag": "text", "text": "本周暂无符合条件的动态"}])

    return {
        "msg_type": "post",
        "content": {"post": {"zh_cn": {"title": REPORT_TITLE, "content": content}}},
    }

def send_lark(payload):
    if not LARK_WEBHOOK:
        print("⚠️ 没配 webhook，不发送")
        return
    try:
        r = requests.post(LARK_WEBHOOK, json=payload, timeout=15)
        print(f"📤 飞书发送: {r.status_code}")
    except Exception as e:
        print(f"📤 飞书失败: {e}")

# ========== 主函数 ==========
def main():
    print("🚀 开始")
    df = pd.read_csv(CSV_FILE, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    test_kw = ["斯内德", "库伊特", "罗本", "德弗赖"]
    df_test = df[df["全名"].astype(str).str.contains("|".join(test_kw), na=False)]
    df = df_test if len(df_test) else df.head(4)
    print(f"📋 测试 {len(df)} 人")

    all_people = []
    for _, row in df.iterrows():
        search_names, identity = get_person(row)
        name = identity["full"] or identity["common"]
        print(f"\n=== {name} ===")

        items = collect_items(search_names, identity["eng_name"])
        if not items:
            print("  无候选")
            all_people.append((name, []))
            continue

        picked = llm_group_and_translate(identity, items)
        print(f"  选中 {len(picked)} 条")
        all_people.append((name, picked))
        time.sleep(2)

    payload = build_post(all_people)
    send_lark(payload)
    print("\n🏁 完成")

if __name__ == "__main__":
    main()
