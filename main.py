import os
import re
import json
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from openai import OpenAI
from ddgs import DDGS

# ========== 配置 ==========
CSV_FILE = "Dutch-National-Football-Team-Coach&Player-since-1994.csv"

LARK_WEBHOOK = os.environ.get("LARK_WEBHOOK")
API_KEY = os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL = os.environ.get("MODEL_NAME", "gpt-4o-mini")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 周一跑 → 上周一到上周日；其他时间 → 过去7天
_today = datetime.now()
_wd = _today.weekday()  # 0=周一
if _wd == 0:
    WEEK_END = (_today - timedelta(days=1)).strftime("%Y%m%d")   # 昨天=上周日
    WEEK_START = (_today - timedelta(days=7)).strftime("%Y%m%d") # 上周一
else:
    WEEK_END = _today.strftime("%Y%m%d")
    WEEK_START = (_today - timedelta(days=7)).strftime("%Y%m%d")

CURRENT_YEAR = _today.year
REPORT_TITLE = f"{WEEK_START}-{WEEK_END} 荷兰国家队历史球员资讯"
print(f"📅 本周区间: {WEEK_START} ~ {WEEK_END}")

BLOCK_DOMAINS = ["yahoo.com", "msn.com", "consent.", "privacy"]
MAX_PER_PERSON = 3
NEWS_MAX = 5
TEXT_MAX = 5

REVIEW_KW = ["回顾", "复盘", "专访", "谈当年", "谈20", "回忆", "昔日", "当年",
             "旧将", "名宿", "传奇", "诈伤", "曾效力", "曾执教", "退役",
             "当年那支", "那届世界杯", "那届欧洲杯"]

# ========== 工具 ==========
def cell_to_list(v):
    if not isinstance(v, str): return []
    return [p.strip() for p in v.replace("；", "/").replace(";", "/").split("/") if p.strip()]

def get_person(row):
    full = row.get("全名"); common = row.get("常用名"); orig = row.get("原名")
    search_names = []
    for n in [common, full]:
        if isinstance(n, str) and n.strip(): search_names.append(n.strip())
    if isinstance(orig, str) and orig.strip():
        search_names.append(orig.strip().split()[-1])
    search_names = list(dict.fromkeys(search_names))
    eng_name = orig.strip() if isinstance(orig, str) and orig.strip() else (full.strip() if isinstance(full, str) else "")
    return search_names, {
        "full": full if isinstance(full, str) else "",
        "common": common if isinstance(common, str) else "",
        "orig": orig if isinstance(orig, str) else "",
        "eng_name": eng_name,
        "alias": cell_to_list(row.get("别名")),
        "position": row.get("位置", ""),
    }

def host_of(url):
    if not url: return ""
    m = re.search(r"https?://([^/]+)/?", url)
    return m.group(1).lower() if m else ""

def is_blocked(url):
    return any(b in host_of(url) for b in BLOCK_DOMAINS)

def safe_url(url):
    if not isinstance(url, str): return ""
    url = url.strip()
    return url if url.startswith(("http://", "https://")) else ""

def parse_yyyymmdd(s):
    if not s: return 0
    digits = re.sub(r"\D", "", str(s))
    return int(digits[:8]) if len(digits) >= 8 else 0

def in_week(date_str):
    d = parse_yyyymmdd(date_str)
    if d == 0: return False
    return int(WEEK_START) <= d <= int(WEEK_END)

def title_has_old_year(title):
    ys = re.findall(r"(20\d{2})", title)
    return any(int(y) < CURRENT_YEAR for y in ys)

def is_review(title, text):
    blob = title + " " + text[:200]
    return any(k in blob for k in REVIEW_KW)

def name_in_title(name_list, title):
    return any(n in title for n in name_list if n)

def is_subject_match(name_list, title, text):
    if name_in_title(name_list, title):
        return True
    head = text[:120]
    return any(n in head for n in name_list if n)

# ========== 搜索 ==========
def search_news(query):
    out = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.news(query, region="nl-nl", timelimit="w", max_results=NEWS_MAX):
                url = safe_url(r.get("url"))
                if not url or is_blocked(url): continue
                title = (r.get("title") or "").strip()
                if not title: continue
                date_str = r.get("date", "")
                if not in_week(date_str):
                    print(f"    ⏩ 旧闻({date_str}): {title[:30]}")
                    continue
                body = r.get("body") or ""
                if is_review(title, body):
                    print(f"    🕘 复盘: {title[:30]}")
                    continue
                out.append({"title": title, "url": url, "text": body[:200],
                            "date": date_str, "lang": "auto"})
        print(f"  ✅ news {len(out)}")
    except Exception as e:
        print(f"  ❌ news: {e}")
    return out

def search_text_cn(query, name_list):
    out = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=TEXT_MAX):
                url = safe_url(r.get("href"))
                if not url or is_blocked(url): continue
                title = (r.get("title") or "").strip()
                if not title: continue
                body = r.get("body") or ""
                h = host_of(url)
                if not any(d in h for d in ["dongqiudi","hupu","zhibo8","sina.com.cn","qq.com","sohu.com","163.com"]):
                    continue
                if title_has_old_year(title):
                    print(f"    ⏩ 旧年: {title[:30]}")
                    continue
                if is_review(title, body):
                    print(f"    🕘 复盘: {title[:30]}")
                    continue
                if not is_subject_match(name_list, title, body):
                    print(f"    ❌ 误关联: {title[:30]}")
                    continue
                out.append({"title": title, "url": url, "text": body[:200],
                            "date": "", "lang": "cn"})
        print(f"  ✅ text {len(out)}")
    except Exception as e:
        print(f"  ❌ text: {e}")
    return out

def collect_items(search_names, alias_list):
    items = []; seen = set()
    all_n = list(dict.fromkeys(search_names + alias_list))
    for nm in all_n[:3]:
        if not nm: continue
        items += search_news(f"{nm} 荷兰队")
        time.sleep(1.2)
        items += search_news(f"{nm} footballer")
        time.sleep(1.2)
    cn = [n for n in all_n if any('一' <= c <= '鿿' for c in n)]
    for nm in cn[:2]:
        items += search_text_cn(f"{nm} 荷兰队", all_n)
        time.sleep(1.2)
    for it in items:
        if it["url"] not in seen:
            seen.add(it["url"])
    return [it for it in items if it["url"] in seen]

# ========== LLM ==========
def llm_filter(identity, items):
    if not items: return []
    hints = "、".join(filter(None, [identity["full"], identity["common"], identity["orig"], *identity["alias"]]))
    blocks = [f"[{i}] {'外文' if it['lang']=='auto' else '中文'} {it['title']}\n{it['text']}"
              for i, it in enumerate(items, 1)]
    prompt = f"""目标球员：{hints}
下面是本周内候选新闻。只选「本周新发生、且该球员是事件主体」的条。
规则：
- 旧事复盘/球员介绍/仅提及名字 → 不选
- 中文条：标题原样，摘要写 -
- 外文条：中文标题 + 120~180字摘要
输出：序号 | 中文标题 | 摘要
---
""" + "\n\n".join(blocks)
    try:
        resp = client.chat.completions.create(model=MODEL, messages=[{"role":"user","content":prompt}], temperature=0)
        raw = resp.choices[0].message.content
        out = []
        for line in raw.splitlines():
            line = line.strip()
            if "|" not in line: continue
            m = re.search(r"(\d+)", line)
            if not m: continue
            i = int(m.group(1)) - 1
            if not (0 <= i < len(items)): continue
            title = line.split("|")[1].strip()
            if not title: continue
            summary = line.split("|")[2].strip() if line.count("|")>=2 else "-"
            out.append({"item": items[i], "title": title, "summary": summary})
        return out[:MAX_PER_PERSON]
    except Exception as e:
        print(f"  ❌ llm: {e}"); return []

# ========== 飞书 ==========
def build_post(all_people):
    content = [[{"tag":"text","text":f"🇳🇱 {REPORT_TITLE}","style":"bold"}],
               [{"tag":"text","text":"\n"}]]
    any_news = False
    for name, news in all_people:
        if not news: continue
        any_news = True
        content.append([{"tag":"text","text":name,"style":"bold"}])
        for n in news:
            it = n["item"]
            para = [{"tag":"text","text":f"• {n['title']}\n"}]
            if it["lang"] == "auto" and n.get("summary") and n["summary"]!="-":
                para.append({"tag":"text","text":f"{n['summary']}\n"})
            if safe_url(it.get("url")):
                para.append({"tag":"a","text":"原文链接","href":it["url"]})
            content.append(para)
        content.append([{"tag":"text","text":"\n"}])  # 每人空一行
    if not any_news:
        content.append([{"tag":"text","text":"本周暂无符合条件的动态"}])
    return {"msg_type":"post","content":{"post":{"zh_cn":{"title":REPORT_TITLE,"content":content}}}}

def send_lark(payload):
    if not LARK_WEBHOOK:
        print("⚠️ 无 webhook"); return
    try:
        r = requests.post(LARK_WEBHOOK, json=payload, timeout=15)
        print(f"📤 {r.status_code} {r.text}")
    except Exception as e:
        print(f"📤 失败: {e}")

# ========== main ==========
def main():
    print("🚀")
    df = pd.read_csv(CSV_FILE, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    print(f"📋 {len(df)} 人")
    all_people = []
    for idx, row in df.iterrows():
        sn, idt = get_person(row)
        name = idt["full"] or idt["common"]
        print(f"\n=== [{idx+1}/{len(df)}] {name} ===")
        items = collect_items(sn, idt["alias"])
        picked = llm_filter(idt, items) if items else []
        print(f"  选中 {len(picked)}")
        all_people.append((name, picked))
        time.sleep(1)
    send_lark(build_post(all_people))
    print("🏁")

if __name__ == "__main__":
    main()
