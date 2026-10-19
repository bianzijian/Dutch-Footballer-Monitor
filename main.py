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
MODE = os.environ.get("MODE", "build")

LARK_WEBHOOK = os.environ.get("LARK_WEBHOOK")
API_KEY = os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL = os.environ.get("MODEL_NAME", "gpt-4o-mini")

_today = datetime.now()
_wd = _today.weekday()
if _wd == 0:
    WEEK_END = (_today - timedelta(days=1)).strftime("%Y%m%d")
    WEEK_START = (_today - timedelta(days=7)).strftime("%Y%m%d")
else:
    WEEK_END = _today.strftime("%Y%m%d")
    WEEK_START = (_today - timedelta(days=7)).strftime("%Y%m%d")

CURRENT_YEAR = _today.year
REPORT_TITLE = f"{WEEK_START}-{WEEK_END} 荷兰国家队历史球员资讯"
print(f"📅 区间: {WEEK_START}~{WEEK_END} | MODE={MODE}")

BLOCK_DOMAINS = ["yahoo.com", "msn.com", "consent.", "privacy"]
MAX_PER_PERSON = 3
NEWS_MAX = 5
TEXT_MAX = 5

REVIEW_KW = ["回顾","复盘","专访","谈当年","谈20","回忆","昔日","当年",
             "旧将","名宿","传奇","诈伤","曾效力","曾执教","退役",
             "那届世界杯","那届欧洲杯","球员介绍","生涯回顾"]

# ========== 工具 ==========
def cell_to_list(v):
    if not isinstance(v, str): return []
    return [p.strip() for p in v.replace("；","/").replace(";","/").split("/") if p.strip()]

def get_person(row):
    full = row.get("全名"); common = row.get("常用名"); orig = row.get("原名")
    sn = []
    for n in [common, full]:
        if isinstance(n, str) and n.strip(): sn.append(n.strip())
    if isinstance(orig, str) and orig.strip():
        sn.append(orig.strip().split()[-1])
    sn = list(dict.fromkeys(sn))
    return sn, {
        "full": full if isinstance(full, str) else "",
        "common": common if isinstance(common, str) else "",
        "orig": orig if isinstance(orig, str) else "",
        "alias": cell_to_list(row.get("别名")),
    }

def host_of(u):
    if not u: return ""
    m = re.search(r"https?://([^/]+)/?", u)
    return m.group(1).lower() if m else ""

def is_blocked(u):
    return any(b in host_of(u) for b in BLOCK_DOMAINS)

def safe_url(u):
    if not isinstance(u, str): return ""
    u = u.strip()
    return u if u.startswith(("http://","https://")) else ""

def parse_yyyymmdd(s):
    if not s: return 0
    d = re.sub(r"\D","",str(s))
    return int(d[:8]) if len(d) >= 8 else 0

def in_week(ds):
    d = parse_yyyymmdd(ds)
    if d == 0: return False
    return int(WEEK_START) <= d <= int(WEEK_END)

def title_has_old_year(t):
    for y in re.findall(r"(20\d{2})", t):
        if int(y) < CURRENT_YEAR:
            return True
    return False

def is_review(t, body):
    return any(k in (t + " " + body[:200]) for k in REVIEW_KW)

def is_subject(names, t, body):
    if any(n in t for n in names if n): return True
    return any(n in body[:120] for n in names if n)

# ========== 搜索 ==========
def search_news(q):
    out = []
    try:
        with DDGS() as d:
            for r in d.news(q, region="nl-nl", timelimit="w", max_results=NEWS_MAX):
                url = safe_url(r.get("url"))
                if not url or is_blocked(url): continue
                t = (r.get("title") or "").strip()
                if not t: continue
                ds = r.get("date","")
                if not in_week(ds):
                    continue
                body = r.get("body") or ""
                if is_review(t, body): continue
                out.append({"title":t,"url":url,"text":body[:200],"lang":"auto"})
    except Exception as e:
        print("  ❌ news", e)
    return out

def search_text_cn(q, names):
    out = []
    try:
        with DDGS() as d:
            for r in d.text(q, max_results=TEXT_MAX):
                url = safe_url(r.get("href"))
                if not url or is_blocked(url): continue
                t = (r.get("title") or "").strip()
                if not t: continue
                body = r.get("body") or ""
                h = host_of(url)
                if not any(x in h for x in ["dongqiudi","hupu","zhibo8","sina.com.cn","qq.com","sohu.com","163.com"]):
                    continue
                if title_has_old_year(t): continue
                if is_review(t, body): continue
                if not is_subject(names, t, body): continue
                out.append({"title":t,"url":url,"text":body[:200],"lang":"cn"})
    except Exception as e:
        print("  ❌ text", e)
    return out

def collect(sn, alias):
    items = []; seen = set()
    alln = list(dict.fromkeys(sn + alias))
    for nm in alln[:3]:
        if not nm: continue
        items += search_news(f"{nm} 荷兰队"); time.sleep(1.2)
        items += search_news(f"{nm} footballer"); time.sleep(1.2)
    cn = [n for n in alln if any('一'<=c<='鿿' for c in n)]
    for nm in cn[:2]:
        items += search_text_cn(f"{nm} 荷兰队", alln); time.sleep(1.2)
    for it in items:
        if it["url"] not in seen: seen.add(it["url"])
    return [it for it in items if it["url"] in seen]

# ========== LLM ==========
def llm_filter(idt, items):
    if not items: return []
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    hints = "、".join(filter(None,[idt["full"],idt["common"],idt["orig"],*idt["alias"]]))
    blocks = [f"[{i}] {'外文' if it['lang']=='auto' else '中文'} {it['title']}\n{it['text']}"
              for i,it in enumerate(items,1)]
    prompt = f"""目标球员：{hints}
只选「本周内新发生、该球员是事件主体」的新闻。
规则：
- 旧事复盘/球员介绍/仅提及名字 → 不选
- 中文新闻：输出 序号###原文标题###-
- 外文新闻：输出 序号###中文标题###中文摘要(120~180字)
- 外文条必须写摘要，不写摘要等于不选
- 输出格式每行一条：序号###标题###摘要
不要解释，不要空行
===
""" + "\n\n".join(blocks)

    try:
        resp = client.chat.completions.create(
            model=MODEL, messages=[{"role":"user","content":prompt}], temperature=0)
        raw = resp.choices[0].message.content
        out = []
        for line in raw.splitlines():
            line = line.strip()
            if "###" not in line: continue
            m = re.match(r"(\d+)\s*###\s*(.+?)\s*###\s*(.*)", line)
            if not m: continue
            i = int(m.group(1)) - 1
            if not (0 <= i < len(items)): continue
            title = m.group(2).strip()
            summary = m.group(3).strip()
            if not title: continue
            it = items[i]
            # 外文必须有摘要
            if it["lang"] == "auto":
                if not summary or summary == "-":
                    continue
            else:
                summary = "-"
            out.append({"item":it,"title":title,"summary":summary})
        return out[:MAX_PER_PERSON]
    except Exception as e:
        print("  ❌ llm", e)
        return []

# ========== 飞书 ==========
def build_post(people):
    content = [
        [{"tag":"text","text":f"🇳🇱 {REPORT_TITLE}","style":"bold"}],
        [{"tag":"text","text":"\n"}]
    ]
    any_news = False
    for name, news in people:
        if not news: continue
        any_news = True
        content.append([{"tag":"text","text":name,"style":"bold"}])
        for n in news:
            it = n["item"]
            para = [{"tag":"text","text":f"• {n['title']}\n"}]
            if it["lang"]=="auto" and n.get("summary") and n["summary"]!="-":
                para.append({"tag":"text","text":f"{n['summary']}\n"})
            if safe_url(it.get("url")):
                para.append({"tag":"a","text":"原文链接","href":it["url"]})
            content.append(para)
        content.append([{"tag":"text","text":"\n"}])
    if not any_news:
        content.append([{"tag":"text","text":"本周暂无符合条件的动态"}])
    return {"msg_type":"post","content":{"post":{"zh_cn":{"title":REPORT_TITLE,"content":content}}}}

def main():
    if MODE == "send":
        send_from_file(); return

    print("🚀 build")
    df = pd.read_csv(CSV_FILE, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    people = []
    for idx,row in df.iterrows():
        sn,idt = get_person(row)
        name = idt["full"] or idt["common"]
        print(f"\n=== [{idx+1}/{len(df)}] {name} ===")
        items = collect(sn, idt["alias"])
        picked = llm_filter(idt, items) if items else []
        print(f"  选中 {len(picked)}")
        people.append((name, picked))
        time.sleep(1)

    with open("report.json","w",encoding="utf-8") as f:
        json.dump(build_post(people), f, ensure_ascii=False)
    print("💾 report.json 已生成")

def send_from_file():
    if not os.path.exists("report.json"):
        print("⚠️ 没 report.json"); return
    with open("report.json","r",encoding="utf-8") as f:
        payload = json.load(f)
    if not LARK_WEBHOOK:
        print("⚠️ 无 webhook"); return
    try:
        r = requests.post(LARK_WEBHOOK, json=payload, timeout=15)
        print(f"📤 {r.status_code} {r.text}")
    except Exception as e:
        print("📤 失败", e)

if __name__ == "__main__":
    main()
