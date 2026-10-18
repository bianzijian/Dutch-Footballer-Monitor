import os
import time
import requests
import pandas as pd
from openai import OpenAI

CSV_FILE = "Dutch-National-Football-Team-Coach&Player-since-1994.csv"

LARK_WEBHOOK = os.environ.get("LARK_WEBHOOK")
API_KEY = os.environ.get("OPENAI_API_KEY")
BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
MODEL = os.environ.get("MODEL_NAME", "gpt-4o-mini")

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

def search_weibo(keyword):
    try:
        r = requests.get("https://m.weibo.cn/api/container/getIndex", params={"containerid": f"100103type=1&q={keyword}", "page_type": "searchall"}, headers=HEADERS, timeout=10)
        data = r.json()
        out = []
        for card in data.get("data",{}).get("cards",[]):
            txt = card.get("mblog",{}).get("text","")
            if txt: out.append({"platform":"微博","text":txt[:200]})
        return out
    except: return []

def search_web(query):
    try:
        r = requests.post("https://html.duckduckgo.com/html/", data={"q":query}, headers=HEADERS, timeout=10)
        snips = []
        for line in r.text.split("<a")[100:140]:
            if "result__snippet" in line:
                t = line.split(">",1)[-1].split("<",1)[0].strip()
                if t: snips.append(t[:200])
        return [{"platform":"网页","text":s} for s in snips[:5]]
    except: return []

def collect_items(search_names, identity):
    items, seen = [], set()
    for nm in search_names[:2]:
        items += search_weibo(nm)
        items += search_web(f"{nm} 荷兰队")
        time.sleep(1.5)
    dedup = []
    for it in items:
        k = it["text"][:50]
        if k not in seen: seen.add(k); dedup.append(it)
    return dedup

def llm_filter(identity, items):
    if not items: return []
    name_hints = "、".join(filter(None,[identity["full"],identity["common"],identity["orig"],*identity["alias"],*identity["nick"]]))
    ctx = []
    if identity["position"]: ctx.append(f"位置：{identity['position']}")
    if identity["wc"]: ctx.append(f"世界杯：{identity['wc']}")
    if identity["ec"]: ctx.append(f"欧洲杯：{identity['ec']}")
    ctx_line = "；".join(ctx) or "荷兰足球人物"
    numbered = "\n".join(f"{i+1}. [{it['platform']}] {it['text']}" for i,it in enumerate(items))
    prompt = f"目标人物称呼：{name_hints}\n背景：{ctx_line}\n\n只保留确实在讲此人的动态/新闻。同名不同人、非荷兰足球、纯广告抽奖不要。\n\n输出：\n序号 | 平台 | 一句话摘要\n\n{numbered}"
    try:
        resp = client.chat.completions.create(model=MODEL, messages=[{"role":"user","content":prompt}], temperature=0)
        return [l.strip() for l in resp.choices[0].message.content.splitlines() if "|" in l]
    except: return []

def send_lark(title, lines):
    text = f"球员周报 | {title}\n\n" + "\n".join(lines)
    requests.post(LARK_WEBHOOK, json={"msg_type":"text","content":{"text":text[:18000]}})

def main():
    df = pd.read_csv(CSV_FILE, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    df = df.head(5)
    for _, row in df.iterrows():
        search_names, identity = get_person(row)
        if not search_names: continue
        items = collect_items(search_names, identity)
        res = llm_filter(identity, items)
        if res: send_lark(identity["full"] or identity["common"], res)
        time.sleep(3)

if __name__ == "__main__":
    main()