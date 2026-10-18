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

# 自动计算最近 7 天区间
_today = datetime.now()
_end = _today.strftime("%Y%m%d")
_start = (_today - timedelta(days=7)).strftime("%Y%m%d")
WEEK_START = _start
WEEK_END = _end
CURRENT_YEAR = str(_today.year)
REPORT_TITLE = f"{WEEK_START}-{WEEK_END} 荷兰国家队历史球员资讯"

print(f"📅 本周区间: {WEEK_START} ~ {WEEK_END}")

BLOCK_DOMAINS = ["yahoo.com", "msn.com", "consent.", "privacy"]
MAX_PER_PERSON = 3

# 旧事复盘关键词
REVIEW_KEYWORDS = ["回顾", "复盘", "专访", "谈当年", "谈20", "回忆", "昔日", "当年", "旧将", "名宿", "传奇", "缅怀"]

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
    return any(b in host_of(url) for b in BLOCK_DOMAINS)

def safe_url(url):
    if not isinstance(url, str):
        return ""
    url = url.strip()
    if url.startswith("http://") or url.startswith("https://"):
        return url
    return ""

def parse_date_to_yyyymmdd(date_str):
    """把 DDG news 的 date 字段转成 YYYYMMDD 整数"""
    if not date_str:
        return 0
    s = str(date_str).replace("-", "").replace(":", "").replace("T", "")
    digits = re.sub(r"\D", "", s)
    if len(digits) >= 8:
        return int(digits[:8])
    return 0

def in_week_range(date_str):
    """判断日期是否在本周区间内"""
    d = parse_date_to_yyyymmdd(date_str)
    if d == 0:
        return False
    try:
        start_i = int(WEEK_START)
        end_i = int(WEEK_END)
    except:
        return False
    return start_i <= d <= end_i

def has_old_year(text):
    """检查文本中是否包含旧年份（非当前年份的4位数年份）"""
    years = re.findall(r"\b(20\d{2})\b", text)
    for y in years:
        if y != CURRENT_YEAR:
            return True
    return False

def is_review_article(title, text):
    """判断是否为旧事复盘文章"""
    combined = title + " " + text[:200]
    return any(kw in combined for kw in REVIEW_KEYWORDS)

# ========== 搜索 ==========
def search_news(query):
    """news 搜索，硬过滤日期"""
    out = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.news(query, region="nl-nl", timelimit="w", max_results=8):
                url = safe_url(r.get("url"))
                if not url or is_blocked(url):
                    continue
                title = (r.get("title") or "").strip()
                if not title:
                    continue
                date_str = r.get("date", "")
                # 日期过滤：不在本周区间内直接丢
                if not in_week_range(date_str):
                    print(f"    ⏩ 旧闻跳过 ({date_str}): {title[:40]}")
                    continue
                # 旧事复盘检查
                if is_review_article(title, r.get("body") or ""):
                    print(f"    🕘 旧事复盘跳过: {title[:40]}")
                    continue
                out.append({
                    "title": title,
                    "url": url,
                    "text": (r.get("body") or "")[:300],
                    "source": (r.get("source") or "").lower(),
                    "date": date_str,
                    "lang": "auto",
                })
        print(f"  ✅ news 本周内 {len(out)} 条")
    except Exception as e:
        print(f"  ❌ news 失败: {e}")
    return out

def search_text_cn(query):
    """中文源 text 搜索，带年份过滤"""
    out = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=8):
                url = safe_url(r.get("href"))
                if not url or is_blocked(url):
                    continue
                title = (r.get("title") or "").strip()
                if not title:
                    continue
                body = r.get("body") or ""
                # 中文源检查
                h = host_of(url)
                cn_domains = ["dongqiudi", "hupu", "zhibo8", "sina.com.cn", "qq.com", "sohu.com", "163.com", "toutiao.com"]
                if not any(d in h for d in cn_domains):
                    continue
                # 年份过滤：标题/正文含旧年份直接丢
                if has_old_year(title + " " + body[:200]):
                    print(f"    ⏩ 旧年份跳过: {title[:40]}")
                    continue
                # 旧事复盘检查
                if is_review_article(title, body):
                    print(f"    🕘 旧事复盘跳过: {title[:40]}")
                    continue
                out.append({
                    "title": title,
                    "url": url,
                    "text": body[:300],
                    "source": h,
                    "date": "",
                    "lang": "cn",
                })
        print(f"  ✅ text 中文 {len(out)} 条")
    except Exception as e:
        print(f"  ❌ text 失败: {e}")
    return out

def collect_items(search_names, eng_name, alias_list):
    """收集候选新闻：别名全进搜索"""
    items = []
    seen = set()

    # news 搜索：英文名 + 常用名 + 别名
    all_names = list(dict.fromkeys(search_names + alias_list))  # 去重保序
    for nm in all_names[:4]:  # 最多4个搜索词防太多
        if nm:
            items += search_news(f"{nm} 荷兰队")
            time.sleep(1.5)
            items += search_news(f"{nm} footballer")
            time.sleep(1.5)

    # 中文 text 兜底：中文名 + 别名
    cn_names = [n for n in all_names if any('\u4e00' <= c <= '\u9fff' for c in n)]
    for nm in cn_names[:3]:
        items += search_text_cn(f"{nm} 荷兰队")
        time.sleep(1.5)

    # 去重
    dedup = []
    for it in items:
        if it["url"] not in seen:
            seen.add(it["url"])
            dedup.append(it)
    return dedup

# ========== LLM：合并 + 翻译 + 排除旧事复盘 ==========
def llm_group_and_translate(identity, items):
    if not items:
        return []
    name_hints = "、".join(filter(None, [identity["full"], identity["common"], identity["orig"], *identity["alias"]]))
    blocks = []
    for i, it in enumerate(items, 1):
        blocks.append(f"[{i}] ({'外文' if it['lang']=='auto' else '中文'}) {it['title']}\n{it['text']}")
    joined = "\n\n".join(blocks)

    prompt = f"""目标人物：{name_hints}
以下是已经过滤好的「本周内」候选新闻（近7天发布）。

请做三件事：
1. 把「同一件事」归到一组，每组只留最该保留的 1 条（中文优先；多条中文选最权威；无中文选最权威外文）
2. 对被选中的条目：
   - 中文条目：只输出中文标题，摘要写 -
   - 外文条目：输出中文标题 + 150-200字摘要（包含事件经过、当事人角色、背景、后续）
3. 【重要】如果某条新闻虽然本周发布，但内容是回顾旧事、复盘旧比赛、旧专访等（非本周新发生的事件），不要选中它。旧事复盘不发。

输出格式（每行一条，最多 {MAX_PER_PERSON} 条）：
序号 | 中文标题 | 摘要
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
            m = re.search(r"(\d+)", line)
            if not m:
                continue
            i = int(m.group(1)) - 1
            if not (0 <= i < len(items)):
                continue
            parts = [p.strip() for p in line.split("|")]
            title = parts[1] if len(parts) > 1 else ""
            title = title.strip()
            if not title:
                continue
            summary = parts[2].strip() if len(parts) > 2 else "-"
            parsed.append({
                "item": items[i],
                "title": title,
                "summary": summary,
            })
        return parsed[:MAX_PER_PERSON]
    except Exception as e:
        print(f"  ❌ LLM 失败: {e}")
        return []

# ========== 飞书拼接 ==========
def build_post(all_people):
    content = []
    content.append([{"tag": "text", "text": f"🇳🇱 {REPORT_TITLE}", "style": "bold"}])
    content.append([{"tag": "text", "text": f"\n"}])

    any_news = False
    for name, news_list in all_people:
        if not news_list:
            continue
        any_news = True
        content.append([{"tag": "text", "text": name, "style": "bold"}])

        for n in news_list:
            it = n["item"]
            title = n.get("title") or ""
            if not title:
                continue
            para = [{"tag": "text", "text": f"• {title}\n"}]
            if it["lang"] == "auto":
                summary = n.get("summary", "")
                if summary and summary != "-":
                    para.append({"tag": "text", "text": f"{summary}\n"})
            url = safe_url(it.get("url"))
            if url:
                para.append({"tag": "a", "text": "原文链接", "href": url})
            content.append(para)

    if not any_news:
        content.append([{"tag": "text", "text": "本周暂无符合条件的动态（近7天内无新闻）"}])

    return {
        "msg_type": "post",
        "content": {
            "post": {
                "zh_cn": {
                    "title": REPORT_TITLE,
                    "content": content,
                }
            }
        },
    }

def send_lark(payload):
    if not LARK_WEBHOOK:
        print("⚠️ 没配 webhook")
        return
    print("📦 payload 预览（前1500字）:")
    print(json.dumps(payload, ensure_ascii=False)[:1500])

    try:
        r = requests.post(LARK_WEBHOOK, json=payload, timeout=15)
        print(f"📤 状态码: {r.status_code}")
        print(f"📤 响应体: {r.text}")
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
        print(f"  搜索名: {search_names}")
        print(f"  别名: {identity['alias']}")

        items = collect_items(search_names, identity["eng_name"], identity["alias"])
        if not items:
            print("  本周无候选（近7天内没搜到）")
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
