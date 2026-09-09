# -*- coding: utf-8 -*-
"""只抓摘要（不下载 PDF），快速建立语料元数据，供摘要级 RAG/评测使用。"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

UA = {"User-Agent": "rscd-corpus/1.0 (research assistant)"}
ARXIV_API = "https://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom"}

KEYWORDS = [
    "remote sensing change detection",
    "building change detection deep learning",
    "semantic change detection remote sensing",
    "bitemporal change detection",
    "change detection transformer",
    "change detection siamese network",
    "change detection mamba",
    "change detection diffusion model",
    "change detection foundation model",
    "remote sensing change detection dataset",
    "cropland change detection",
    "change detection weakly supervised",
]


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (s or "").lower()).strip()


def _get(url: str, retries: int = 4) -> bytes:
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 503, 500) and a < retries - 1:
                time.sleep(10 * (a + 1))
                continue
            raise
        except Exception:
            if a < retries - 1:
                time.sleep(8)
                continue
            raise
    raise RuntimeError("unreachable")


def parse(xml_text: str) -> list[dict]:
    root = ET.fromstring(xml_text)
    out = []
    for e in root.findall("a:entry", NS):
        eid = e.findtext("a:id", default="", namespaces=NS)
        out.append({
            "arxiv_id": eid.split("/abs/")[-1] if "/abs/" in eid else "",
            "title": re.sub(r"\s+", " ", e.findtext("a:title", default="", namespaces=NS)).strip(),
            "abstract": re.sub(r"\s+", " ", e.findtext("a:summary", default="", namespaces=NS)).strip(),
            "year": e.findtext("a:published", default="", namespaces=NS)[:4],
        })
    return out


def search(keyword: str, start: int, n: int = 50) -> list[dict]:
    url = ARXIV_API + "?" + urllib.parse.urlencode({
        "search_query": f'all:"{keyword}"', "start": start, "max_results": n, "sortBy": "relevance"})
    return parse(_get(url).decode("utf-8"))


def search_title(title: str) -> list[dict]:
    url = ARXIV_API + "?" + urllib.parse.urlencode({
        "search_query": f'ti:"{title[:200]}"', "max_results": 3})
    return parse(_get(url).decode("utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--manifest", default="")
    ap.add_argument("--target", type=int, default=120)
    args = ap.parse_args()

    seen: dict[str, dict] = {}
    for kw in KEYWORDS:
        if len(seen) >= args.target:
            break
        for start in (0, 50):
            if len(seen) >= args.target:
                break
            try:
                for e in search(kw, start):
                    if e["arxiv_id"] and e["arxiv_id"] not in seen:
                        seen[e["arxiv_id"]] = e
            except Exception as ex:
                print(f"search failed {kw}@{start}: {ex}", flush=True)
            print(f"  [{kw}] unique={len(seen)}", flush=True)
            time.sleep(4)

    if args.manifest and Path(args.manifest).exists() and len(seen) < args.target:
        for p in json.loads(Path(args.manifest).read_text(encoding="utf-8"))["papers"]:
            if len(seen) >= args.target:
                break
            try:
                es = search_title(p["title"])
                nt = norm(p["title"])
                if es:
                    best = max(es, key=lambda e: (1.0 if norm(e["title"]) == nt else 0.5 if nt in norm(e["title"]) else 0))
                    if best["arxiv_id"] and best["arxiv_id"] not in seen:
                        best["aka"] = p.get("aka", "")
                        seen[best["arxiv_id"]] = best
            except Exception as ex:
                print(f"seed failed {p['title'][:40]}: {ex}", flush=True)
            time.sleep(4)

    papers = list(seen.values())
    out = Path(args.out) / "metadata_abstracts.jsonl"
    out.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in papers), encoding="utf-8")
    print(f"DONE: {len(papers)} abstracts saved to {out}", flush=True)


if __name__ == "__main__":
    main()
