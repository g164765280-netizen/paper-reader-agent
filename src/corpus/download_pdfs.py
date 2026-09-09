# -*- coding: utf-8 -*-
"""下载 120 篇论文全文 PDF（从 metadata_abstracts.jsonl 读 arxiv_id，带限流+退避+增量）。"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

UA = {"User-Agent": "rscd-fulltext/1.0 (research assistant)"}
CORPUS = Path("/media/sdb1/gzj/data/rscd/corpus")


def _get(url: str, retries: int = 4) -> bytes:
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (429, 503, 500) and a < retries - 1:
                time.sleep(12 * (a + 1))
                continue
            raise
        except Exception:
            if a < retries - 1:
                time.sleep(8)
                continue
            raise
    raise RuntimeError("unreachable")


def main():
    papers = [json.loads(l) for l in (CORPUS / "metadata_abstracts.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    pdf_dir = CORPUS / "pdf_fulltext"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    done = 0
    fail = []
    for i, p in enumerate(papers, 1):
        aid = p["arxiv_id"].replace("/", "_")
        dest = pdf_dir / f"{aid}.pdf"
        if dest.exists() and dest.stat().st_size > 10_000:
            done += 1
            continue
        url = f"https://arxiv.org/pdf/{p['arxiv_id']}.pdf"
        try:
            data = _get(url)
            if data[:4] != b"%PDF":
                raise RuntimeError("not pdf")
            dest.write_bytes(data)
            done += 1
            if i % 10 == 0 or i == len(papers):
                print(f"[{i}/{len(papers)}] {aid} {dest.stat().st_size//1024}KB ok", flush=True)
        except Exception as ex:
            fail.append({"arxiv_id": p["arxiv_id"], "err": str(ex)[:120]})
            print(f"[{i}/{len(papers)}] {aid} FAIL {ex}", flush=True)
        time.sleep(3)

    (CORPUS / "pdf_fulltext_report.json").write_text(
        json.dumps({"total": len(papers), "ok": done, "failed": fail}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"DONE ok={done}/{len(papers)}", flush=True)


if __name__ == "__main__":
    main()
