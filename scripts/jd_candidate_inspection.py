"""Inspect a fixed set of public JD help pages before adding them to the corpus.

This command only reads public pages and prints a short content audit. It does
not change the knowledge base or create files.
"""

from concurrent.futures import ThreadPoolExecutor
import re
import sys

import requests
from bs4 import BeautifulSoup


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CANDIDATES = {
    "品类售后细则": [
        "325-914", "325-915", "325-919", "325-920", "325-921",
        "325-922", "325-924", "325-925", "325-926", "325-927",
        "325-928",
    ],
    "退换货办理": [
        "126-300", "126-412", "127-782", "118-265", "118-544",
        "118-4278", "119-463",
    ],
    "退款及运费": [
        "130-262", "130-263", "130-546", "130-978", "131-494",
        "130-481", "130-485", "109-188",
    ],
    "物流及发票": [
        "102-250", "103-983", "502-974", "502-247",
    ],
}


def inspect(group: str, article_id: str) -> str:
    url = f"https://help.jd.com/user/issue/{article_id}.html"
    try:
        response = requests.get(url, timeout=25)
        response.raise_for_status()
        body = BeautifulSoup(response.text, "html.parser").select_one(
            "#pdfContainer .contxt"
        )
        if body is None:
            return f"{group}\t{article_id}\tNO_BODY\t{url}"
        title_node = body.select_one(".help-tit1")
        title = title_node.get_text(" ", strip=True) if title_node else "NO_TITLE"
        text = body.get_text(" ", strip=True)
        dates = re.findall(
            r"20\d{2}[-./年]\d{1,2}(?:[-./月]\d{1,2})?日?", text[-120:]
        )
        date = dates[-1] if dates else "unknown"
        return (
            f"{group}\t{article_id}\t{title}\tchars={len(text)}"
            f"\timages={len(body.select('img'))}"
            f"\ttables={len(body.select('table'))}"
            f"\tpage_date={date}"
        )
    except Exception as exc:
        return f"{group}\t{article_id}\tERROR={type(exc).__name__}: {exc}"


def main() -> None:
    items = [
        (group, article_id)
        for group, article_ids in CANDIDATES.items()
        for article_id in article_ids
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for result in pool.map(lambda item: inspect(*item), items):
            print(result)
    print(f"TOTAL_CANDIDATES={len(items)}")


if __name__ == "__main__":
    main()
