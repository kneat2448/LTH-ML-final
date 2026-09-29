"""Rewrite the results table in Methodology.md (section 10) from results/<condition>/seed<k>/round*.json.

Usage: python -m analysis.methodology_table
"""
import json
import re
from pathlib import Path

from src.metrics import stability_summary

ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ["low", "high", "warm03", "high_warm"]
START, END = "<!-- results-table:start -->", "<!-- results-table:end -->"


def rows() -> list[str]:
    """One markdown row per finished training, in condition / seed / round / variant order."""
    out = []
    for cond in CONDITIONS:
        for f in sorted((ROOT / "results" / cond).glob("seed*/round*.json"),
                        key=lambda p: (p.parent.name, int(re.search(r"round(\d+)", p.name).group(1)), p.name)):
            d = json.loads(f.read_text())
            name = "ticket (dense)" if d["round"] == 0 else d["variant"]
            st = stability_summary(d["sharpness"])
            out.append(f"| {d['condition']} | {d['seed']} | {d['round']} | {100 * d['remaining']:.1f} | {name} "
                       f"| {100 * d['test']['acc']:.2f}% | {st['S0']:.3f} | {st['max_S_train']:.3f} | {d['R02']:.3f} "
                       f"| {'yes' if d['diverged'] else 'no'} |")
    return out


def main() -> None:
    path = ROOT / "Methodology.md"
    text = path.read_text()
    header = ["| Condition | Seed | Round | % remaining | Variant | Test acc | S(0) | max S(25–3k) | R_0.2 | Diverged |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    table = "\n".join([START, *header, *rows(), END])
    text = re.sub(re.escape(START) + ".*?" + re.escape(END), table, text, flags=re.S)
    path.write_text(text)


if __name__ == "__main__":
    main()
