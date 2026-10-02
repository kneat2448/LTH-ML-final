#!/bin/bash
# Build the final report (PDF + DOCX) and the presentation from results/.
# Needs: pandoc, weasyprint (pip), python-pptx (pip). Run from anywhere.
set -e
cd "$(dirname "$0")/.."
python -m analysis.plots > /dev/null
python -m analysis.final_figs > /dev/null
python report/make_deck.py
cd report
pandoc FINAL_REPORT.md -s --css report.css --embed-resources --metadata lang=en -o FINAL_REPORT.html
python -c "import weasyprint; weasyprint.HTML('FINAL_REPORT.html').write_pdf('FINAL_REPORT.pdf')"
pandoc FINAL_REPORT.md -o FINAL_REPORT.docx --resource-path=.
rm -f FINAL_REPORT.html
ls -la FINAL_REPORT.pdf FINAL_REPORT.docx LTH_Warmup_Final.pptx
