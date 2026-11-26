"""Build the KTN paper draft PDF from the Markdown draft via WeasyPrint."""
import re

import markdown
from pathlib import Path

SRC = Path("/home/hatch/workspace/your_files/ktn-paper-draft.md")
OUT_HTML = Path("/home/hatch/workspace/your_files/ktn-paper-draft.html")
OUT_PDF = Path("/home/hatch/workspace/your_files/ktn-paper-draft.pdf")

md = SRC.read_text(encoding="utf-8")
body = markdown.markdown(md, extensions=["fenced_code", "tables"])

# Keep each figure image together with its caption on one page.
body, n_wrapped = re.subn(
    r'(<p><img[^>]*></p>\s*<p><em>Figure.*?</em></p>)',
    r'<div class="figwrap">\1</div>',
    body, flags=re.DOTALL)
print("wrapped figures:", n_wrapped)

# Set the references in smaller, tighter type so the whole block sits on one page.
body, n_refs = re.subn(
    r'(<h2>References</h2>.*)',
    r'<div class="refs">\1</div>',
    body, flags=re.DOTALL)
print("wrapped references:", n_refs)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>The First Multi-State Neuron: The Knowledge Time Neuron</title>
<style>
@page {{
  size: Letter;
  margin: 2.5cm 2.5cm 2.5cm 2.5cm;
  @bottom-center {{ content: counter(page); font-size: 9pt; color: #666; }}
}}
body {{
  font-family: "Liberation Serif", "DejaVu Serif", Georgia, serif;
  font-size: 11pt;
  line-height: 1.55;
  color: #111;
}}
h1 {{ font-size: 17pt; line-height: 1.3; margin: 0 0 6pt 0; }}
h2 {{ font-size: 13pt; margin: 18pt 0 6pt 0; page-break-after: avoid; }}
h3 {{ font-size: 11.5pt; margin: 14pt 0 4pt 0; page-break-after: avoid; }}
p {{ margin: 0 0 8pt 0; text-align: justify; }}
ul, ol {{ margin: 0 0 8pt 0; }}
li {{ margin: 0 0 4pt 0; }}
table {{ border-collapse: collapse; margin: 0 0 10pt 0; font-size: 10pt; }}
th, td {{ border: 1px solid #999; padding: 4pt 6pt; text-align: left; }}
th {{ background: #eee; }}
pre {{
  font-family: "Liberation Mono", "DejaVu Sans Mono", monospace;
  font-size: 9pt;
  line-height: 1.4;
  background: #f6f6f6;
  border: 1px solid #ddd;
  padding: 8pt 10pt;
  white-space: pre-wrap;
  word-wrap: break-word;
  page-break-inside: avoid;
}}
code {{ font-family: "Liberation Mono", "DejaVu Sans Mono", monospace; font-size: 9.5pt; }}
a {{ color: #111; }}
img {{ max-width: 100%; height: auto; }}
.figwrap {{ page-break-inside: avoid; margin: 10pt 0; }}
.figwrap p {{ text-align: left; }}
.refs {{ font-size: 10pt; line-height: 1.4; }}
.refs p {{ margin: 0 0 5pt 0; text-align: left; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""

OUT_HTML.write_text(html, encoding="utf-8")
print("wrote", OUT_HTML)

from weasyprint import HTML
HTML(filename=str(OUT_HTML)).write_pdf(str(OUT_PDF))
print("wrote", OUT_PDF, OUT_PDF.stat().st_size, "bytes")
