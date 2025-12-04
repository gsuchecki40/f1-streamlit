#!/usr/bin/env python3
"""Create a self-contained HTML by inlining images as base64 and embedding local HTML snippets.

Reads: artifacts/models_report_with_locals.html
Writes: artifacts/models_report_selfcontained.html

This script uses only Python stdlib and produces a single HTML file suitable for sharing.
"""
from pathlib import Path
import base64
import mimetypes
import re

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / 'artifacts' / 'models_report_with_locals.html'
OUT = ROOT / 'artifacts' / 'models_report_selfcontained.html'


def inline_images(html, root):
    """Replace local image src references with data: URIs."""

    def repl(m):
        q = m.group('q')
        src = m.group('src')
        if src.startswith(('http://', 'https://', 'data:')):
            return m.group(0)
        p = (root / src).resolve()
        if not p.exists():
            print('warning: image not found', p)
            return m.group(0)
        mime, _ = mimetypes.guess_type(str(p))
        if mime is None:
            mime = 'application/octet-stream'
        data = base64.b64encode(p.read_bytes()).decode('ascii')
        return f"src={q}data:{mime};base64,{data}{q}"

    return re.sub(r"src=(?P<q>['\"])(?P<src>[^'\"]+)(?P=q)", repl, html)


def embed_local_html(html, root):
    """Embed local HTML pages (links under artifacts/) inline inside <details> tags."""

    def repl(m):
        href = m.group('href')
        label = m.group('label').strip()
        if not href.startswith('artifacts/'):
            return m.group(0)
        p = (root / href).resolve()
        if not p.exists():
            print('warning: local html not found', p)
            return m.group(0)
        content = p.read_text(encoding='utf-8')
        body = re.search(r"<body[^>]*>(?P<body>.*?)</body>", content, re.IGNORECASE | re.DOTALL)
        content = body.group('body') if body else content
        content = content.strip()
        return f"<li><details class='embedded-local'><summary>{label}</summary><div>{content}</div></details></li>"

    return re.sub(
        r"<li>\s*<a[^>]+href=['\"](?P<href>[^'\"]+)['\"][^>]*>(?P<label>.*?)</a>\s*</li>",
        repl,
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )


def main():
    if not IN.exists():
        print('input missing:', IN)
        raise SystemExit(2)
    html = IN.read_text(encoding='utf-8')
    html = inline_images(html, ROOT)
    html = embed_local_html(html, ROOT)
    OUT.write_text('<!-- self-contained -->\n' + html, encoding='utf-8')
    print('wrote', OUT)


if __name__ == '__main__':
    main()
