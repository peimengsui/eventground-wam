"""Verify retained sources, reporting artifacts and an optional sealed package."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
from html.parser import HTMLParser
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
from urllib.parse import unquote, urlsplit
import xml.etree.ElementTree as ET

from PIL import Image
from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []
        self.ids = set()
        self.images = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if 'id' in attrs:
            assert attrs['id'] not in self.ids, f'Duplicate HTML ID: {attrs["id"]}'
            self.ids.add(attrs['id'])
        for key in ('href', 'src'):
            if key in attrs:
                self.targets.append(attrs[key])
        if tag == 'img':
            assert attrs.get('alt'), 'Image missing alternative text'
            self.images.append(attrs['src'])
        assert tag not in ('script', 'iframe'), 'Static gallery must not execute network content'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--write-report', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    index_path = out / 'package-index.json'
    if args.write_report and index_path.exists():
        raise ValueError('Refusing to rewrite verification inside a sealed package')

    manifest = read(out / 'source-manifest.json')['files_sha256']
    for name, expected in manifest.items():
        assert digest(ROOT / name) == expected, f'Source changed: {name}'

    frozen = ROOT / 'docs/evaluation/week12-scope-v1'
    scope_index = read(frozen / 'package-index.json')
    for name, expected in scope_index['files_sha256'].items():
        assert digest(frozen / name) == expected, f'Step 1 package changed: {name}'
    for name in ('WEEK12_SCOPE.md', 'WEEK12_REPORT_OUTLINE.md'):
        assert digest(ROOT / 'docs' / name) == digest(frozen / 'after/docs' / name)

    numerical = read(out / 'numerical-verification.json')
    assert numerical['status'] == 'pass'
    assert numerical['check_count'] == len(numerical['checks']) == 2210
    assert all(numerical[k] == 0 for k in ('new_model_calls', 'new_rollouts', 'new_training_updates'))

    table_counts = {}
    for name, count in numerical['tables'].items():
        with (out / 'tables' / (name + '.csv')).open() as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
        assert len(rows) == count, f'Table row count: {name}'
        table_counts[name] = count
        for row in rows:
            assert None not in row and all(v is not None for v in row.values())
            source = row['source']
            pinned = source in manifest or ((ROOT / source).is_dir() and
                                           any(p.startswith(source + '/') for p in manifest))
            assert pinned, f'Unpinned table source: {source}'
            for value in row.values():
                try:
                    number = float(value)
                except ValueError:
                    continue
                assert math.isfinite(number), f'Nonfinite table cell: {name}'
            if name == 'success_rates':
                assert math.isclose(float(row['rate_percent']), 100 * int(row['successes']) / int(row['n']))
            elif name == 'contrasts':
                assert float(row['ci_low_pp']) <= float(row['estimate_pp']) <= float(row['ci_high_pp'])
            elif name == 'latency':
                assert 0 < float(row['median_ms']) <= float(row['p95_ms'])

    figures = read(out / 'figure-index.json')['figures']
    assert len(figures) == 9
    exports = []
    for figure in figures:
        for source in figure['sources']:
            assert source in manifest, f'Unpinned figure source: {source}'
        for name in figure['files']:
            assert (out / name).is_file(), f'Missing export: {name}'
            exports.append(name)
    assert len(exports) == 19 and len(set(exports)) == 19

    examples = read(out / 'examples.json')['examples']
    groups = Counter(e['group'] for e in examples)
    assert groups == {'formal paired outcomes': 8, 'exploratory mechanism cases': 3,
                      'exploratory event forecasts': 3}
    copied = []
    for example in examples:
        assert example['source'] in manifest
        rows = example.get('frames', [example])
        for row in rows:
            assert row['source'] in manifest
            assert digest(out / row['media']) == row['sha256'] == manifest[row['source']]
            copied.append(row['media'])
        if 'frames' in example:
            assert len(example['frames']) == 6
            assert example['row']['prompt_id'] == example['prediction']['prompt_id']
            assert example['row']['target'] == example['prediction']['target']
            for frame in example['frames']:
                assert frame['absolute_frame'] == example['row']['anchor'] + frame['offset']
    assert len(copied) == len(set(copied)) == 29

    image_sizes = {}
    for path in sorted(out.glob('figures/*.png')) + sorted(out.glob('examples/*.png')):
        with Image.open(path) as img:
            assert img.width > 0 and img.height > 0
            image_sizes[str(path.relative_to(out))] = list(img.size)
            img.verify()
    assert len(image_sizes) == 34
    for path in out.glob('figures/*.svg'):
        assert ET.parse(path).getroot().tag == '{http://www.w3.org/2000/svg}svg'
    for path in out.glob('figures/*.pdf'):
        pdf = PdfReader(path)
        assert len(pdf.pages) == 1 and len(pdf.pages[0].extract_text()) > 200

    gallery = Links()
    gallery.feed((out / 'gallery.html').read_text())
    assert len(gallery.images) == 34
    checked_links = 0

    def verify_link(path, target):
        nonlocal checked_links
        url = urlsplit(target)
        assert not url.scheme and not url.netloc, f'Unexpected external link: {target}'
        if not url.path:
            assert url.fragment in gallery.ids
        else:
            dest = path.parent / unquote(url.path)
            pending_seal = args.write_report and dest in (out / 'verification.json', index_path)
            assert dest.exists() or pending_seal, f'Broken link in {path}: {target}'
        checked_links += 1

    for target in gallery.targets:
        verify_link(out / 'gallery.html', target)
    # Snapshot links have their original repository context; check live output prose here.
    for path in [*out.glob('*.md'), *out.glob('tables/*.md')]:
        for target in re.findall(r'\]\(([^\s)]+)\)', path.read_text()):
            verify_link(path, target)

    scripts = ('build_week12_results.py', 'render_week12_figures.py', 'write_week12_results.py',
               'verify_week12_results.py')
    for name in scripts:
        assert digest(out / 'source' / name) == digest(ROOT / 'scripts' / name)
        compile((ROOT / 'scripts' / name).read_text(), name, 'exec')

    sealed_files = None
    if index_path.exists():
        index = read(index_path)
        sealed_files = index['file_count']
        actual = {str(p.relative_to(out)) for p in out.rglob('*') if p.is_file() and p != index_path}
        assert actual == set(index['files_sha256']), 'Package inventory differs'
        assert len(actual) == sealed_files
        for name, expected in index['files_sha256'].items():
            assert digest(out / name) == expected, f'Sealed artifact changed: {name}'
        assert digest(out / 'source/verify_week12_results.py') == digest(Path(__file__))

    report = {
        'status': 'pass', 'numerical_checks': numerical['check_count'],
        'pinned_source_files': len(manifest), 'step1_package_files_unchanged': len(scope_index['files_sha256']),
        'scope_and_outline_unchanged': True, 'table_rows': table_counts,
        'figure_entries': len(figures), 'figure_exports': len(exports),
        'example_groups': dict(groups), 'unchanged_copied_images': len(copied),
        'decoded_pngs': image_sizes, 'parsed_svgs': 5, 'parsed_single_page_pdfs': 5,
        'local_links_checked': checked_links,
        'runtime': {'python': platform.python_version(), **{
            k: importlib.metadata.version(k) for k in ('numpy', 'matplotlib', 'Pillow', 'pypdf')}},
        'scope': 'Local reporting artifacts and pinned source files; prior remote recording/input audits remain authoritative.'
    }
    if args.write_report:
        (out / 'verification.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ('decoded_pngs', 'runtime', 'scope')}
                     | {'sealed_files_checked': sealed_files}))


if __name__ == '__main__':
    main()
