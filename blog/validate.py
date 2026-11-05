#!/usr/bin/env python3
"""Validate blog_response/index.html:
(a) well-formed HTML, local links/anchors/citations resolve;
(b) every element with lang="en" (except <html>) has a lang="zh" sibling and vice versa;
(c) for each section, the multiset of numbers (regex incl. decimals and %) in EN text equals that in ZH text;
(d) count TODO / TODO-VERIFY markers.
"""
from collections import Counter
import html
import os
import re
import sys
from html.parser import HTMLParser

VOID_ELEMENTS = {
    "area", "base", "br", "col", "embed", "hr", "img", "input",
    "link", "meta", "param", "source", "track", "wbr"
}

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
MONTH_RE = re.compile(r"\b(?:" + "|".join(MONTHS) + r")\b(?=\s+\d{1,2},\s+\d{4})")
NUMBER_RE = re.compile(r"(?<![A-Za-z_])\d+(?:,\d{3})*(?:\.\d+)?%?")


class Node:
    def __init__(self, tag, attrs, line, parent=None):
        self.tag = tag
        self.attrs = dict(attrs)
        self.line = line
        self.parent = parent
        self.children = []
        self.text_parts = []


class Validator(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.root = Node("document", {}, 0)
        self.curr = self.root
        self.stack = []
        self.errors = []
        self.ids = set()
        self.local_refs = []
        self.cite_keys = []
        self.todo_spans = []
        self.todo_boxes = []
        self._in_todo_span = 0
        self._in_todo_box = 0
        self._in_bib = False
        self._span_buf = []
        self._box_buf = []
        self.bib_text = ""

    def handle_starttag(self, tag, attrs):
        line = self.getpos()[0]
        attr_dict = dict(attrs)
        if "id" in attr_dict and attr_dict["id"]:
            if attr_dict["id"] in self.ids:
                self.errors.append(f"Duplicate id: {attr_dict['id']} at line {line}")
            self.ids.add(attr_dict["id"])

        for attr_name in ("href", "src"):
            val = attr_dict.get(attr_name)
            if val and not val.startswith(("http://", "https://", "mailto:", "data:")):
                self.local_refs.append((attr_name, val, line))

        if tag == "d-cite" and "key" in attr_dict:
            for k in attr_dict["key"].split(","):
                self.cite_keys.append((k.strip(), line))

        cls = set(attr_dict.get("class", "").split())
        if "todo" in cls or "todo-verify" in cls or "TODO-VERIFY" in cls:
            if self._in_todo_span == 0:
                self._span_buf = []
            self._in_todo_span += 1
        elif self._in_todo_span > 0 and tag == "span":
            self._in_todo_span += 1

        if "todo-box" in cls:
            if self._in_todo_box == 0:
                self._box_buf = []
            self._in_todo_box += 1
        elif self._in_todo_box > 0 and tag == "div":
            self._in_todo_box += 1

        if tag == "script" and attr_dict.get("type") == "text/bibtex":
            self._in_bib = True

        node = Node(tag, attrs, line, parent=self.curr)
        self.curr.children.append(node)
        if tag not in VOID_ELEMENTS:
            self.stack.append((tag, line))
            self.curr = node

    def handle_endtag(self, tag):
        line = self.getpos()[0]
        if tag in VOID_ELEMENTS:
            return
        if not self.stack:
            self.errors.append(f"Unexpected closing tag </{tag}> at line {line}")
            return
        open_tag, open_line = self.stack.pop()
        if open_tag != tag:
            self.errors.append(
                f"Mismatched tag: opened <{open_tag}> at line {open_line}, closed </{tag}> at line {line}"
            )
        if self.curr.parent is not None:
            self.curr = self.curr.parent

        if self._in_todo_span > 0 and tag == "span":
            self._in_todo_span -= 1
            if self._in_todo_span == 0:
                self.todo_spans.append("".join(self._span_buf).strip())
        if self._in_todo_box > 0 and tag == "div":
            self._in_todo_box -= 1
            if self._in_todo_box == 0:
                text = re.sub(r"\s+", " ", "".join(self._box_buf)).strip()
                self.todo_boxes.append(text)
        if self._in_bib and tag == "script":
            self._in_bib = False

    def handle_data(self, data):
        self.curr.text_parts.append(data)
        if self._in_todo_span > 0:
            self._span_buf.append(data)
        if self._in_todo_box > 0:
            self._box_buf.append(data)
        if self._in_bib:
            self.bib_text += data

    def handle_entityref(self, name):
        val = html.unescape(f"&{name};")
        self.curr.text_parts.append(val)
        if self._in_todo_span > 0:
            self._span_buf.append(val)
        if self._in_todo_box > 0:
            self._box_buf.append(val)

    def handle_charref(self, name):
        val = html.unescape(f"&#{name};")
        self.curr.text_parts.append(val)
        if self._in_todo_span > 0:
            self._span_buf.append(val)
        if self._in_todo_box > 0:
            self._box_buf.append(val)


def get_text(node, skip_orig_quote_en=False):
    cls = set(node.attrs.get("class", "").split())
    if skip_orig_quote_en and "orig-quote-en" in cls:
        return ""
    parts = list(node.text_parts)
    for child in node.children:
        parts.append(get_text(child, skip_orig_quote_en=skip_orig_quote_en))
    return " ".join(parts)


def walk(node):
    yield node
    for child in node.children:
        yield from walk(child)


def find_section_id(node):
    curr = node
    while curr is not None:
        if curr.tag in ("d-title", "d-bibliography"):
            return curr.tag
        if "id" in curr.attrs and curr.attrs["id"]:
            return curr.attrs["id"]
        cls = set(curr.attrs.get("class", "").split())
        if "tldr" in cls:
            return "tldr"
        curr = curr.parent
    return "body"


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    html_path = os.path.join(base_dir, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        content = f.read()

    v = Validator()
    v.feed(content)
    v.close()

    if v.stack:
        for tag, line in v.stack:
            v.errors.append(f"Unclosed tag <{tag}> opened at line {line}")

    for attr_name, ref, line in v.local_refs:
        if ref.startswith("#"):
            target_id = ref[1:]
            if target_id not in v.ids:
                v.errors.append(f"Broken anchor {ref} at line {line}")
        else:
            full_path = os.path.join(base_dir, ref)
            if not os.path.exists(full_path):
                v.errors.append(f"Missing local file {ref} ({attr_name}) at line {line}")

    bib_entries = set(re.findall(r"@\w+\{\s*([^,\s]+)\s*,", v.bib_text))
    for key, line in v.cite_keys:
        if key not in bib_entries:
            v.errors.append(f"Missing BibTeX entry for d-cite key='{key}' at line {line}")

    # (b) Check that every element with lang="en" (except <html>) has a lang="zh" sibling and vice versa
    en_count = 0
    zh_count = 0
    for node in walk(v.root):
        if node.tag == "html":
            continue
        lang = node.attrs.get("lang")
        if lang in ("en", "zh"):
            if lang == "en":
                en_count += 1
            else:
                zh_count += 1
            siblings = node.parent.children if node.parent else []
            sib_langs = [s.attrs.get("lang") for s in siblings if s is not node]
            other = "zh" if lang == "en" else "en"
            if other not in sib_langs:
                v.errors.append(
                    f"Element <{node.tag} lang='{lang}'> at line {node.line} has no lang='{other}' sibling"
                )
            en_sibs = sum(1 for s in siblings if s.attrs.get("lang") == "en")
            zh_sibs = sum(1 for s in siblings if s.attrs.get("lang") == "zh")
            if en_sibs != zh_sibs:
                v.errors.append(
                    f"Parent <{node.parent.tag}> at line {node.parent.line} has unequal lang children: en={en_sibs}, zh={zh_sibs}"
                )

    # (c) Group sections inside <d-article> (by heading/exp-head id) + header/byline/tldr and compare number multisets
    # Determine section for each [lang="en"] / [lang="zh"] element in document order
    current_section = "header"
    sec_en_nums = {}
    sec_zh_nums = {}
    section_order = []

    def ensure_sec(sec):
        if sec not in sec_en_nums:
            sec_en_nums[sec] = []
            sec_zh_nums[sec] = []
            section_order.append(sec)

    for node in walk(v.root):
        if node.tag == "html":
            continue
        if node.tag == "d-title":
            current_section = "title"
        elif "id" in node.attrs and node.attrs["id"] in (
            "byline", "intro", "summary-table", "what-we-are-not-claiming",
            "exp1", "exp2", "exp3", "exp4", "exp5", "exp6", "exp7", "exp8", "exp8-gemma",
            "methods", "acknowledgements"
        ):
            current_section = node.attrs["id"]
        elif "tldr" in set(node.attrs.get("class", "").split()):
            current_section = "tldr"

        lang = node.attrs.get("lang")
        if lang in ("en", "zh"):
            # Skip if ancestor already has lang
            p = node.parent
            nested = False
            while p is not None and p.tag != "html":
                if p.attrs.get("lang") in ("en", "zh"):
                    nested = True
                    break
                p = p.parent
            if nested:
                continue
            ensure_sec(current_section)
            if lang == "en":
                txt = get_text(node, skip_orig_quote_en=False)
                # English dates spell the month ("November 5, 2026"); the Chinese twin writes "11 月".
                txt = MONTH_RE.sub(lambda m: f" {MONTHS.index(m.group(0)) + 1} ", txt)
                sec_en_nums[current_section].extend(NUMBER_RE.findall(txt))
            else:
                txt = get_text(node, skip_orig_quote_en=True)
                sec_zh_nums[current_section].extend(NUMBER_RE.findall(txt))

    num_mismatches = []
    for sec in section_order:
        c_en = Counter(sec_en_nums[sec])
        c_zh = Counter(sec_zh_nums[sec])
        if c_en != c_zh:
            only_en = c_en - c_zh
            only_zh = c_zh - c_en
            num_mismatches.append((sec, dict(only_en), dict(only_zh)))
            v.errors.append(
                f"Section '{sec}' number multiset mismatch: EN-only={dict(only_en)}, ZH-only={dict(only_zh)}"
            )

    # (d) Count TODO / TODO-VERIFY markers directly from AST nodes
    todo_en = 0
    todo_zh = 0
    todo_other = 0
    for node in walk(v.root):
        cls = set(node.attrs.get("class", "").split())
        if "todo" in cls or "todo-verify" in cls or "TODO-VERIFY" in cls:
            # Find enclosing lang
            p = node
            enc_lang = None
            while p is not None and p.tag != "html":
                if p.attrs.get("lang") in ("en", "zh"):
                    enc_lang = p.attrs.get("lang")
                    break
                p = p.parent
            if enc_lang == "en":
                todo_en += 1
            elif enc_lang == "zh":
                todo_zh += 1
            else:
                todo_other += 1
    if todo_en != todo_zh:
        v.errors.append(f"TODO span count mismatch between EN ({todo_en}) and ZH ({todo_zh})")

    if v.errors:
        print("VALIDATION FAILED:")
        for e in v.errors:
            print("  -", e)
        sys.exit(1)

    print("HTML VALIDATION PASSED: well-formed, 0 tag mismatches, all local links/anchors/citations resolve.")
    print(f"Local links checked: {len(v.local_refs)} | IDs defined: {len(v.ids)} | BibTeX keys checked: {len(v.cite_keys)}")
    print(f"Bilingual pairing check PASSED: lang='en' elements={en_count}, lang='zh' elements={zh_count} (all paired).")
    print(f"Number multiset check PASSED across {len(section_order)} sections (0 mismatches):")
    for sec in section_order:
        print(f"  - {sec}: {len(sec_en_nums[sec])} numbers matched (EN == ZH)")
    print(f"TODO / TODO-VERIFY span count: {todo_en + todo_zh + todo_other} (EN={todo_en}, ZH={todo_zh})")
    print(f"TODO box count: {len(v.todo_boxes)}")
    for i, box in enumerate(v.todo_boxes, 1):
        print(f"  [{i}] {box[:140]}{'...' if len(box) > 140 else ''}")


if __name__ == "__main__":
    main()
