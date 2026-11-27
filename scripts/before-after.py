#!/usr/bin/env python3
"""Draws the before-and-after diagram of the README: what the server did with DeepSeek-V4.1 before the work of
this fork, and what it does now, stage by stage, with the figures measured on the 2-socket server.

    python3 scripts/before-after.py

writes media/deepseek-before-after-{light,dark}.svg. The figures are those of docs/deepseek/releases.md; a row
of ROWS is one stage: its name, then what was there before and what is there now (a figure, its unit, a text).

The lines of text are cut by a count of characters. To check that none is wider than its place once drawn,
`python3 scripts/before-after.py --check DIR` writes the files to DIR with the room of every text in a
data-maxw attribute, for a browser to compare with getComputedTextLength().
"""

import os
import sys
from xml.sax.saxutils import escape

TITLE    = "DeepSeek-V4.1 on the 2-socket server, before and after"
SUBTITLE = "Same machine and same model files: 2 x 16 Xeon cores, 1 TB of RAM, CPU only"

# name, what it is measured on, (figure, unit, text) before, (figure, unit, text) after
ROWS = [
    ("Generating", "tokens per second, without a draft",
     ("9.4", "tokens/s", "All the weights in the memory of one socket."),
     ("22.9", "tokens/s", "One copy of the weights per socket (17.1). Then the weights are asked for ahead of "
                          "their use, the small steps of a token run on one thread, and there is one thread per "
                          "core.")),
    ("Generating with the draft", "DSpark, n_max=4",
     ("25.6", "tokens/s", "The first measurement, on one subject."),
     ("29.6", "tokens/s", "Mean of five subjects, from 27.2 to 33.6 (31.5 on the subject of the first measurement). "
                          "The tokens of a draft are verified by all the threads, each on the rows it sees.")),
    ("Reading a prompt", "tokens per second",
     ("236", "tokens/s", "Batches of 512 tokens."),
     ("361", "tokens/s", "Batches of 2048 tokens (326), then their matrix products through Intel AMX, and the "
                         "indexer in AVX-512.")),
    ("Starting the server again", "a conversation of 10,547 tokens",
     ("29 s", "", "Read again from its first token (computed: 10,547 tokens at 361 tokens/s)."),
     ("0.5 s", "", "Brought back from its file on a RAM disk. After a kill it comes back as it was last saved, and "
                   "what came after is read again.")),
    ("A request larger than the context", "131,072 tokens of context",
     ("6-7 min", "", "Refused, then read all the same, while the next request waited."),
     ("At once", "", "Refused without being read. The conversation the server was holding is kept.")),
    ("Agents and tools", "DeepSeek's API, Metabase",
     ("", "", "A tool call was held to the wrong form of its schema: the model could not write a Metabase query."),
     ("", "", "Tool arguments are held as JSON Schema means them, the fields of DeepSeek's API are understood, "
              "and DeepSeek Harness runs on the server.")),
]

NOTE = ("DeepSeek-V4.1-Flash UD-Q4_K_XL. Before: the first builds, 19 to 21 Nov. After: rhel8-b5379 to rhel8-b5390, "
        "26 and 27 Nov. Generation: means of 5 requests of 200 tokens after a short prompt, 32 threads (64 for the "
        "25.6). Prompts read: 3,159 to 15,761 tokens before, 12,616 after.")

THEMES = {
    "light": dict(surface="#fcfcfb", border="#e1e0d9", primary="#0b0b0b", secondary="#52514e", muted="#898781",
                  grid="#e1e0d9", before="#898781", after="#2a78d6"),
    "dark":  dict(surface="#1a1a19", border="#2c2c2a", primary="#f0efec", secondary="#c3c2b7", muted="#898781",
                  grid="#2c2c2a", before="#898781", after="#3987e5"),
}

W = 920
FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"
X_NAME, W_NAME     = 24, 158
X_BEFORE, W_BEFORE = 200, 262
X_AFTER, W_AFTER   = 500, 396
Y_HEAD   = 86          # the line of "Before" and "After"
H_SCHEMA = 184         # the drawing of the machine
LINE     = 16          # a line of text
CHAR     = 0.56        # width of a character over the font size, on the wide side: the text must fit anywhere
CHECK    = False       # --check: every text says how much room it has


def wrap(text, width, size):
    """The text in lines no wider than width, by a count of characters."""
    per_line = max(8, int(width/(CHAR*size)))
    lines, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > per_line:
            lines.append(line)
            line = word
        else:
            line = (line + " " + word) if line else word
    if line:
        lines.append(line)
    return lines


def text(o, x, y, s, size, fill, weight=None, anchor=None, maxw=None):
    attrs = f'x="{x}" y="{y}" font-size="{size}" fill="{fill}"'
    if weight:
        attrs += f' font-weight="{weight}"'
    if anchor:
        attrs += f' text-anchor="{anchor}"'
    if maxw and CHECK:
        attrs += f' data-maxw="{maxw}"'
    o.append(f'<text {attrs}>{escape(s)}</text>')


def paragraph(o, x, y, s, width, size, fill, weight=None):
    """Lines of text from the baseline y down; the baseline after the last one."""
    for line in wrap(s, width, size):
        text(o, x, y, line, size, fill, weight=weight, maxw=width)
        y += LINE
    return y


def cell_height(cell, width):
    figure, _, words = cell
    return (34 if figure else 0) + LINE*len(wrap(words, width, 12))


def cell(o, t, x, y, width, c, mark):
    """A figure, its unit and a text, with the mark of its column at the left."""
    figure, unit, words = c
    h = cell_height(c, width - 14)
    o.append(f'<rect x="{x}" y="{y}" width="3" height="{h}" rx="1.5" fill="{mark}"/>')
    ty = y + 12
    if figure:
        room = f' data-maxw="{width-14}"' if CHECK else ""
        o.append(f'<text x="{x+14}" y="{y+24}" font-size="26" font-weight="600" fill="{t["primary"]}"{room}>'
                 f'{escape(figure)}'
                 + (f'<tspan dx="6" font-size="12" font-weight="400" fill="{t["secondary"]}">{escape(unit)}</tspan>' if unit else '')
                 + '</text>')
        ty = y + 34 + 12
    paragraph(o, x + 14, ty, words, width - 14, 12, t["secondary"])


def box(o, x, y, w, h, stroke, fill="none", opacity=None):
    extra = f' fill-opacity="{opacity}"' if opacity else ""
    o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="5" fill="{fill}"{extra} stroke="{stroke}" stroke-width="1.5"/>')


def socket(o, t, x, y, n, stroke, w=124):
    box(o, x, y, w, 40, stroke)
    text(o, x + w/2, y + 18, f"Socket {n}", 12, t["primary"], weight=600, anchor="middle", maxw=w - 8)
    text(o, x + w/2, y + 33, "16 cores", 11, t["secondary"], anchor="middle", maxw=w - 8)


def schema(o, t, y):
    """The machine: where the weights and the saved conversations are, before and after."""
    g, b = t["before"], t["after"]
    text(o, X_NAME, y + 22, "The machine", 13, t["primary"], weight=600, maxw=W_NAME)
    paragraph(o, X_NAME, y + 40, "where the weights and the conversations are", W_NAME, 11, t["muted"])

    # before: one place for the weights, the threads of the other socket read across the machine
    x = X_BEFORE
    socket(o, t, x, y + 6, 0, g)
    socket(o, t, x + 138, y + 6, 1, g)
    box(o, x, y + 72, 124, 36, g, fill=g, opacity=0.14)
    text(o, x + 62, y + 88, "The weights", 12, t["primary"], weight=600, anchor="middle", maxw=116)
    text(o, x + 62, y + 102, "277 GiB", 11, t["secondary"], anchor="middle", maxw=116)
    o.append(f'<path d="M{x+62} {y+46} V{y+72}" fill="none" stroke="{g}" stroke-width="2" stroke-linecap="round"/>')
    o.append(f'<path d="M{x+200} {y+46} V{y+90} H{x+124}" fill="none" stroke="{g}" stroke-width="2" '
             f'stroke-linecap="round" stroke-linejoin="round"/>')
    text(o, x + 132, y + 108, "half the threads read", 11, t["muted"], maxw=130)
    text(o, x + 132, y + 122, "across the machine", 11, t["muted"], maxw=130)
    box(o, x, y + 142, 262, 30, g)
    text(o, x + 131, y + 161, "Conversations: lost when the server stops", 11, t["secondary"],
         anchor="middle", maxw=254)

    # after: a copy of the weights next to each socket, and the conversations on a RAM disk
    x = X_AFTER
    for n in (0, 1):
        sx = x + 206*n
        socket(o, t, sx, y + 6, n, b, 190)
        box(o, sx, y + 72, 190, 36, b, fill=b, opacity=0.14)
        text(o, sx + 95, y + 88, "A copy of the weights", 12, t["primary"], weight=600, anchor="middle", maxw=182)
        text(o, sx + 95, y + 102, "277 GiB, on a RAM disk", 11, t["secondary"], anchor="middle", maxw=182)
        o.append(f'<path d="M{sx+95} {y+46} V{y+72}" fill="none" stroke="{b}" stroke-width="2" stroke-linecap="round"/>')
    text(o, x + W_AFTER/2, y + 126, "every thread reads the copy of its own socket", 11, t["muted"],
         anchor="middle", maxw=W_AFTER - 8)
    box(o, x, y + 142, W_AFTER, 30, b)
    text(o, x + W_AFTER/2, y + 161, "Conversations: files on a RAM disk, kept when the server restarts",
         11, t["secondary"], anchor="middle", maxw=W_AFTER - 8)


def diagram(t):
    o = []
    rows_h = [max(44, cell_height(r[2], W_BEFORE - 14), cell_height(r[3], W_AFTER - 14)) + 28 for r in ROWS]
    y_rows = Y_HEAD + 14 + H_SCHEMA
    y_note = y_rows + sum(rows_h) + 22
    note = wrap(NOTE, W - 48, 11)
    H = y_note + LINE*(len(note) - 1) + 20

    o.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
             f'aria-labelledby="t d" font-family="{FONT}">')
    o.append(f'<title id="t">{escape(TITLE)}</title>')
    said = "; ".join(f"{r[0]}: {r[2][0] + ' ' + r[2][1] if r[2][0] else r[2][2]} before, "
                     f"{r[3][0] + ' ' + r[3][1] if r[3][0] else r[3][2]} after" for r in ROWS)
    o.append(f'<desc id="d">Before and after, stage by stage. {escape(said)}. The table under the diagram has '
             f'the same figures.</desc>')
    o.append(f'<rect x="0.5" y="0.5" width="{W-1}" height="{H-1}" rx="8" fill="{t["surface"]}" stroke="{t["border"]}"/>')
    text(o, 24, 32, TITLE, 16, t["primary"], weight=600, maxw=W - 48)
    text(o, 24, 52, SUBTITLE, 12, t["secondary"], maxw=W - 48)

    # the two columns: a swatch and a name
    for x, name, mark in ((X_BEFORE, "Before", t["before"]), (X_AFTER, "After", t["after"])):
        o.append(f'<rect x="{x}" y="{Y_HEAD-10}" width="12" height="12" rx="3" fill="{mark}"/>')
        text(o, x + 20, Y_HEAD, name, 13, t["primary"], weight=600, maxw=120)
    o.append(f'<line x1="24" y1="{Y_HEAD+12}" x2="{W-24}" y2="{Y_HEAD+12}" stroke="{t["grid"]}" stroke-width="1"/>')

    schema(o, t, Y_HEAD + 20)

    y = y_rows
    for r, h in zip(ROWS, rows_h):
        o.append(f'<line x1="24" y1="{y}" x2="{W-24}" y2="{y}" stroke="{t["grid"]}" stroke-width="1"/>')
        top = y + 14
        ny = paragraph(o, X_NAME, top + 14, r[0], W_NAME, 13, t["primary"], weight=600)
        paragraph(o, X_NAME, ny + 2, r[1], W_NAME, 11, t["muted"])
        cell(o, t, X_BEFORE, top, W_BEFORE, r[2], t["before"])
        cell(o, t, X_AFTER, top, W_AFTER, r[3], t["after"])
        # from before to after
        ax, ay = X_BEFORE + W_BEFORE + 10, top + 16
        o.append(f'<path d="M{ax} {ay} H{ax+16} M{ax+11} {ay-5} L{ax+16} {ay} L{ax+11} {ay+5}" fill="none" '
                 f'stroke="{t["muted"]}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>')
        y += h

    o.append(f'<line x1="24" y1="{y}" x2="{W-24}" y2="{y}" stroke="{t["grid"]}" stroke-width="1"/>')
    ny = y_note
    for line in note:
        text(o, 24, ny, line, 11, t["muted"], maxw=W - 48)
        ny += LINE
    o.append('</svg>')
    return "\n".join(o) + "\n"


def main():
    global CHECK
    media = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "media")
    if len(sys.argv) == 3 and sys.argv[1] == "--check":
        CHECK, media = True, sys.argv[2]
    for name, theme in THEMES.items():
        path = os.path.join(media, f"deepseek-before-after-{name}.svg")
        with open(path, "w", encoding="utf-8") as f:
            f.write(diagram(theme))
        print("wrote", os.path.normpath(path))


if __name__ == "__main__":
    main()
