#!/usr/bin/env python3
"""Draws the charts of the README section "Release by release", from the figures measured with
DeepSeek-V4.1-Flash on the 2-socket server:

  - the tokens per second generated, one point per release that was measured;
  - the tokens per second at which a long prompt is read, for the settings that change it.

    python3 scripts/release-chart.py

writes media/deepseek-releases-{light,dark}.svg and media/deepseek-prompt-{light,dark}.svg. To add a release,
add a row to RELEASES below (and to the table of the README) and run it again.
"""

import os
from xml.sax.saxutils import escape

# release, tokens/s without a draft, tokens/s with the DSpark draft (None = not measured), and for the releases
# that are pointed out, what changed. The best thread count measured for each release: 64 up to b5301, 32 from
# b5308 on (with the draft: from b5317 on).
RELEASES = [
    ("before", 9.7,  None, "weights interleaved between the two sockets (9.4 on one socket)"),
    ("b5294",  18.4, 25.6, "--numa-mirror: one copy of the weights per socket"),
    ("b5297",  16.9, 23.8, None),
    ("b5301",  18.4, 26.0, None),
    ("b5308",  21.3, 25.9, "weight prefetch, small nodes on one thread, 32 threads instead of 64"),
    ("b5317",  22.6, 30.3, "hc_proj, even row split, row sharing"),
    ("b5340",  22.9, 31.1, "attention on the rows a query sees"),
    ("b5342",  23.1, 30.9, None),
    ("b5345",  23.1, 31.5, None),
    ("b5353",  22.1, None, None),
    ("b5357",  22.9, None, None),
    ("b5361",  22.2, None, None),
    ("b5371",  23.0, 30.9, None),
]

TITLE    = "Generation speed, release by release"
SUBTITLE = "Tokens per second, DeepSeek-V4.1-Flash UD-Q4_K_XL on 2 x 16 cores, 200 tokens after a short prompt"
NOTE     = ("Best thread count measured for each release. Means of 5 runs; one run for b5353, b5357 and b5361, "
            "single requests for b5294 and before.")
SERIES   = ("Without a draft", "With the DSpark draft (n_max=4)")

# the reading of a long prompt: what was run, tokens/s. One prompt and one build for all the bars.
PROMPT_TITLE    = "Reading a prompt of 12616 tokens"
PROMPT_SUBTITLE = "Tokens per second, rhel8-b5371 on 2 x 16 cores, 32 threads unless said, means of 3 readings"
PROMPT = [
    ("Batches of 512 tokens (UBATCH=512)",                   241.3),
    ("Batches of 2048, former indexer (GGML_INDEXER_512=0)", 315.2),
    ("Batches of 2048 tokens, what the scripts do",          326.1),
    ("Batches of 4096 tokens (UBATCH=4096)",                 325.4),
    ("Batches of 2048 tokens, 64 threads",                   339.2),
]
PROMPT_NOTE = ("One build and one setting changed at a time, without the drafter. With the drafter loaded the "
               "figures are the same within 1% at 32 threads.")

THEMES = {
    "light": dict(surface="#fcfcfb", border="#e1e0d9", primary="#0b0b0b", secondary="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", s1="#2a78d6", s2="#eb6834"),
    "dark":  dict(surface="#1a1a19", border="#2c2c2a", primary="#f0efec", secondary="#c3c2b7", muted="#898781",
                  grid="#2c2c2a", axis="#383835", s1="#3987e5", s2="#d95926"),
}

W, H = 920, 452
LEFT, TOP, PLOT_W, PLOT_H = 46, 104, 726, 280
Y_MAX = 35
FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"


def x_of(i):
    return LEFT + 14 + i*(PLOT_W - 28)/(len(RELEASES) - 1)


def y_of(v):
    return TOP + PLOT_H*(1 - v/Y_MAX)


def num(v):
    return ("%.1f" % v).rstrip("0").rstrip(".")


def chart(t):
    o = []
    add = o.append
    add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
        f'aria-labelledby="t d" font-family="{FONT}">')
    add(f'<title id="t">{escape(TITLE)}</title>')
    last = RELEASES[-1]
    best_draft = [r for r in RELEASES if r[2] is not None][-1]
    add(f'<desc id="d">Line chart. Without a draft: {num(RELEASES[0][1])} tokens per second before --numa-mirror, '
        f'{num(RELEASES[1][1])} with {RELEASES[1][0]}, {num(last[1])} with {last[0]}. With the DSpark draft: '
        f'{num(RELEASES[1][2])} with {RELEASES[1][0]}, {num(best_draft[2])} with {best_draft[0]}. '
        f'The table under the chart has every figure.</desc>')
    add(f'<rect x="0.5" y="0.5" width="{W-1}" height="{H-1}" rx="8" fill="{t["surface"]}" stroke="{t["border"]}"/>')

    # title, subtitle, legend
    add(f'<text x="{LEFT-22}" y="32" font-size="16" font-weight="600" fill="{t["primary"]}">{escape(TITLE)}</text>')
    add(f'<text x="{LEFT-22}" y="52" font-size="12" fill="{t["secondary"]}">{escape(SUBTITLE)}</text>')
    lx = LEFT - 22
    for name, color in zip(SERIES, (t["s1"], t["s2"])):
        add(f'<line x1="{lx}" y1="74" x2="{lx+22}" y2="74" stroke="{color}" stroke-width="2" stroke-linecap="round"/>')
        add(f'<circle cx="{lx+11}" cy="74" r="4" fill="{color}" stroke="{t["surface"]}" stroke-width="2"/>')
        add(f'<text x="{lx+30}" y="78" font-size="12" fill="{t["secondary"]}">{escape(name)}</text>')
        lx += 30 + 7*len(name) + 26

    # grid and axis
    for v in (10, 20, 30):
        add(f'<line x1="{LEFT}" y1="{y_of(v):.1f}" x2="{LEFT+PLOT_W}" y2="{y_of(v):.1f}" stroke="{t["grid"]}" stroke-width="1"/>')
    add(f'<line x1="{LEFT}" y1="{y_of(0):.1f}" x2="{LEFT+PLOT_W}" y2="{y_of(0):.1f}" stroke="{t["axis"]}" stroke-width="1"/>')
    for v in (0, 10, 20, 30):
        add(f'<text x="{LEFT-8}" y="{y_of(v)+4:.1f}" font-size="11" text-anchor="end" fill="{t["muted"]}" '
            f'style="font-variant-numeric: tabular-nums">{v}</text>')
    for i, r in enumerate(RELEASES):
        add(f'<text x="{x_of(i):.1f}" y="{y_of(0)+18:.1f}" font-size="11" text-anchor="middle" fill="{t["muted"]}">{r[0]}</text>')

    # what changed: a leader from the point down to a line of text, lower for the earlier releases so that no
    # text crosses a leader
    noted = [i for i, r in enumerate(RELEASES) if r[3]]
    for k, i in enumerate(noted):
        r = RELEASES[i]
        ty = y_of(0) - 16 - 24*k - (2 if k >= 3 else 0)   # the fourth one stays clear of the gridline at 10
        x, y = x_of(i), y_of(r[1])
        add(f'<line x1="{x:.1f}" y1="{y+8:.1f}" x2="{x:.1f}" y2="{ty+3:.1f}" stroke="{t["axis"]}" stroke-width="1"/>')
        add(f'<text x="{x+7:.1f}" y="{ty:.1f}" font-size="11.5" fill="{t["secondary"]}">'
            f'<tspan font-weight="600" fill="{t["primary"]}">{num(r[1])}</tspan>'
            f'<tspan dx="7">{escape(r[3])}</tspan></text>')

    # the two lines
    for col, color, name in ((1, t["s1"], SERIES[0]), (2, t["s2"], SERIES[1])):
        pts = [(x_of(i), y_of(r[col]), r) for i, r in enumerate(RELEASES) if r[col] is not None]
        add(f'<polyline fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round" '
            f'points="{" ".join("%.1f,%.1f" % (x, y) for x, y, _ in pts)}"/>')
        for x, y, r in pts:
            add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}" stroke="{t["surface"]}" stroke-width="2">'
                f'<title>{r[0]}, {name.lower()}: {num(r[col])} tokens/s</title></circle>')

    # the ends of the lines, by name and value
    x, y, r = [(x_of(i), y_of(r[2]), r) for i, r in enumerate(RELEASES) if r[2] is not None][-1]
    add(f'<text x="{x-13:.1f}" y="{y-11:.1f}" font-size="12" fill="{t["secondary"]}">'
        f'<tspan font-weight="600" fill="{t["primary"]}">{num(r[2])}</tspan><tspan dx="7">with the draft</tspan></text>')
    x, y, r = [(x_of(i), y_of(r[2]), r) for i, r in enumerate(RELEASES) if r[2] is not None][0]
    add(f'<text x="{x:.1f}" y="{y-11:.1f}" font-size="12" font-weight="600" text-anchor="middle" fill="{t["primary"]}">{num(r[2])}</text>')
    x, y = x_of(len(RELEASES) - 1), y_of(last[1])
    add(f'<text x="{x+10:.1f}" y="{y+4:.1f}" font-size="12" fill="{t["secondary"]}">'
        f'<tspan font-weight="600" fill="{t["primary"]}">{num(last[1])}</tspan><tspan dx="7">without a draft</tspan></text>')

    add(f'<text x="{LEFT-22}" y="{H-16}" font-size="11" fill="{t["muted"]}">{escape(NOTE)}</text>')
    add('</svg>')
    return "\n".join(o) + "\n"


def prompt_chart(t):
    w, bar_h, row_h = 920, 18, 36
    left, top, x_max = 392, 84, 350
    plot_w = w - left - 96
    bottom = top + row_h*len(PROMPT)
    h = bottom + 66
    x_of_v = lambda v: left + plot_w*v/x_max

    o = []
    add = o.append
    add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" '
        f'aria-labelledby="t d" font-family="{FONT}">')
    add(f'<title id="t">{escape(PROMPT_TITLE)}</title>')
    add(f'<desc id="d">Bar chart, tokens per second. '
        + " ".join(f"{escape(name)}: {v:.0f}." for name, v in PROMPT) + '</desc>')
    add(f'<rect x="0.5" y="0.5" width="{w-1}" height="{h-1}" rx="8" fill="{t["surface"]}" stroke="{t["border"]}"/>')
    add(f'<text x="24" y="32" font-size="16" font-weight="600" fill="{t["primary"]}">{escape(PROMPT_TITLE)}</text>')
    add(f'<text x="24" y="52" font-size="12" fill="{t["secondary"]}">{escape(PROMPT_SUBTITLE)}</text>')

    for v in (100, 200, 300):
        add(f'<line x1="{x_of_v(v):.1f}" y1="{top-8}" x2="{x_of_v(v):.1f}" y2="{bottom}" stroke="{t["grid"]}" stroke-width="1"/>')
    for v in (0, 100, 200, 300):
        add(f'<text x="{x_of_v(v):.1f}" y="{bottom+18}" font-size="11" text-anchor="middle" fill="{t["muted"]}" '
            f'style="font-variant-numeric: tabular-nums">{v}</text>')
    for i, (name, v) in enumerate(PROMPT):
        y = top + row_h*i + (row_h - bar_h)/2 - 4
        length = x_of_v(v) - left
        add(f'<text x="24" y="{y+bar_h/2+4:.1f}" font-size="12" fill="{t["secondary"]}">{escape(name)}</text>')
        # square at the baseline, round at the end that carries the value
        add(f'<path d="M{left},{y:.1f} h{length-4:.1f} a4,4 0 0 1 4,4 v{bar_h-8} a4,4 0 0 1 -4,4 h{-(length-4):.1f} z" '
            f'fill="{t["s1"]}"><title>{escape(name)}: {v:.1f} tokens/s</title></path>')
        add(f'<text x="{x_of_v(v)+8:.1f}" y="{y+bar_h/2+4:.1f}" font-size="12" font-weight="600" fill="{t["primary"]}">{v:.0f}</text>')
    add(f'<line x1="{left}" y1="{top-8}" x2="{left}" y2="{bottom}" stroke="{t["axis"]}" stroke-width="1"/>')
    add(f'<text x="24" y="{h-16}" font-size="11" fill="{t["muted"]}">{escape(PROMPT_NOTE)}</text>')
    add('</svg>')
    return "\n".join(o) + "\n"


def main():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "media")
    for name, theme in THEMES.items():
        for stem, draw in (("deepseek-releases", chart), ("deepseek-prompt", prompt_chart)):
            path = os.path.normpath(os.path.join(root, f"{stem}-{name}.svg"))
            with open(path, "w", encoding="utf-8") as f:
                f.write(draw(theme))
            print(path)


if __name__ == "__main__":
    main()
