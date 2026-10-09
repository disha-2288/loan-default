"""Presentation layer: theme injection, motion engine and reusable HTML blocks."""
import math
from html import escape
from pathlib import Path

import altair as alt
import streamlit as st

ASSETS = Path(__file__).parent / "assets"

# Palette (mirrors assets/style.css)
IRIS, AQUA, AMBER, ROSE, MINT = "#8B7CFF", "#5EEAD4", "#FBBF24", "#F87171", "#34D399"
INK, MUTED, FAINT, GRID = "#E8EAF0", "#8A90A2", "#5B6175", "rgba(255,255,255,0.06)"

LOW_CUTOFF, HIGH_CUTOFF = 30, 55  # risk tier thresholds in %


# ----------------------------------------------------------------- theme
def html(body):
    """st.html without the default sanitiser, which strips inline SVG (gauges, rings, icons)."""
    st.html(body, unsafe_allow_javascript=True)


def svg(markup, cls=""):
    """Inline SVG survives Streamlit's HTML sanitiser as a data attribute; motion.js mounts it."""
    return f'<div class="fp-svg {cls}" data-svg="{escape(markup, quote=True)}"></div>'


def inject_theme():
    css = (ASSETS / "style.css").read_text(encoding="utf-8")
    js = (ASSETS / "motion.js").read_text(encoding="utf-8")
    html(f"<style>{css}</style>")
    # Ambient background layers (fixed, behind everything)
    html(
        '<div class="fp-ambient" aria-hidden="true">'
        '<div class="fp-orb fp-orb-a"></div><div class="fp-orb fp-orb-b"></div>'
        '<div class="fp-orb fp-orb-c"></div><div class="fp-grid"></div><div class="fp-grain"></div>'
        "</div>"
    )
    with st.container(key="fp_motion_engine"):
        st.html(f"<script>{js}</script>", unsafe_allow_javascript=True)


# ----------------------------------------------------------------- helpers
def tier_for(score):
    """score is a probability in % (0-100)."""
    if score < LOW_CUTOFF:
        return "low", "Low risk", "Approve"
    if score < HIGH_CUTOFF:
        return "med", "Moderate risk", "Conditional"
    return "high", "High risk", "Decline"


def tier_color(tier):
    return {"low": MINT, "med": AMBER, "high": ROSE}[tier]


def count(value, decimals=0, prefix="", suffix="", key=None):
    """A number that GSAP counts up to. Renders the final value without JS."""
    # The HTML sanitiser trims attribute values, so keep edge spaces as non-breaking spaces
    prefix, suffix = prefix.replace(" ", " "), suffix.replace(" ", " ")
    text = f"{prefix}{value:,.{decimals}f}{suffix}"
    key_attr = f' data-key="{escape(key)}"' if key else ""
    return (f'<span class="fp-num" data-count="{value}" data-decimals="{decimals}" '
            f'data-prefix="{escape(prefix)}" data-suffix="{escape(suffix)}"{key_attr}>{text}</span>')


def page_header(index, eyebrow, title, subtitle, accent=None):
    accent_html = f' <em class="fp-serif">{accent}</em>' if accent else ""
    html(f"""
    <header class="fp-page-head">
      <div class="fp-eyebrow" data-reveal><span class="fp-index">{index}</span><span class="fp-dash"></span>{eyebrow}</div>
      <h1 class="fp-h1" data-split>{title}{accent_html}</h1>
      <p class="fp-lede" data-reveal data-delay="0.25">{subtitle}</p>
    </header>
    """)


def section(title, hint=None):
    hint_html = f'<span class="fp-section-hint">{hint}</span>' if hint else ""
    html(f'<div class="fp-section" data-reveal><h3>{title}</h3>{hint_html}</div>')


def kpi_row(items):
    """items: list of dicts(label, value, decimals, prefix, suffix, sub, accent, bar)."""
    cards = []
    for i, it in enumerate(items):
        bar = ""
        if it.get("bar") is not None:
            bar = (f'<div class="fp-kpi-bar"><i data-bar="{min(100, max(0, it["bar"])):.1f}" '
                   f'style="width:{min(100, max(0, it["bar"])):.1f}%;background:{it.get("accent", IRIS)}"></i></div>')
        cards.append(f"""
        <div class="fp-kpi fp-spot" data-reveal data-delay="{0.06 * i:.2f}" style="--accent:{it.get('accent', IRIS)}">
          <div class="fp-kpi-label"><span class="fp-dot"></span>{it['label']}</div>
          <div class="fp-kpi-value">{count(it['value'], it.get('decimals', 0), it.get('prefix', ''), it.get('suffix', ''), it.get('key'))}</div>
          {bar}
          <div class="fp-kpi-sub">{it.get('sub', '')}</div>
        </div>""")
    html(f'<div class="fp-kpis">{"".join(cards)}</div>')


def marquee(items):
    chunk = "".join(f'<span class="fp-mq-item"><b>{escape(v)}</b>{escape(k)}</span><span class="fp-mq-sep">✦</span>'
                    for k, v in items)
    html(f'<div class="fp-marquee" data-reveal><div class="fp-mq-track" data-marquee>{chunk}{chunk}</div></div>')


def gauge(score, source, key="gauge", compact=False):
    tier, label, decision = tier_for(score)
    color = tier_color(tier)
    angle = -90 + 180 * min(score, 100) / 100
    ticks = []
    for t in range(0, 101, 5):
        rad = math.radians(180 - 1.8 * t)
        cls = "fp-tick-major" if t in (LOW_CUTOFF, HIGH_CUTOFF) else "fp-tick"
        ticks.append(f'<line x1="{120 + 86 * math.cos(rad):.1f}" y1="{116 - 86 * math.sin(rad):.1f}" '
                     f'x2="{120 + 78 * math.cos(rad):.1f}" y2="{116 - 78 * math.sin(rad):.1f}" class="{cls}"/>')
    ticks = "".join(ticks)
    desc = {
        "low": "Strong repayment profile. The applicant sits comfortably inside standard lending thresholds.",
        "med": "Borderline profile. Approve with conditions such as extra documentation or a co-signer.",
        "high": "Elevated likelihood of default. The profile exceeds standard risk thresholds.",
    }[tier]
    gauge_svg = f"""<svg class="fp-gauge" viewBox="0 0 240 132" role="img" aria-label="Default probability {score:.1f}%">
        <defs>
          <linearGradient id="fpg-{key}" x1="0" x2="1" y1="0" y2="0">
            <stop offset="0%" stop-color="{MINT}"/><stop offset="45%" stop-color="{AMBER}"/><stop offset="100%" stop-color="{ROSE}"/>
          </linearGradient>
          <filter id="fpglow-{key}" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="4"/></filter>
        </defs>
        {ticks}
        <path d="M20 116 A100 100 0 0 1 220 116" class="fp-gauge-track" pathLength="100"/>
        <path d="M20 116 A100 100 0 0 1 220 116" class="fp-gauge-glow" pathLength="100" stroke="url(#fpg-{key})"
              filter="url(#fpglow-{key})" style="stroke-dashoffset:{100 - min(score, 100):.2f}" data-gauge-arc="{score:.2f}" data-key="{key}-glow"/>
        <path d="M20 116 A100 100 0 0 1 220 116" class="fp-gauge-arc" pathLength="100" stroke="url(#fpg-{key})"
              style="stroke-dashoffset:{100 - min(score, 100):.2f}" data-gauge-arc="{score:.2f}" data-key="{key}-arc"/>
        <g class="fp-needle" transform="rotate({angle:.2f} 120 116)" data-gauge-needle="{angle:.2f}" data-key="{key}-needle">
          <line x1="120" y1="116" x2="120" y2="40"/><circle cx="120" cy="116" r="6"/>
        </g>
      </svg>"""
    html(f"""
    <div class="fp-gauge-card fp-spot {'is-compact' if compact else ''}" data-reveal data-rid="gauge-{key}" style="--tier:{color}">
      <div class="fp-gauge-top">
        <span class="fp-badge fp-badge-{tier}"><span class="fp-pulse"></span>{label}</span>
        <span class="fp-decision">Decision · <b>{decision}</b></span>
      </div>
      {svg(gauge_svg, "fp-gauge-mount")}
      <div class="fp-gauge-readout">
        <div class="fp-gauge-value">{count(score, 1, suffix='%', key=key + '-num')}</div>
        <div class="fp-gauge-caption">probability of default</div>
      </div>
      <p class="fp-gauge-desc">{desc}</p>
      <div class="fp-gauge-scale"><span>0</span><span>Low &lt; {LOW_CUTOFF}%</span><span>Moderate</span><span>High ≥ {HIGH_CUTOFF}%</span><span>100</span></div>
      <div class="fp-source">{source}</div>
    </div>
    """)


def factor_list(title, items, kind):
    icon = "↑" if kind == "good" else "↓"
    if not items:
        items, kind, icon = [("Nothing flagged", "No signals in this group for the current profile")], "none", "·"
    rows = "".join(
        f'<li data-reveal data-rid="fi-{title}-{i}" data-delay="{0.05 * i:.2f}"><span class="fp-fi fp-fi-{kind}">{icon}</span>'
        f'<div><b>{t}</b><small>{d}</small></div></li>'
        for i, (t, d) in enumerate(items)
    )
    html(f'<div class="fp-factors fp-spot" data-reveal data-rid="factors-{title}"><h4>{title}</h4><ul>{rows}</ul></div>')


def stat_strip(items):
    """items: list of (label, value, decimals, prefix, suffix, color)."""
    cells = "".join(
        f'<div class="fp-strip-cell"><span>{lbl}</span><b style="color:{col}">{count(v, d, p, s, key=f"strip-{lbl}")}</b></div>'
        for i, (lbl, v, d, p, s, col) in enumerate(items)
    )
    rid = "strip-" + "-".join(str(lbl) for lbl, *_ in items)
    html(f'<div class="fp-strip" data-reveal data-rid="{escape(rid)}">{cells}</div>')


def note(body, kind="info"):
    html(f'<div class="fp-note fp-note-{kind}" data-reveal>{body}</div>')


# ----------------------------------------------------------------- charts
def style_chart(chart, height=300):
    return (
        chart.properties(height=height, background="transparent", padding={"left": 12, "right": 8, "top": 12, "bottom": 4})
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=MUTED, titleColor=FAINT, gridColor=GRID, domainColor=GRID, tickColor=GRID,
            labelFont="Geist", titleFont="Geist", labelFontSize=11, titleFontSize=11, titleFontWeight=500,
            labelPadding=8, titlePadding=12,
        )
        .configure_legend(labelColor=MUTED, titleColor=FAINT, labelFont="Geist", titleFont="Geist", orient="top",
                          symbolType="circle", labelFontSize=11, titleFontSize=11)
        .configure_text(font="Geist", color=INK)
    )


def show_chart(chart, height=300):
    st.altair_chart(style_chart(chart, height), theme=None, width="stretch")
