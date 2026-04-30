import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BikeFlow — Bike Sharing Analytics",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

:root {
    --bg:      #090b0f;
    --surface: #0e1117;
    --card:    #111520;
    --border:  #1e2738;
    --accent:  #00e5ff;
    --acc2:    #ff6b35;
    --acc3:    #7c3aed;
    --text:    #e2e8f0;
    --muted:   #4a5568;
    --mono:    'Space Mono', monospace;
    --sans:    'Syne', sans-serif;
}

.stApp {
    background: var(--bg) !important;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% 0%, rgba(0,229,255,0.04) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 100%, rgba(255,107,53,0.04) 0%, transparent 60%) !important;
    font-family: var(--sans) !important;
}

section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] > div { background: transparent !important; }

html, body, [class*="css"], .stMarkdown, p, span, label {
    color: var(--text) !important;
    font-family: var(--sans) !important;
}

.stSelectbox > div > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: var(--mono) !important;
    font-size: 0.8rem !important;
    border-radius: 4px !important;
}
.stSelectbox label {
    font-family: var(--mono) !important;
    font-size: 0.68rem !important;
    color: var(--muted) !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

[data-testid="stMetric"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 1.2rem 1.4rem !important;
    position: relative;
    overflow: hidden;
}
[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent), transparent);
}
[data-testid="stMetricLabel"] {
    font-family: var(--mono) !important;
    font-size: 0.6rem !important;
    color: var(--muted) !important;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}
[data-testid="stMetricValue"] {
    font-family: var(--mono) !important;
    font-size: 1.55rem !important;
    color: var(--accent) !important;
    font-weight: 700 !important;
}

.stDataFrame {
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    overflow: hidden;
}

.stAlert {
    background: rgba(0,229,255,0.05) !important;
    border: 1px solid rgba(0,229,255,0.2) !important;
    border-radius: 6px !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
}
.stSuccess {
    background: rgba(124,58,237,0.08) !important;
    border: 1px solid rgba(124,58,237,0.25) !important;
    border-radius: 6px !important;
    font-family: var(--mono) !important;
    font-size: 0.78rem !important;
}

hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ─── Matplotlib dark theme ────────────────────────────────────────────────────
BG     = "#090b0f"
CARD   = "#111520"
BORDER = "#1e2738"
ACCENT = "#00e5ff"
ACC2   = "#ff6b35"
ACC3   = "#7c3aed"
TEXT   = "#e2e8f0"
MUTED  = "#4a5568"

plt.rcParams.update({
    "figure.facecolor":   CARD,
    "axes.facecolor":     CARD,
    "axes.edgecolor":     BORDER,
    "axes.labelcolor":    MUTED,
    "axes.titlecolor":    TEXT,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.color":        MUTED,
    "ytick.color":        MUTED,
    "grid.color":         BORDER,
    "grid.alpha":         0.6,
    "text.color":         TEXT,
    "font.family":        "monospace",
    "figure.dpi":         130,
})

# ─── Load & Preprocess ───────────────────────────────────────────────────────
@st.cache_data
def load_data():
    hour = pd.read_csv("main_data.csv", parse_dates=["dteday"])
    day  = pd.read_csv("day_data.csv",  parse_dates=["dteday"])

    season_map  = {1:"Spring",2:"Summer",3:"Fall",4:"Winter"}
    year_map    = {0:"2011",1:"2012"}
    weekday_map = {0:"Sun",1:"Mon",2:"Tue",3:"Wed",4:"Thu",5:"Fri",6:"Sat"}

    for df in [hour, day]:
        df["season_label"]  = df["season"].map(season_map)
        df["year_label"]    = df["yr"].map(year_map)
        df["weekday_label"] = df["weekday"].map(weekday_map)

    return hour, day

hour_df, day_df = load_data()

# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding:0.5rem 0 1.5rem 0;">
        <div style="font-family:'Space Mono',monospace;font-size:0.55rem;color:#2d3748;
                    letter-spacing:0.25em;text-transform:uppercase;margin-bottom:0.4rem;">
            SYSTEM // ACTIVE
        </div>
        <div style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;
                    color:#00e5ff;letter-spacing:-0.03em;line-height:1;">
            BikeFlow
        </div>
        <div style="font-family:'Space Mono',monospace;font-size:0.6rem;color:#4a5568;margin-top:0.4rem;">
            Capital Bikeshare · D.C. · 2011–2012
        </div>
    </div>
    <div style="height:1px;background:linear-gradient(90deg,#1e2738,transparent);margin-bottom:1.5rem;"></div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-family:\'Space Mono\',monospace;font-size:0.58rem;color:#4a5568;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.5rem;">FILTER · TAHUN</div>', unsafe_allow_html=True)
    sel_year = st.selectbox("Tahun", ["Semua Tahun","2011","2012"], label_visibility="collapsed")

    st.markdown('<div style="font-family:\'Space Mono\',monospace;font-size:0.58rem;color:#4a5568;letter-spacing:0.15em;text-transform:uppercase;margin:1rem 0 0.5rem 0;">FILTER · MUSIM</div>', unsafe_allow_html=True)
    sel_season = st.selectbox("Musim", ["Semua Musim","Spring","Summer","Fall","Winter"], label_visibility="collapsed")

    st.markdown('<div style="height:1px;background:linear-gradient(90deg,#1e2738,transparent);margin:1.5rem 0;"></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:\'Space Mono\',monospace;font-size:0.56rem;color:#2d3748;line-height:2.2;">Proyek Analisis Data<br>Dicoding · 2024<br><br>Satrio Faza Mubarok</div>', unsafe_allow_html=True)

# ─── Filter ──────────────────────────────────────────────────────────────────
h = hour_df.copy()
d = day_df.copy()
if sel_year != "Semua Tahun":
    h = h[h["year_label"] == sel_year]
    d = d[d["year_label"] == sel_year]
if sel_season != "Semua Musim":
    h = h[h["season_label"] == sel_season]
    d = d[d["season_label"] == sel_season]

# ─── Header ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding:1rem 0 0.5rem 0;">
    <div style="font-family:'Space Mono',monospace;font-size:0.58rem;color:#00e5ff;
                letter-spacing:0.22em;text-transform:uppercase;margin-bottom:0.6rem;">
        ◈ BIKE SHARING · ANALYTICS DASHBOARD
    </div>
    <div style="font-family:'Syne',sans-serif;font-size:2.4rem;font-weight:800;
                color:#e2e8f0;letter-spacing:-0.03em;line-height:1.1;">
        Pola Peminjaman<br>
        <span style="color:#00e5ff;">Sepeda</span> Washington D.C.
    </div>
    <div style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#4a5568;
                margin-top:0.8rem;max-width:580px;line-height:1.7;">
        Analisis granular pola permintaan armada berbasis musim dan jam operasional
        untuk mengoptimalkan distribusi dan strategi pertumbuhan layanan.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div style="height:1px;background:linear-gradient(90deg,rgba(0,229,255,0.25),rgba(124,58,237,0.15),transparent);margin:1.2rem 0 1.8rem 0;"></div>', unsafe_allow_html=True)

# ─── KPI ─────────────────────────────────────────────────────────────────────
k1,k2,k3,k4 = st.columns(4)
k1.metric("TOTAL PEMINJAMAN",   f"{d['cnt'].sum():,}")
k2.metric("RATA-RATA / HARI",   f"{d['cnt'].mean():,.0f}")
k3.metric("PENGGUNA KASUAL",    f"{d['casual'].sum():,}")
k4.metric("PENGGUNA TERDAFTAR", f"{d['registered'].sum():,}")

st.markdown('<div style="height:1px;background:linear-gradient(90deg,#1e2738,transparent);margin:2rem 0;"></div>', unsafe_allow_html=True)

# ─── Q1: MUSIM ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:1.2rem;">
    <div style="font-family:'Space Mono',monospace;font-size:0.56rem;color:#00e5ff;
                letter-spacing:0.22em;text-transform:uppercase;margin-bottom:0.35rem;">
        ◈ PERTANYAAN 01
    </div>
    <div style="font-family:'Syne',sans-serif;font-size:1.35rem;font-weight:700;color:#e2e8f0;">
        Rata-rata Peminjaman per Musim
    </div>
    <div style="font-family:'Space Mono',monospace;font-size:0.66rem;color:#4a5568;margin-top:0.3rem;">
        Musim mana yang paling produktif dan paling sepi selama 2011–2012?
    </div>
</div>
""", unsafe_allow_html=True)

season_order = ["Spring","Summer","Fall","Winter"]
season_agg = (
    d.groupby("season_label")["cnt"].mean()
    .reindex([s for s in season_order if s in d["season_label"].unique()])
)

col_a, col_b = st.columns([3, 2], gap="large")

with col_a:
    fig1, ax1 = plt.subplots(figsize=(7.5, 4.5))
    s_colors = {"Spring":"#4a90d9","Summer":"#00b4d8","Fall":"#ff6b35","Winter":"#7c3aed"}
    bar_colors = [s_colors.get(s,"#4a5568") for s in season_agg.index]

    bars = ax1.bar(season_agg.index, season_agg.values,
                   color=bar_colors, edgecolor=CARD, linewidth=2, width=0.48, zorder=3)
    # Subtle glow behind each bar
    for bar, col in zip(bars, bar_colors):
        ax1.bar(bar.get_x()+bar.get_width()/2, bar.get_height(),
                width=bar.get_width()+0.06, color=col, alpha=0.1, edgecolor="none", zorder=2)

    overall = d["cnt"].mean()
    ax1.axhline(overall, color=ACCENT, linestyle=(0,(4,3)), linewidth=1.2,
                label=f"MEAN  {overall:,.0f}", zorder=4)

    for bar, val in zip(bars, season_agg.values):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+55,
                 f"{val:,.0f}", ha="center", va="bottom",
                 fontsize=8.5, fontweight="bold", color=TEXT, fontfamily="monospace")

    ax1.set_title("AVG DAILY RENTALS BY SEASON", fontsize=9, fontweight="bold",
                  color=TEXT, loc="left", pad=12, fontfamily="monospace")
    ax1.set_xlabel("SEASON", fontsize=7.5, color=MUTED, labelpad=8)
    ax1.set_ylabel("AVG RENTALS / DAY", fontsize=7.5, color=MUTED, labelpad=8)
    ax1.legend(fontsize=7.5, framealpha=0, labelcolor=ACCENT)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
    ax1.set_ylim(0, season_agg.max()*1.28)
    ax1.grid(axis="y", zorder=1)
    ax1.tick_params(colors=MUTED, labelsize=8)
    fig1.tight_layout(pad=1.5)
    st.pyplot(fig1)

with col_b:
    season_table = (
        d.groupby("season_label")["cnt"]
        .agg(["mean","sum"])
        .reindex([s for s in season_order if s in d["season_label"].unique()])
        .rename(columns={"mean":"Rata-rata","sum":"Total"})
        .round(0).astype(int)
    )
    season_table.index.name = "Musim"
    st.dataframe(
        season_table.style.format({"Rata-rata":"{:,}","Total":"{:,}"}),
        use_container_width=True,
    )

    if len(season_agg) > 1:
        best  = season_agg.idxmax()
        worst = season_agg.idxmin()
        gap   = season_agg.max() - season_agg.min()
        pct_a = (season_agg.max() - overall) / overall * 100
        pct_b = (overall - season_agg.min()) / overall * 100
        st.markdown(f"""
        <div style="margin-top:0.8rem;font-family:'Space Mono',monospace;font-size:0.7rem;
                    line-height:2.1;border:1px solid {BORDER};border-radius:6px;
                    padding:1rem 1.1rem;background:rgba(0,229,255,0.025);">
            <div style="color:#4a5568;font-size:0.56rem;letter-spacing:0.15em;
                        text-transform:uppercase;margin-bottom:0.5rem;">INSIGHT</div>
            <span style="color:#00e5ff;">▲ {best}</span>
            &nbsp;<span style="color:#e2e8f0;">{season_agg.max():,.0f}/hari</span>
            <span style="color:#4a5568;"> +{pct_a:.1f}%</span><br>
            <span style="color:#ff6b35;">▼ {worst}</span>
            &nbsp;<span style="color:#e2e8f0;">{season_agg.min():,.0f}/hari</span>
            <span style="color:#4a5568;"> -{pct_b:.1f}%</span><br>
            <span style="color:#4a5568;">── GAP</span>
            &nbsp;<span style="color:#7c3aed;">{gap:,.0f}</span>
            <span style="color:#4a5568;"> unit/hari</span>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div style="height:1px;background:linear-gradient(90deg,#1e2738,transparent);margin:2rem 0;"></div>', unsafe_allow_html=True)

# ─── Q2: JAM ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:1.2rem;">
    <div style="font-family:'Space Mono',monospace;font-size:0.56rem;color:#ff6b35;
                letter-spacing:0.22em;text-transform:uppercase;margin-bottom:0.35rem;">
        ◈ PERTANYAAN 02
    </div>
    <div style="font-family:'Syne',sans-serif;font-size:1.35rem;font-weight:700;color:#e2e8f0;">
        Pola Peminjaman per Jam
    </div>
    <div style="font-family:'Space Mono',monospace;font-size:0.66rem;color:#4a5568;margin-top:0.3rem;">
        Jam puncak dan lembah — dibedakan antara hari kerja dan hari libur.
    </div>
</div>
""", unsafe_allow_html=True)

hourly = h.groupby(["hr","workingday"])["cnt"].mean().reset_index()
hourly["hr"] = hourly["hr"].astype(int)
hourly["Tipe Hari"] = hourly["workingday"].map({0:"Libur/Akhir Pekan",1:"Hari Kerja"})

fig2, ax2 = plt.subplots(figsize=(13, 4.8))
for label, color, ls in [("Hari Kerja",ACCENT,"-"),("Libur/Akhir Pekan",ACC2,"--")]:
    sub = hourly[hourly["Tipe Hari"]==label].sort_values("hr")
    ax2.fill_between(sub["hr"], sub["cnt"], alpha=0.07, color=color)
    ax2.plot(sub["hr"], sub["cnt"], marker="o", markersize=3.5,
             linewidth=2, color=color, linestyle=ls, label=label, zorder=4)

# Annotate working day peaks
wk = hourly[hourly["Tipe Hari"]=="Hari Kerja"].sort_values("hr")
for ph in [8, 17]:
    row = wk[wk["hr"]==ph]
    if not row.empty:
        v = row["cnt"].values[0]
        ax2.annotate(f"{ph:02d}:00 · {v:.0f}",
                     xy=(ph, v), xytext=(ph+0.7, v+22),
                     fontsize=7, color=ACCENT, fontfamily="monospace",
                     arrowprops=dict(arrowstyle="-", color=BORDER, lw=1))

ax2.set_title("AVG HOURLY RENTALS — WORKDAY vs HOLIDAY", fontsize=9,
              fontweight="bold", color=TEXT, loc="left", pad=12, fontfamily="monospace")
ax2.set_xlabel("HOUR (00–23)", fontsize=7.5, color=MUTED, labelpad=8)
ax2.set_ylabel("AVG RENTALS / HOUR", fontsize=7.5, color=MUTED, labelpad=8)
ax2.set_xticks(range(0,24))
ax2.tick_params(colors=MUTED, labelsize=7.5)
ax2.legend(fontsize=8, framealpha=0, labelcolor=TEXT)
ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x,_: f"{int(x):,}"))
ax2.grid(axis="y", zorder=1)
fig2.tight_layout(pad=1.5)
st.pyplot(fig2)

cc1, cc2 = st.columns(2, gap="large")
for tipe_val, label_str, col_w, accent in [
    (1,"HARI KERJA",     cc1, ACCENT),
    (0,"LIBUR / AKHIR PEKAN", cc2, ACC2),
]:
    sub = hourly[hourly["workingday"]==tipe_val]
    if not sub.empty:
        peak   = sub.loc[sub["cnt"].idxmax()]
        trough = sub.loc[sub["cnt"].idxmin()]
        col_w.markdown(f"""
        <div style="border:1px solid {BORDER};border-left:3px solid {accent};
                    border-radius:6px;padding:1rem 1.2rem;
                    background:rgba(255,255,255,0.012);font-family:'Space Mono',monospace;
                    margin-top:0.8rem;">
            <div style="font-size:0.56rem;color:{accent};letter-spacing:0.2em;
                        text-transform:uppercase;margin-bottom:0.75rem;">{label_str}</div>
            <div style="display:flex;justify-content:space-between;margin-bottom:0.45rem;">
                <span style="color:#4a5568;font-size:0.68rem;">JAM PUNCAK</span>
                <span style="color:#e2e8f0;font-size:0.78rem;font-weight:bold;">
                    {int(peak['hr']):02d}:00 &nbsp;·&nbsp; {peak['cnt']:,.1f} /jam
                </span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:#4a5568;font-size:0.68rem;">JAM TERBAWAH</span>
                <span style="color:#4a5568;font-size:0.78rem;">
                    {int(trough['hr']):02d}:00 &nbsp;·&nbsp; {trough['cnt']:,.1f} /jam
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<div style="height:1px;background:linear-gradient(90deg,#1e2738,transparent);margin:2rem 0;"></div>', unsafe_allow_html=True)

# ─── HEATMAP ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:1.2rem;">
    <div style="font-family:'Space Mono',monospace;font-size:0.56rem;color:#7c3aed;
                letter-spacing:0.22em;text-transform:uppercase;margin-bottom:0.35rem;">
        ◈ ANALISIS LANJUTAN
    </div>
    <div style="font-family:'Syne',sans-serif;font-size:1.35rem;font-weight:700;color:#e2e8f0;">
        Heatmap · Jam × Hari dalam Seminggu
    </div>
</div>
""", unsafe_allow_html=True)

weekday_order = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
heat_data = (
    h.groupby(["weekday_label","hr"])["cnt"].mean().unstack()
    .reindex([w for w in weekday_order if w in h["weekday_label"].unique()])
)
heat_data.columns = heat_data.columns.astype(int)

cmap_dark = LinearSegmentedColormap.from_list(
    "bikeflow", [CARD,"#0d1f3c","#1a4a72","#0e8fa3","#00c8e0","#00e5ff"], N=256
)

fig3, ax3 = plt.subplots(figsize=(14, 4))
sns.heatmap(heat_data, ax=ax3, cmap=cmap_dark, linewidths=0.4, linecolor=BG,
            cbar_kws={"label":"AVG RENTALS","shrink":0.65})
ax3.set_title("AVERAGE RENTALS · HOUR × WEEKDAY", fontsize=9, fontweight="bold",
              color=TEXT, loc="left", pad=12, fontfamily="monospace")
ax3.set_xlabel("HOUR", fontsize=7.5, color=MUTED, labelpad=8)
ax3.set_ylabel("")
ax3.tick_params(colors=MUTED, labelsize=8)
ax3.collections[0].colorbar.ax.tick_params(colors=MUTED, labelsize=7)
ax3.collections[0].colorbar.ax.yaxis.label.set_color(MUTED)
ax3.collections[0].colorbar.ax.yaxis.label.set_fontsize(7)
fig3.tight_layout(pad=1.5)
st.pyplot(fig3)

st.markdown('<div style="height:1px;background:linear-gradient(90deg,#1e2738,transparent);margin:2rem 0;"></div>', unsafe_allow_html=True)

# ─── CONCLUSION ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom:1.2rem;">
    <div style="font-family:'Space Mono',monospace;font-size:0.56rem;color:#4a5568;
                letter-spacing:0.22em;text-transform:uppercase;margin-bottom:0.35rem;">
        ◈ OUTPUT
    </div>
    <div style="font-family:'Syne',sans-serif;font-size:1.35rem;font-weight:700;color:#e2e8f0;">
        Kesimpulan & Rekomendasi
    </div>
</div>
""", unsafe_allow_html=True)

cards = [
    (ACCENT, "01", "Pola Musiman",
     "Fall mencatat rata-rata tertinggi (+25% dari mean global). Spring paling sepi (−42%). Gap ~3.040 unit/hari menciptakan peluang optimalisasi musiman yang besar."),
    (ACC2,   "02", "Pola Jam & Tipe Hari",
     "Hari kerja → bimodal: puncak 08:00 & 17:00 (komuting). Hari libur → unimodal: puncak 13:00 (rekreasi). Jam 04:00 selalu lembah di semua kondisi."),
    (ACC3,   "03", "Rekomendasi Aksi",
     "① Promo membership di musim Spring. ② Maksimalkan armada koridor komuter 07–09 & 16–18. ③ Ekspansi unit & stasiun — demand tumbuh 65% (2011→2012)."),
]

c1, c2, c3 = st.columns(3, gap="medium")
for col, (accent, num, title, body) in zip([c1,c2,c3], cards):
    col.markdown(f"""
    <div style="border:1px solid {BORDER};border-top:2px solid {accent};
                border-radius:6px;padding:1.3rem;background:rgba(255,255,255,0.01);
                font-family:'Space Mono',monospace;height:100%;">
        <div style="font-size:2.2rem;font-weight:800;color:{accent};
                    opacity:0.15;font-family:'Syne',sans-serif;line-height:1;
                    margin-bottom:0.4rem;">{num}</div>
        <div style="font-size:0.78rem;font-weight:700;color:#e2e8f0;
                    margin-bottom:0.8rem;letter-spacing:0.02em;">{title}</div>
        <div style="font-size:0.67rem;color:#4a5568;line-height:1.85;">{body}</div>
    </div>
    """, unsafe_allow_html=True)

# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown('<div style="height:2.5rem;"></div>', unsafe_allow_html=True)
st.markdown("""
<div style="font-family:'Space Mono',monospace;font-size:0.55rem;color:#1e2738;
            text-align:center;padding:0.8rem 0;letter-spacing:0.12em;">
    BIKEFLOW &nbsp;·&nbsp; CAPITAL BIKESHARE ANALYTICS &nbsp;·&nbsp; WASHINGTON D.C. 2011–2012
    &nbsp;·&nbsp; SATRIO FAZA MUBAROK &nbsp;·&nbsp; DICODING DATA ANALYST
</div>
""", unsafe_allow_html=True)