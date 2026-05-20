import streamlit as st
import pandas as pd
import numpy as np
import time
import textwrap
import plotly.graph_objects as go

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


st.set_page_config(
    page_title="CricScope | IPL Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)


def html(code):
    code = textwrap.dedent(str(code)).strip()
    code = "\n".join(line.lstrip() for line in code.splitlines())
    st.markdown(code, unsafe_allow_html=True)


if "page" not in st.session_state:
    st.session_state.page = "Dashboard"

if "prob_history" not in st.session_state:
    st.session_state.prob_history = []


team_data = {
    "Chennai Super Kings": {"abbr": "CSK", "color": "#FFD400", "dark": "#3A3200"},
    "Delhi Capitals": {"abbr": "DC", "color": "#3B82F6", "dark": "#071D46"},
    "Punjab Kings": {"abbr": "PBKS", "color": "#FF4444", "dark": "#3A0707"},
    "Kolkata Knight Riders": {"abbr": "KKR", "color": "#A855F7", "dark": "#24103D"},
    "Mumbai Indians": {"abbr": "MI", "color": "#60A5FA", "dark": "#092846"},
    "Rajasthan Royals": {"abbr": "RR", "color": "#EC4899", "dark": "#3B1029"},
    "Royal Challengers Bangalore": {"abbr": "RCB", "color": "#FF4D4D", "dark": "#3A0909"},
    "Sunrisers Hyderabad": {"abbr": "SRH", "color": "#F97316", "dark": "#3B1904"},
    "Lucknow Super Giants": {"abbr": "LSG", "color": "#06B6D4", "dark": "#06323B"},
    "Gujarat Titans": {"abbr": "GT", "color": "#14B8A6", "dark": "#063831"},
    "Deccan Chargers": {"abbr": "DC2", "color": "#2563EB", "dark": "#061D4E"},
    "Pune Warriors India": {"abbr": "PWI", "color": "#0F766E", "dark": "#052D2A"},
    "Rising Pune Supergiant": {"abbr": "RPS", "color": "#7C3AED", "dark": "#210D43"},
    "Gujarat Lions": {"abbr": "GL", "color": "#F59E0B", "dark": "#3B2502"},
    "Kochi Tuskers Kerala": {"abbr": "KTK", "color": "#10B981", "dark": "#053429"},
}


html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
--bg: #05070A;
--panel: #0C1420;
--panel2: #111D2B;
--panel3: #17263A;
--text: #F4F7FB;
--muted: #A6B8D0;
--faint: #61758E;
--gold: #D4A843;
--green: #22C87A;
--red: #E05252;
--border: rgba(255,255,255,0.10);
--gold-border: rgba(212,168,67,0.35);
}

* {
box-sizing: border-box;
}

html, body, .stApp, [class*="css"] {
font-family: 'Inter', sans-serif;
color: var(--text);
background: var(--bg);
}

[data-testid="stAppViewContainer"] {
background:
radial-gradient(circle at 16% 0%, rgba(212,168,67,0.10), transparent 26%),
radial-gradient(circle at 86% 14%, rgba(59,130,246,0.10), transparent 28%),
linear-gradient(180deg, #05070A 0%, #080D14 52%, #05070A 100%);
}

.block-container {
max-width: 100% !important;
padding: 0 !important;
}

#MainMenu, footer, [data-testid="stDecoration"] {
display: none !important;
}

section[data-testid="stSidebar"] {
width: 285px !important;
background: #070C13;
border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] > div {
background: transparent;
}

.sb-brand {
padding: 34px 22px 24px;
border-bottom: 1px solid var(--border);
}

.sb-logo {
font-family: 'Syne', sans-serif;
font-size: 23px;
font-weight: 800;
letter-spacing: 0.03em;
color: var(--gold);
line-height: 1.05;
white-space: nowrap;
}

.sb-tag {
margin-top: 10px;
color: var(--faint);
font-size: 9px;
letter-spacing: 0.18em;
text-transform: uppercase;
font-weight: 700;
}

.sb-sec {
padding: 22px 24px 10px;
color: var(--faint);
font-size: 9px;
letter-spacing: 0.20em;
text-transform: uppercase;
font-weight: 700;
}

.stButton > button {
width: 100%;
min-height: 46px;
text-align: left;
padding: 12px 24px;
background: transparent;
color: #CAD8EB;
border: 0;
border-left: 2px solid transparent;
border-radius: 0;
font-size: 14px;
font-weight: 600;
transition: all 0.18s ease;
}

.stButton > button:hover,
.stButton > button:focus,
.stButton > button:active {
background: rgba(212,168,67,0.08);
color: var(--gold);
border-left-color: var(--gold);
box-shadow: none !important;
}

.sb-divider {
height: 1px;
background: var(--border);
margin: 20px 20px 10px;
}

.sb-card {
margin: 0 16px;
padding: 16px;
border-radius: 14px;
background: linear-gradient(145deg, #111D2B, #0B111A);
border: 1px solid var(--border);
}

.sb-avatar {
width: 42px;
height: 42px;
border-radius: 50%;
display: flex;
align-items: center;
justify-content: center;
color: var(--gold);
border: 1px solid var(--gold-border);
background: rgba(212,168,67,0.10);
font-family: 'Syne', sans-serif;
font-weight: 800;
margin-bottom: 12px;
}

.sb-name {
font-weight: 800;
font-size: 13px;
margin-bottom: 4px;
}

.sb-role {
color: var(--faint);
font-size: 9px;
letter-spacing: 0.13em;
text-transform: uppercase;
margin-bottom: 12px;
}

.sb-link {
display: block;
color: var(--muted);
font-size: 11px;
text-decoration: none;
margin-top: 8px;
overflow-wrap: anywhere;
}

.sb-link:hover {
color: var(--gold);
}

.sb-ver {
text-align: center;
color: var(--faint);
font-size: 9px;
letter-spacing: 0.14em;
padding: 18px 0;
}

.hero {
padding: 56px 56px 46px;
border-bottom: 1px solid var(--border);
background:
radial-gradient(circle at 80% 12%, rgba(212,168,67,0.16), transparent 25%),
linear-gradient(145deg, rgba(17,29,43,0.96), rgba(5,7,10,0.98));
}

.hero-pill {
display: inline-flex;
align-items: center;
gap: 8px;
border: 1px solid rgba(34,200,122,0.28);
background: rgba(34,200,122,0.08);
color: var(--green);
border-radius: 100px;
padding: 6px 14px;
font-size: 11px;
font-weight: 700;
letter-spacing: 0.04em;
margin-bottom: 26px;
}

.hero-dot {
width: 7px;
height: 7px;
border-radius: 50%;
background: var(--green);
animation: pulseDot 1.8s infinite;
}

@keyframes pulseDot {
0%, 100% { opacity: 1; transform: scale(1); }
50% { opacity: 0.45; transform: scale(0.75); }
}

.hero-title {
font-family: 'Syne', sans-serif;
font-size: clamp(52px, 7vw, 92px);
line-height: 0.92;
font-weight: 800;
letter-spacing: -0.02em;
}

.hero-title span {
color: var(--gold);
}

.hero-sub {
margin-top: 20px;
max-width: 560px;
color: var(--muted);
font-size: 15px;
line-height: 1.7;
}

.stat-strip {
display: grid;
grid-template-columns: repeat(5, 1fr);
background: rgba(8,13,20,0.90);
border-bottom: 1px solid var(--border);
}

.stat-cell {
padding: 24px 30px;
border-right: 1px solid var(--border);
}

.stat-cell:last-child {
border-right: 0;
}

.stat-num {
font-family: 'JetBrains Mono', monospace;
font-size: 27px;
color: var(--gold);
margin-bottom: 6px;
}

.stat-lbl {
color: var(--faint);
font-size: 9px;
letter-spacing: 0.16em;
text-transform: uppercase;
font-weight: 800;
}

.sec {
padding: 42px 56px 0;
}

.sec-kicker {
color: var(--gold);
font-size: 10px;
letter-spacing: 0.34em;
text-transform: uppercase;
margin-bottom: 12px;
}

.sec-title {
font-family: 'Syne', sans-serif;
font-size: 28px;
font-weight: 800;
}

.sec-desc {
color: var(--faint);
font-size: 12px;
margin-top: 4px;
}

.pad {
padding: 30px 56px 70px;
}

.team-orb,
.wr-orb,
.fixture-orb {
border-radius: 50%;
display: flex;
align-items: center;
justify-content: center;
font-family: 'Syne', sans-serif;
font-weight: 800;
letter-spacing: 0.08em;
color: var(--orb-color);
background:
radial-gradient(circle at 30% 24%, rgba(255,255,255,0.16), transparent 22%),
radial-gradient(circle at center, var(--orb-dark), #05070A 70%);
border: 1px solid var(--orb-color);
box-shadow:
0 0 26px color-mix(in srgb, var(--orb-color) 38%, transparent),
inset 0 0 24px rgba(255,255,255,0.04);
text-shadow: 0 0 18px color-mix(in srgb, var(--orb-color) 60%, transparent);
position: relative;
overflow: hidden;
}

.team-orb::after,
.fixture-orb::after {
content: "";
position: absolute;
inset: -45%;
background: linear-gradient(120deg, transparent 40%, rgba(255,255,255,0.22) 50%, transparent 60%);
transform: translateX(-80%) rotate(12deg);
transition: transform 0.7s ease;
}

.tc:hover .team-orb::after,
.fx:hover .fixture-orb::after {
transform: translateX(80%) rotate(12deg);
}

.team-orb {
width: 96px;
height: 96px;
margin: 0 auto 16px;
font-size: 22px;
}

.wr-orb {
width: 42px;
height: 42px;
flex-shrink: 0;
font-size: 11px;
letter-spacing: 0.02em;
}

.fixture-orb {
width: 116px;
height: 116px;
margin: 0 auto;
font-size: 27px;
}

.tc {
min-height: 205px;
padding: 30px 16px 24px;
text-align: center;
border-radius: 16px;
border: 1px solid var(--border);
background:
radial-gradient(circle at 50% 0%, color-mix(in srgb, var(--card-color) 12%, transparent), transparent 42%),
linear-gradient(145deg, rgba(17,29,43,0.96), rgba(8,13,20,0.96));
margin-bottom: 18px;
transition: transform 0.22s ease, border-color 0.22s ease, box-shadow 0.22s ease;
}

.tc:hover {
transform: translateY(-6px) scale(1.015);
border-color: var(--card-color);
box-shadow: 0 18px 45px rgba(0,0,0,0.36);
}

.tc-abbr {
font-family: 'Syne', sans-serif;
font-size: 19px;
font-weight: 800;
letter-spacing: 0.10em;
margin-bottom: 8px;
}

.tc-name {
font-size: 11px;
color: var(--muted);
}

.wr {
border-radius: 14px;
border: 1px solid var(--border);
background: linear-gradient(145deg, rgba(17,29,43,0.96), rgba(8,13,20,0.96));
padding: 15px;
margin-bottom: 13px;
transition: transform 0.2s ease, border-color 0.2s ease;
}

.wr:hover {
transform: translateY(-3px);
border-color: var(--gold-border);
}

.wr-top {
display: flex;
align-items: center;
gap: 11px;
margin-bottom: 11px;
}

.wr-info {
flex: 1;
min-width: 0;
}

.wr-abbr {
font-family: 'Syne', sans-serif;
font-size: 12px;
font-weight: 800;
}

.wr-rec {
font-size: 10px;
color: var(--faint);
}

.wr-pct {
font-family: 'JetBrains Mono', monospace;
color: var(--gold);
font-size: 16px;
}

.wr-track {
height: 5px;
background: var(--panel3);
border-radius: 100px;
overflow: hidden;
}

.wr-fill {
height: 100%;
border-radius: 100px;
}

.a-hero {
padding: 38px 56px 32px;
}

.a-hero .hero-title {
font-size: clamp(38px, 5vw, 62px);
}

.ic {
border-radius: 16px;
border: 1px solid var(--border);
background: linear-gradient(145deg, rgba(17,29,43,0.96), rgba(8,13,20,0.96));
padding: 24px;
}

.ic-lbl {
color: var(--gold);
font-size: 9px;
letter-spacing: 0.20em;
text-transform: uppercase;
font-weight: 800;
margin-bottom: 16px;
}

.stSelectbox label,
.stNumberInput label,
.stSlider label {
color: var(--faint) !important;
font-size: 10px !important;
letter-spacing: 0.12em !important;
text-transform: uppercase !important;
font-weight: 700 !important;
}

div[data-baseweb="select"] > div,
.stNumberInput input {
background: var(--panel3) !important;
border: 1px solid var(--border) !important;
color: var(--text) !important;
border-radius: 10px !important;
min-height: 44px !important;
}

.fx {
text-align: center;
border-radius: 16px;
border: 1px solid var(--border);
background:
radial-gradient(circle at 50% 0%, color-mix(in srgb, var(--card-color) 13%, transparent), transparent 42%),
linear-gradient(145deg, rgba(17,29,43,0.96), rgba(8,13,20,0.96));
padding: 32px 20px;
transition: transform 0.22s ease, border-color 0.22s ease;
}

.fx:hover {
transform: translateY(-4px);
border-color: var(--card-color);
}

.vs-wrap {
min-height: 190px;
display: flex;
align-items: center;
justify-content: center;
color: rgba(212,168,67,0.24);
font-family: 'Syne', sans-serif;
font-size: 58px;
font-weight: 800;
}

.badge {
display: inline-block;
margin-top: 12px;
padding: 6px 13px;
border-radius: 7px;
background: var(--panel3);
color: var(--faint);
font-size: 9px;
letter-spacing: 0.18em;
text-transform: uppercase;
font-weight: 800;
}

.run-btn .stButton > button {
background: var(--gold) !important;
color: #05070A !important;
text-align: center !important;
border-radius: 13px !important;
height: 54px !important;
border-left: none !important;
font-family: 'Syne', sans-serif !important;
font-weight: 800 !important;
letter-spacing: 0.10em !important;
text-transform: uppercase !important;
}

.pc,
.pc-dim {
border-radius: 16px;
padding: 28px;
background: linear-gradient(145deg, rgba(17,29,43,0.96), rgba(8,13,20,0.96));
}

.pc {
border: 1px solid var(--gold-border);
}

.pc-dim {
border: 1px solid var(--border);
}

.pc-eye,
.pc-sub,
.v-lbl {
color: var(--faint);
font-size: 9px;
letter-spacing: 0.16em;
text-transform: uppercase;
font-weight: 800;
}

.pc-team,
.v-team {
font-family: 'Syne', sans-serif;
font-size: 21px;
font-weight: 800;
margin: 12px 0 18px;
}

.pc-pct,
.pc-pct-d {
font-family: 'JetBrains Mono', monospace;
font-size: 68px;
line-height: 1;
}

.pc-pct {
color: var(--gold);
}

.pc-pct-d {
color: var(--faint);
}

.bar {
height: 5px;
background: var(--panel3);
border-radius: 100px;
overflow: hidden;
margin: 18px 0 8px;
}

.bar-g {
height: 100%;
background: var(--gold);
}

.bar-d {
height: 100%;
background: var(--faint);
}

.bar-lbls {
display: flex;
justify-content: space-between;
color: var(--faint);
font-size: 10px;
font-family: 'JetBrains Mono', monospace;
margin-bottom: 18px;
}

.chips {
display: flex;
gap: 9px;
}

.chip {
flex: 1;
padding: 11px 8px;
border-radius: 10px;
background: var(--panel3);
border: 1px solid var(--border);
text-align: center;
}

.chip-v {
font-family: 'JetBrains Mono', monospace;
font-size: 16px;
color: var(--text);
}

.chip-l {
color: var(--faint);
font-size: 8px;
letter-spacing: 0.14em;
text-transform: uppercase;
font-weight: 800;
}

.verdict {
margin-top: 20px;
padding: 22px 28px;
border-radius: 14px;
border: 1px solid var(--border);
background: linear-gradient(145deg, rgba(17,29,43,0.96), rgba(8,13,20,0.96));
display: flex;
justify-content: space-between;
gap: 16px;
flex-wrap: wrap;
}

.v-conf {
font-family: 'JetBrains Mono', monospace;
font-size: 19px;
}

.cta {
margin-top: 44px;
padding: 28px 34px;
text-align: center;
border-radius: 16px;
border: 1px solid var(--gold-border);
background: linear-gradient(145deg, rgba(212,168,67,0.12), rgba(17,29,43,0.96));
}

.cta-lbl {
color: var(--green);
font-size: 9px;
letter-spacing: 0.22em;
text-transform: uppercase;
font-weight: 800;
margin-bottom: 8px;
}

.cta-txt {
font-family: 'Syne', sans-serif;
font-size: 20px;
font-weight: 800;
}

@media (max-width: 950px) {
.hero, .a-hero, .sec, .pad {
padding-left: 24px;
padding-right: 24px;
}

.stat-strip {
grid-template-columns: repeat(2, 1fr);
}
}
</style>
""")


def team_orb(team_name, size="team"):
    d = team_data[team_name]

    cls = {
        "team": "team-orb",
        "small": "wr-orb",
        "fixture": "fixture-orb",
    }[size]

    return (
        f'<div class="{cls}" '
        f'style="--orb-color:{d["color"]}; --orb-dark:{d["dark"]};">'
        f'{d["abbr"]}'
        f'</div>'
    )


@st.cache_data
def compute_win_rates():
    matches = pd.read_csv("matches.csv")

    name_map = {
        "Delhi Daredevils": "Delhi Capitals",
        "Kings XI Punjab": "Punjab Kings",
        "Royal Challengers Bengaluru": "Royal Challengers Bangalore",
        "Rising Pune Supergiants": "Rising Pune Supergiant",
    }

    for col in ("team1", "team2", "winner"):
        matches[col] = matches[col].replace(name_map)

    stats = {}

    for team in team_data:
        played = matches[(matches["team1"] == team) | (matches["team2"] == team)]
        played = played[played["result"] == "normal"]

        wins = played[played["winner"] == team].shape[0]
        total = played.shape[0]

        stats[team] = {
            "wins": wins,
            "total": total,
            "rate": round(wins / total * 100, 1) if total else 0,
        }

    return stats


@st.cache_resource
def train_model():
    matches = pd.read_csv("matches.csv")
    deliveries = pd.read_csv("deliveries.csv")

    df = deliveries.merge(matches, left_on="match_id", right_on="id")

    target_df = df[df["inning"] == 1].groupby("match_id")["total_runs"].sum().reset_index()
    target_df.rename(columns={"total_runs": "target"}, inplace=True)

    df = df.merge(target_df, on="match_id")
    df = df[df["inning"] == 2]

    df["current_score"] = df.groupby("match_id")["total_runs"].cumsum()
    df["runs_left"] = df["target"] - df["current_score"]
    df["balls_left"] = 120 - (df["over"] * 6 + df["ball"])

    df["player_dismissed"] = df["player_dismissed"].notna().astype(int)
    df["wickets"] = 10 - df.groupby("match_id")["player_dismissed"].cumsum()

    df["over"] = df["over"].replace(0, 0.1)
    df["crr"] = df["current_score"] / (df["over"] + df["ball"] / 6)
    df["rrr"] = (df["runs_left"] * 6) / df["balls_left"]

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df["result"] = np.where(df["batting_team"] == df["winner"], 1, 0)

    final_df = df[
        [
            "batting_team",
            "bowling_team",
            "city",
            "runs_left",
            "balls_left",
            "wickets",
            "target",
            "crr",
            "rrr",
            "result",
        ]
    ].dropna()

    X = final_df.drop("result", axis=1)
    y = final_df["result"]

    preprocessor = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                ["batting_team", "bowling_team", "city"],
            ),
            (
                "num",
                "passthrough",
                ["runs_left", "balls_left", "wickets", "target", "crr", "rrr"],
            ),
        ]
    )

    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "model",
                XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    random_state=42,
                    eval_metric="logloss",
                ),
            ),
        ]
    )

    model.fit(X, y)
    return model


win_stats = compute_win_rates()
pipe = train_model()


with st.sidebar:
    html("""
    <div class="sb-brand">
    <div class="sb-logo">CRICSCOPE</div>
    <div class="sb-tag">Stadium Intelligence</div>
    </div>
    <div class="sb-sec">Navigation</div>
    """)

    if st.button("◈ Dashboard", key="nav_dash"):
        st.session_state.page = "Dashboard"

    if st.button("◉ Match Analysis", key="nav_analysis"):
        st.session_state.page = "Analysis"

    html("""
    <div class="sb-divider"></div>
    <div class="sb-sec">Built By</div>
    <div class="sb-card">
    <div class="sb-avatar">AS</div>
    <div class="sb-name">Arnav Singh</div>
    <div class="sb-role">ML · Data · Analytics</div>
    <a class="sb-link" href="mailto:itsarnav.singh80@gmail.com">itsarnav.singh80@gmail.com</a>
    <a class="sb-link" href="https://www.linkedin.com/in/arnav-singh-a87847351" target="_blank">LinkedIn Profile</a>
    <a class="sb-link" href="https://github.com/Arnav-Singh-5080" target="_blank">GitHub Profile</a>
    </div>
    <div class="sb-ver">CricScope v2.0 · IPL Edition</div>
    """)


if st.session_state.page == "Dashboard":
    html("""
    <div class="hero">
    <div class="hero-pill"><div class="hero-dot"></div>IPL Match Intelligence · Season 2025</div>
    <div class="hero-title">Cric<span>Scope</span></div>
    <div class="hero-sub">
    Precision win probability powered by machine learning.
    Track every ball, wicket, and pressure swing of a T20 chase.
    </div>
    </div>

    <div class="stat-strip">
    <div class="stat-cell"><div class="stat-num">15</div><div class="stat-lbl">Franchises</div></div>
    <div class="stat-cell"><div class="stat-num">XGB</div><div class="stat-lbl">ML Engine</div></div>
    <div class="stat-cell"><div class="stat-num">120</div><div class="stat-lbl">Balls Tracked</div></div>
    <div class="stat-cell"><div class="stat-num">6+</div><div class="stat-lbl">Key Signals</div></div>
    <div class="stat-cell"><div class="stat-num">Live</div><div class="stat-lbl">Predictions</div></div>
    </div>

    <div class="sec">
    <div class="sec-kicker">IPL Teams</div>
    <div class="sec-title">Franchise Directory</div>
    <div class="sec-desc">Logo-free animated team badges, so nothing depends on broken image links.</div>
    </div>
    """)

    html('<div class="pad">')

    cols = st.columns(4, gap="large")

    for i, (name, data) in enumerate(team_data.items()):
        with cols[i % 4]:
            html(f"""
            <div class="tc" style="--card-color:{data['color']};">
            {team_orb(name, "team")}
            <div class="tc-abbr" style="color:{data['color']};">{data['abbr']}</div>
            <div class="tc-name">{name}</div>
            </div>
            """)

    html("""
    <div style="margin-top:42px; margin-bottom:20px;">
    <div class="sec-kicker">Performance</div>
    <div class="sec-title">All-Time Win Rates</div>
    <div class="sec-desc">Historical performance across IPL seasons</div>
    </div>
    """)

    win_cols = st.columns(4, gap="large")

    for i, (name, data) in enumerate(team_data.items()):
        stats = win_stats.get(name, {"wins": 0, "total": 0, "rate": 0})
        pct = min(stats["rate"], 100)

        with win_cols[i % 4]:
            html(f"""
            <div class="wr">
            <div class="wr-top">
            {team_orb(name, "small")}
            <div class="wr-info">
            <div class="wr-abbr" style="color:{data['color']};">{data['abbr']}</div>
            <div class="wr-rec">{stats['wins']}W / {stats['total']}M</div>
            </div>
            <div class="wr-pct">{pct}%</div>
            </div>
            <div class="wr-track">
            <div class="wr-fill" style="width:{pct}%; background:{data['color']};"></div>
            </div>
            </div>
            """)

    html("""
    <div class="cta">
    <div class="cta-lbl">Get Started</div>
    <div class="cta-txt">Open Match Analysis to run live predictions →</div>
    </div>
    </div>
    """)


if st.session_state.page == "Analysis":
    html("""
    <div class="hero a-hero">
    <div class="hero-pill"><div class="hero-dot"></div>Win Probability Engine</div>
    <div class="hero-title">Match <span>Analysis</span></div>
    <div class="hero-sub">Configure the chase state to compute real-time win probabilities via XGBoost.</div>
    </div>
    """)

    html('<div class="pad">')

    teams = list(team_data.keys())

    html('<div class="ic-lbl">Match Configuration</div>')

    col1, col2 = st.columns(2, gap="large")

    with col1:
        html('<div class="ic">')
        html('<div class="ic-lbl">Select Teams</div>')

        batting_team = st.selectbox("Batting Team", teams, key="bat")
        bowling_team = st.selectbox(
            "Bowling Team",
            [team for team in teams if team != batting_team],
            key="bowl",
        )

        html('</div>')

    with col2:
        html('<div class="ic">')
        html('<div class="ic-lbl">Match State</div>')

        target = st.number_input("Target Score", min_value=50, max_value=300, value=180, step=1)
        score = st.number_input("Current Score", min_value=0, max_value=target - 1, value=50, step=1)

        over_col, wicket_col = st.columns(2)

        with over_col:
            overs = st.slider("Overs Completed", min_value=1, max_value=19, value=10)

        with wicket_col:
            wickets = st.number_input("Wickets Fallen", min_value=0, max_value=9, value=2)

        html('</div>')

    html('<div style="height:26px;"></div>')

    bat = team_data[batting_team]
    bowl = team_data[bowling_team]

    f1, f2, f3 = st.columns([2, 1, 2], gap="large")

    with f1:
        html(f"""
        <div class="fx" style="--card-color:{bat['color']}; border-color:{bat['color']}55;">
        {team_orb(batting_team, "fixture")}
        <div style="font-family:'Syne',sans-serif;font-size:28px;font-weight:800;color:{bat['color']};letter-spacing:0.08em;margin-top:16px;">
        {bat['abbr']}
        </div>
        <div class="badge">Batting</div>
        </div>
        """)

    with f2:
        html('<div class="vs-wrap">VS</div>')

    with f3:
        html(f"""
        <div class="fx" style="--card-color:{bowl['color']}; border-color:{bowl['color']}55;">
        {team_orb(bowling_team, "fixture")}
        <div style="font-family:'Syne',sans-serif;font-size:28px;font-weight:800;color:{bowl['color']};letter-spacing:0.08em;margin-top:16px;">
        {bowl['abbr']}
        </div>
        <div class="badge">Bowling</div>
        </div>
        """)

    html('<div style="height:26px;"></div>')

    html('<div class="run-btn">')
    analyze = st.button("Run Analysis", key="analyze_btn", use_container_width=True)
    html('</div>')

    if analyze:
        runs_left = target - score
        balls_left = 120 - (overs * 6)

        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        if runs_left <= 0:
            st.success(f"Match Over - **{batting_team}** already reached the target!")
            st.stop()

        if balls_left <= 0:
            st.error(f"Match Over - No balls remaining. **{bowling_team}** win!")
            st.stop()

        if rrr > 36:
            st.error(f"Invalid state - RRR of {round(rrr, 2)} is physically impossible.")
            st.stop()

        if rrr > 24:
            st.warning(f"Extreme match state - RRR is {round(rrr, 2)}. Prediction may be less reliable.")

        input_df = pd.DataFrame(
            {
                "batting_team": [batting_team],
                "bowling_team": [bowling_team],
                "city": ["Mumbai"],
                "runs_left": [runs_left],
                "balls_left": [balls_left],
                "wickets": [10 - wickets],
                "target": [target],
                "crr": [crr],
                "rrr": [rrr],
            }
        )

        with st.spinner("Analysing..."):
            time.sleep(0.35)
            probability = pipe.predict_proba(input_df)[0]

        win = probability[1]
        lose = probability[0]

        st.session_state.prob_history.append(round(win * 100, 2))

        html('<div style="height:30px;"></div>')
        html('<div class="ic-lbl">Prediction Output</div>')

        r1, r2 = st.columns(2, gap="large")

        with r1:
            bp = round(win * 100)

            html(f"""
            <div class="pc">
            <div class="pc-eye">Batting · {bat['abbr']}</div>
            <div class="pc-team">{batting_team}</div>
            <div class="pc-pct">{bp}%</div>
            <div class="pc-sub">Win Probability</div>
            <div class="bar"><div class="bar-g" style="width:{bp}%;"></div></div>
            <div class="bar-lbls"><span>0%</span><span>{bp}%</span><span>100%</span></div>
            <div class="chips">
            <div class="chip"><div class="chip-v">{score}</div><div class="chip-l">Score</div></div>
            <div class="chip"><div class="chip-v">{runs_left}</div><div class="chip-l">Needed</div></div>
            <div class="chip"><div class="chip-v">{balls_left}</div><div class="chip-l">Balls</div></div>
            </div>
            </div>
            """)

        with r2:
            ep = round(lose * 100)

            html(f"""
            <div class="pc-dim">
            <div class="pc-eye">Bowling · {bowl['abbr']}</div>
            <div class="pc-team">{bowling_team}</div>
            <div class="pc-pct-d">{ep}%</div>
            <div class="pc-sub">Win Probability</div>
            <div class="bar"><div class="bar-d" style="width:{ep}%;"></div></div>
            <div class="bar-lbls"><span>0%</span><span>{ep}%</span><span>100%</span></div>
            <div class="chips">
            <div class="chip"><div class="chip-v">{round(crr, 2)}</div><div class="chip-l">CRR</div></div>
            <div class="chip"><div class="chip-v">{round(rrr, 2)}</div><div class="chip-l">RRR</div></div>
            <div class="chip"><div class="chip-v">{10 - wickets}</div><div class="chip-l">Wkts Left</div></div>
            </div>
            </div>
            """)

        verdict = batting_team if win > 0.5 else bowling_team
        confidence = max(win, lose)

        conf_color = "#22C87A" if confidence > 0.75 else "#D4A843" if confidence > 0.55 else "#E05252"
        conf_label = "High Confidence" if confidence > 0.75 else "Moderate" if confidence > 0.55 else "Close Match"

        html(f"""
        <div class="verdict">
        <div>
        <div class="v-lbl">Model Verdict</div>
        <div class="v-team">{verdict} <span style="color:{conf_color};">favoured</span></div>
        </div>
        <div>
        <div class="v-lbl" style="text-align:right;">Confidence</div>
        <div class="v-conf" style="color:{conf_color};">{conf_label} · {round(confidence * 100)}%</div>
        </div>
        </div>
        """)

        if len(st.session_state.prob_history) > 1:
            html('<div style="height:26px;"></div>')

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=list(range(1, len(st.session_state.prob_history) + 1)),
                    y=st.session_state.prob_history,
                    mode="lines+markers",
                    line=dict(color="#D4A843", width=2.5),
                    marker=dict(color="#D4A843", size=7, line=dict(color="#05070A", width=1.5)),
                    fill="tozeroy",
                    fillcolor="rgba(212,168,67,0.07)",
                )
            )

            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=8, b=0),
                height=230,
                font=dict(family="JetBrains Mono", color="#60738C", size=10),
                xaxis=dict(title="Analysis Run", gridcolor="rgba(255,255,255,0.05)", zeroline=False),
                yaxis=dict(title="Win % Batting", gridcolor="rgba(255,255,255,0.05)", zeroline=False),
                showlegend=False,
            )

            st.plotly_chart(fig, use_container_width=True)

    html("</div>")






















