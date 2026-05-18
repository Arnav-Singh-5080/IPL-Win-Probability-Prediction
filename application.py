import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# -----------------------------------
# CONFIG
# -----------------------------------
st.set_page_config(page_title="CricScope", layout="wide", initial_sidebar_state="expanded")

# -----------------------------------
# SESSION STATE
# -----------------------------------
if "page" not in st.session_state:
    st.session_state.page = "Dashboard"
if "last_prediction" not in st.session_state:
    st.session_state.last_prediction = None
if "prob_history" not in st.session_state:
    st.session_state.prob_history = []
if "lang" not in st.session_state:
    st.session_state.lang = "en"

# Save language preference across refreshes using URL query params
params = st.query_params
if "lang" in params:
    st.session_state.lang = params["lang"]

TRANSLATIONS = {
    "en": {
        "toggle_btn": "हिन्दी",
        "brand_name": "CRICSCOPE", "brand_tagline": "Match Intelligence Platform",
        "nav_label": "Navigation", "nav_dashboard": "◈  Dashboard", "nav_analysis": "◉  Match Analysis",
        "built_by": "Built By", "version": "CricScope v2.0 · IPL Edition",
        "hero_eyebrow": "IPL Match Intelligence · Season 2025", "hero_badge": "Live Predictions Active",
        "hero_title": "CricScope",
        "hero_subtitle": "Precision match analytics engineered for modern cricket. Real-time win probability powered by machine learning.",
        "stat_teams": "IPL Teams", "stat_model": "Model Type", "stat_balls": "Balls Tracked", "stat_signals": "Key Signals",
        "teams_label": "IPL Teams", "win_rates_label": "All-Time Win Rates",
        "cta_label": "Get Started", "cta_text": "Open Match Analysis from the sidebar →",
        "analysis_eyebrow": "Win Probability Engine", "analysis_title": "Match Analysis",
        "analysis_subtitle": "Configure the match state below to compute real-time win probabilities.",
        "match_config_label": "Match Configuration",
        "teams_section_label": "Teams", "batting_team_label": "Batting Team", "bowling_team_label": "Bowling Team",
        "match_state_label": "Match State", "target_label": "Target Score", "score_label": "Current Score",
        "overs_label": "Overs Completed", "wickets_label": "Wickets Fallen",
        "fixture_label": "Fixture", "batting_badge": "BATTING", "bowling_badge": "BOWLING",
        "analyze_btn": "Run Analysis", "prediction_label": "Prediction Output",
        "win_prob_label": "Win Probability", "score_chip": "Score", "needed_chip": "Needed",
        "balls_chip": "Balls Left", "crr_chip": "CRR", "rrr_chip": "RRR", "inhand_chip": "In Hand",
        "verdict_label": "Model Verdict", "verdict_suffix": "favoured to win",
        "confidence_label": "Confidence", "conf_high": "High", "conf_mod": "Moderate", "conf_close": "Close",
        "chart_x": "Overs", "chart_y": "Win Probability (%)", "chart_title": "Win Probability Progression",
    },
    "hi": {
        "toggle_btn": "English",
        "brand_name": "क्रिकस्कोप", "brand_tagline": "मैच इंटेलिजेंस प्लेटफॉर्म",
        "nav_label": "नेविगेशन", "nav_dashboard": "◈  डैशबोर्ड", "nav_analysis": "◉  मैच विश्लेषण",
        "built_by": "निर्माता", "version": "क्रिकस्कोप v2.0 · IPL संस्करण",
        "hero_eyebrow": "IPL मैच इंटेलिजेंस · सीज़न 2025", "hero_badge": "लाइव भविष्यवाणी सक्रिय",
        "hero_title": "क्रिकस्कोप",
        "hero_subtitle": "आधुनिक क्रिकेट के लिए सटीक मैच विश्लेषण। मशीन लर्निंग द्वारा रियल-टाइम जीत संभावना।",
        "stat_teams": "IPL टीमें", "stat_model": "मॉडल प्रकार", "stat_balls": "गेंदें ट्रैक", "stat_signals": "मुख्य संकेत",
        "teams_label": "IPL टीमें", "win_rates_label": "सर्वकालिक जीत दर",
        "cta_label": "शुरू करें", "cta_text": "साइडबार से मैच विश्लेषण खोलें →",
        "analysis_eyebrow": "जीत संभावना इंजन", "analysis_title": "मैच विश्लेषण",
        "analysis_subtitle": "रियल-टाइम जीत संभावना जानने के लिए मैच की स्थिति दर्ज करें।",
        "match_config_label": "मैच कॉन्फ़िगरेशन",
        "teams_section_label": "टीमें", "batting_team_label": "बल्लेबाजी टीम", "bowling_team_label": "गेंदबाजी टीम",
        "match_state_label": "मैच स्थिति", "target_label": "लक्ष्य स्कोर", "score_label": "वर्तमान स्कोर",
        "overs_label": "पूर्ण ओवर", "wickets_label": "गिरे विकेट",
        "fixture_label": "फिक्स्चर", "batting_badge": "बल्लेबाजी", "bowling_badge": "गेंदबाजी",
        "analyze_btn": "विश्लेषण करें", "prediction_label": "भविष्यवाणी परिणाम",
        "win_prob_label": "जीत संभावना", "score_chip": "स्कोर", "needed_chip": "चाहिए",
        "balls_chip": "शेष गेंदें", "crr_chip": "CRR", "rrr_chip": "RRR", "inhand_chip": "हाथ में",
        "verdict_label": "मॉडल निर्णय", "verdict_suffix": "की जीत संभावित",
        "confidence_label": "आत्मविश्वास", "conf_high": "उच्च", "conf_mod": "मध्यम", "conf_close": "कड़ा मुकाबला",
        "chart_x": "ओवर", "chart_y": "जीत संभावना (%)", "chart_title": "जीत संभावना प्रगति",
    },
}

def T(key):
    lang = st.session_state.get("lang", "en")
    return TRANSLATIONS.get(lang, TRANSLATIONS["en"]).get(key, TRANSLATIONS["en"].get(key, key))

# -----------------------------------
# LUXURY CSS - RESPONSIVE DESIGN
# -----------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@300;400;500;600;700&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&family=Noto+Sans+Devanagari:wght@400;500;600&display=swap');

/* ---- RESET & BASE ---- */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'DM Sans', sans-serif;
    color: #e2dfd8;
}

[data-testid="stAppViewContainer"] {
    background: #080808;
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -10%, rgba(212,175,55,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(139,90,30,0.05) 0%, transparent 50%);
    min-height: 100vh;
}

/* Hide only Streamlit branding — leave header & sidebar toggle untouched */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ---- SIDEBAR ---- */
section[data-testid="stSidebar"] {
    background: #0c0c0c;
    border-right: 1px solid rgba(212,175,55,0.12);
    width: 260px !important;
}

section[data-testid="stSidebar"] > div {
    padding: 0;
}

.sidebar-brand {
    padding: clamp(20px, 5vw, 36px) clamp(16px, 4vw, 28px) clamp(16px, 4vw, 24px);
    border-bottom: 1px solid rgba(212,175,55,0.1);
    margin-bottom: 16px;
}

.sidebar-logo-text {
    font-family: 'Noto Sans Devanagari', 'Cormorant Garamond', serif;
    font-size: clamp(18px, 4vw, 28px);
    font-weight: 600;
    letter-spacing: 3px;
    background: linear-gradient(135deg, #f0d060 0%, #d4af37 40%, #a07820 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    display: block;
    margin-bottom: 4px;
}

.sidebar-tagline {
    font-size: clamp(8px, 2vw, 10px);
    letter-spacing: 2.5px;
    text-transform: uppercase;
    color: rgba(212,175,55,0.45);
    font-weight: 400;
}

.sidebar-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(212,175,55,0.2), transparent);
    margin: 8px 0;
}

.sidebar-section-label {
    font-size: clamp(8px, 1.5vw, 9px);
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(180,160,100,0.35);
    padding: clamp(8px, 2vw, 12px) clamp(16px, 4vw, 28px) 6px;
    font-weight: 500;
}

/* ---- NAV BUTTONS ---- */
.stButton > button {
    width: 100%;
    text-align: left;
    background: transparent;
    border: none;
    border-radius: 0;
    color: rgba(220,210,180,0.65);
    font-family: 'DM Sans', sans-serif;
    font-size: clamp(11px, 2vw, 13px);
    font-weight: 400;
    letter-spacing: 0.5px;
    padding: clamp(8px, 2vw, 11px) clamp(16px, 3vw, 28px);
    height: auto;
    min-height: 44px;
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
}

/* ---- LANGUAGE TOGGLE PILL ---- */
.lang-toggle-wrapper > div > button {
    width: auto !important;
    min-height: 34px !important;
    margin-top: 20px !important;
    padding: 5px 20px !important;
    border-radius: 100px !important;
    border: 1.5px solid rgba(212,175,55,0.6) !important;
    background: rgba(212,175,55,0.07) !important;
    color: #d4af37 !important;
    font-family: 'Noto Sans Devanagari', 'DM Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    text-align: center !important;
    transition: all 0.25s ease !important;
}
.lang-toggle-wrapper > div > button:hover {
    background: rgba(212,175,55,0.18) !important;
    border-color: #d4af37 !important;
    box-shadow: 0 0 20px rgba(212,175,55,0.25) !important;
    transform: translateY(-1px) !important;
    color: #f0d060 !important;
}

.stButton > button:hover {
    background: rgba(212,175,55,0.06);
    color: #d4af37;
    border: none;
    box-shadow: none;
}

.stButton > button:active,
.stButton > button:focus {
    background: rgba(212,175,55,0.1);
    color: #f0d060;
    border: none;
    box-shadow: none;
    outline: none;
}

/* ---- MAIN CONTENT AREA ---- */
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ---- HERO SECTION ---- */
.hero-wrapper {
    padding: clamp(32px, 8vw, 64px) clamp(16px, 8vw, 60px) clamp(24px, 6vw, 40px);
    border-bottom: 1px solid rgba(212,175,55,0.08);
    position: relative;
    overflow: hidden;
}

.hero-wrapper::before {
    content: '';
    position: absolute;
    top: -60px; left: -60px; right: -60px;
    height: 200px;
    background: radial-gradient(ellipse, rgba(212,175,55,0.06) 0%, transparent 70%);
    pointer-events: none;
}

.hero-eyebrow {
    font-size: clamp(8px, 2vw, 10px);
    letter-spacing: 4px;
    text-transform: uppercase;
    color: rgba(212,175,55,0.5);
    margin-bottom: clamp(12px, 3vw, 18px);
    font-weight: 400;
}

.hero-title {
    font-family: 'Noto Sans Devanagari', 'Cormorant Garamond', serif;
    font-size: clamp(32px, 10vw, 88px);
    font-weight: 600;
    line-height: 1.4;
    letter-spacing: 0px;
    background: linear-gradient(160deg, #ffffff 0%, #f8f0d0 30%, #d4af37 70%, #a07820 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: clamp(12px, 3vw, 18px);
}

.hero-subtitle {
    font-size: clamp(12px, 3vw, 15px);
    color: rgba(220,210,185,0.55);
    font-weight: 300;
    letter-spacing: 0.3px;
    max-width: 460px;
    line-height: 1.6;
}

.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: rgba(212,175,55,0.08);
    border: 1px solid rgba(212,175,55,0.2);
    border-radius: 100px;
    padding: 5px 14px 5px 10px;
    font-size: clamp(9px, 2vw, 11px);
    color: rgba(212,175,55,0.8);
    letter-spacing: 0.5px;
    margin-bottom: clamp(16px, 4vw, 24px);
    width: fit-content;
}

.hero-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #d4af37;
    animation: pulse-dot 2s infinite;
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.8); }
}

/* ---- STAT PILLS ---- */
.stats-row {
    display: flex;
    flex-wrap: wrap;
    gap: clamp(8px, 3vw, 16px);
    padding: clamp(16px, 4vw, 24px) clamp(16px, 8vw, 60px);
    border-bottom: 1px solid rgba(212,175,55,0.06);
}

.stat-pill {
    flex: 1;
    min-width: clamp(100px, 20vw, 150px);
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: clamp(12px, 3vw, 18px) clamp(12px, 3vw, 22px);
    transition: all 0.25s ease;
}

.stat-pill:hover {
    background: rgba(212,175,55,0.04);
    border-color: rgba(212,175,55,0.15);
    transform: translateY(-1px);
}

.stat-value {
    font-family: 'DM Mono', monospace;
    font-size: clamp(18px, 4vw, 26px);
    font-weight: 500;
    color: #e8d89a;
    line-height: 1;
    margin-bottom: 6px;
}

.stat-label {
    font-size: clamp(8px, 1.5vw, 10px);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: rgba(200,185,140,0.4);
}

/* ---- ANALYSIS SECTION ---- */
.section-header {
    padding: clamp(24px, 6vw, 40px) clamp(16px, 8vw, 60px) 0;
}

.section-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(24px, 6vw, 32px);
    font-weight: 500;
    color: #f0e8cc;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}

.section-desc {
    font-size: clamp(11px, 2vw, 13px);
    color: rgba(200,185,140,0.4);
    letter-spacing: 0.3px;
}

/* ---- INPUT CARD ---- */
.input-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 20px;
    padding: clamp(16px, 4vw, 28px) clamp(16px, 4vw, 32px);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    transition: border-color 0.3s ease;
}

.input-card:hover {
    border-color: rgba(212,175,55,0.15);
}

.input-label {
    font-size: clamp(8px, 1.5vw, 10px);
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(212,175,55,0.5);
    margin-bottom: clamp(8px, 2vw, 12px);
    font-weight: 500;
}

/* ---- STREAMLIT INPUT OVERRIDES ---- */
.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stSlider > div {
    background: rgba(255,255,255,0.03) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: #e2dfd8 !important;
}

.stSelectbox label, .stNumberInput label, .stSlider label, .stTextInput label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: clamp(8px, 1.5vw, 10px) !important;
    letter-spacing: 1.8px !important;
    text-transform: uppercase !important;
    color: rgba(200,185,140,0.5) !important;
    font-weight: 500 !important;
}

/* Slider track */
.stSlider [data-testid="stSlider"] > div {
    background: rgba(212,175,55,0.15) !important;
}

.stSlider [data-testid="stSlider"] > div > div {
    background: linear-gradient(90deg, #d4af37, #f0d060) !important;
}

/* ---- TEAM VS CARD ---- */
.team-vs-wrapper {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 24px;
    padding: clamp(20px, 5vw, 36px) clamp(16px, 4vw, 28px);
    text-align: center;
    backdrop-filter: blur(20px);
    position: relative;
    overflow: hidden;
}

.team-vs-wrapper::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse 80% 60% at 50% 0%, rgba(212,175,55,0.04) 0%, transparent 60%);
    pointer-events: none;
}

.team-abbr {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(16px, 4vw, 22px);
    font-weight: 600;
    letter-spacing: 3px;
    margin-top: 14px;
}

.vs-divider {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(32px, 8vw, 48px);
    font-weight: 300;
    color: rgba(212,175,55,0.25);
    line-height: 1;
    letter-spacing: -2px;
}

.team-logo-glow {
    border-radius: 50%;
    transition: box-shadow 0.3s ease;
    width: clamp(60px, 15vw, 90px);
    height: clamp(60px, 15vw, 90px);
    object-fit: contain;
}

/* ---- ANALYZE BUTTON ---- */
.stButton.analyze-btn > button {
    background: linear-gradient(135deg, #c9a227 0%, #d4af37 40%, #e8c84a 100%);
    color: #0a0800;
    border: none;
    border-radius: 14px;
    height: clamp(44px, 10vw, 52px);
    font-family: 'DM Sans', sans-serif;
    font-size: clamp(11px, 2vw, 13px);
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    transition: all 0.3s ease;
    box-shadow: 0 8px 32px rgba(212,175,55,0.2), 0 0 0 0 rgba(212,175,55,0);
    width: 100%;
}

.stButton.analyze-btn > button:hover {
    box-shadow: 0 12px 48px rgba(212,175,55,0.35), 0 0 60px rgba(212,175,55,0.1);
    transform: translateY(-2px);
    filter: brightness(1.05);
    color: #0a0800;
    border: none;
}

/* ---- PREDICTION CARD ---- */
.prediction-card {
    background: rgba(212,175,55,0.04);
    border: 1px solid rgba(212,175,55,0.18);
    border-radius: 24px;
    padding: clamp(20px, 5vw, 36px) clamp(16px, 4vw, 32px);
    position: relative;
    overflow: hidden;
}

.prediction-card::before {
    content: '';
    position: absolute;
    top: -1px; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, #d4af37, transparent);
}

.prediction-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: radial-gradient(ellipse 70% 60% at 50% 0%, rgba(212,175,55,0.06) 0%, transparent 60%);
    pointer-events: none;
}

.prediction-label {
    font-size: clamp(8px, 1.5vw, 9px);
    letter-spacing: 3px;
    text-transform: uppercase;
    color: rgba(212,175,55,0.4);
    margin-bottom: clamp(16px, 4vw, 24px);
    font-weight: 500;
}

.win-team-name {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(18px, 5vw, 38px);
    font-weight: 600;
    color: #f0e0a0;
    line-height: 1;
    margin-bottom: 8px;
}

.win-probability {
    font-family: 'DM Mono', monospace;
    font-size: clamp(48px, 12vw, 72px);
    font-weight: 500;
    background: linear-gradient(135deg, #f0d060, #d4af37);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin-bottom: 4px;
}

.win-prob-label {
    font-size: clamp(8px, 1.5vw, 10px);
    letter-spacing: 2px;
    text-transform: uppercase;
    color: rgba(200,185,140,0.35);
    margin-bottom: clamp(16px, 4vw, 28px);
}

/* ---- PROGRESS BAR CUSTOM ---- */
.prob-bar-wrapper {
    position: relative;
    margin: clamp(12px, 3vw, 20px) 0 clamp(8px, 2vw, 14px);
}

.prob-bar-track {
    height: 6px;
    background: rgba(255,255,255,0.05);
    border-radius: 100px;
    overflow: hidden;
}

.prob-bar-fill {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #b8962e, #d4af37, #f0d060);
    transition: width 0.8s cubic-bezier(0.34, 1.56, 0.64, 1);
    box-shadow: 0 0 12px rgba(212,175,55,0.4);
}

.prob-bar-labels {
    display: flex;
    justify-content: space-between;
    margin-top: 10px;
    font-size: clamp(9px, 2vw, 11px);
    color: rgba(200,185,140,0.4);
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.5px;
}

/* ---- METRICS ROW ---- */
.metrics-row {
    display: flex;
    gap: clamp(6px, 2vw, 10px);
    margin-top: clamp(12px, 3vw, 18px);
}

.metric-chip {
    flex: 1;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: clamp(8px, 2vw, 12px) clamp(8px, 2vw, 14px);
    text-align: center;
    min-height: 50px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

.metric-chip-value {
    font-family: 'DM Mono', monospace;
    font-size: clamp(14px, 3vw, 16px);
    color: #d4c080;
    font-weight: 500;
    margin-bottom: 4px;
}

.metric-chip-label {
    font-size: clamp(7px, 1.5vw, 9px);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: rgba(180,165,115,0.35);
}

/* ---- STRAY STREAMLIT COMPONENTS ---- */
.stProgress > div > div {
    background: linear-gradient(90deg, #b8962e, #d4af37) !important;
    border-radius: 100px !important;
}

.stProgress > div {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 100px !important;
    height: 6px !important;
}

div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: clamp(12px, 3vw, 16px) clamp(12px, 3vw, 20px);
}

div[data-testid="metric-container"] label {
    color: rgba(200,185,140,0.45) !important;
    font-size: clamp(8px, 1.5vw, 10px) !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
}

div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    color: #e8d89a !important;
    font-size: clamp(20px, 5vw, 28px) !important;
}

/* ---- SEPARATOR ---- */
hr {
    border: none;
    border-top: 1px solid rgba(212,175,55,0.08);
    margin: 0;
}

/* ---- CONTENT PADDING ---- */
.main-pad {
    padding: 0 clamp(32px, 8vw, 60px) clamp(32px, 8vw, 60px) clamp(32px, 4vw, 48px);
}

/* ---- SCROLLBAR ---- */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0c0c0c; }
::-webkit-scrollbar-thumb { background: rgba(212,175,55,0.25); border-radius: 4px; }

/* ---- PROFILE CARD LINKS ---- */
.profile-link {
    display: flex; align-items: center; gap: 10px;
    text-decoration: none; padding: 8px 10px;
    border-radius: 9px; margin-bottom: 4px;
    background: transparent;
    transition: background 0.2s ease;
}
.profile-link:hover { background: rgba(212,175,55,0.07); }
.profile-link span {
    font-size: clamp(10px, 2vw, 11px); color: rgba(200,185,140,0.55);
    font-weight: 400; letter-spacing: 0.2px;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    transition: color 0.2s;
}
.profile-link:hover span { color: rgba(212,175,55,0.8); }
.profile-link svg { flex-shrink: 0; }

/* ---- MOBILE RESPONSIVE BREAKPOINTS ---- */
@media (max-width: 768px) {
    .hero-wrapper {
        padding: clamp(24px, 6vw, 48px) clamp(12px, 6vw, 40px) clamp(20px, 5vw, 32px);
    }
    .hero-title {
        font-size: clamp(24px, 8vw, 48px);
        margin-bottom: clamp(10px, 2vw, 16px);
    }
    .hero-subtitle {
        font-size: clamp(11px, 2.5vw, 14px);
        max-width: 100%;
    }
    .stats-row {
        padding: clamp(12px, 3vw, 20px) clamp(12px, 6vw, 40px);
        gap: clamp(8px, 2vw, 12px);
    }
    .stat-pill {
        min-width: 70px;
        padding: clamp(10px, 2vw, 14px) clamp(10px, 2vw, 16px);
    }
    .stat-value {
        font-size: clamp(16px, 3vw, 22px);
        margin-bottom: 4px;
    }
    .stat-label {
        font-size: clamp(7px, 1.2vw, 8px);
    }
    section[data-testid="stSidebar"] {
        width: 220px !important;
    }
    .sidebar-brand {
        padding: clamp(16px, 4vw, 24px) clamp(12px, 3vw, 20px) clamp(12px, 3vw, 16px);
    }
    .sidebar-logo-text {
        font-size: clamp(16px, 3.5vw, 24px);
    }
    .input-card {
        padding: clamp(12px, 3vw, 20px) clamp(12px, 3vw, 24px);
    }
    .prediction-card {
        padding: clamp(16px, 4vw, 28px) clamp(12px, 3vw, 24px);
    }
    .win-probability {
        font-size: clamp(36px, 8vw, 56px);
    }
    .metrics-row {
        gap: clamp(4px, 1.5vw, 8px);
    }
    .metric-chip {
        padding: clamp(6px, 1.5vw, 10px) clamp(6px, 1.5vw, 10px);
        min-height: 45px;
    }
    .metric-chip-value {
        font-size: clamp(12px, 2.5vw, 14px);
    }
    .metric-chip-label {
        font-size: clamp(6px, 1vw, 7px);
    }
    .main-pad {
        padding: 0 clamp(12px, 4vw, 32px) clamp(24px, 6vw, 48px);
    }
    .team-logo-glow {
        width: clamp(50px, 12vw, 72px);
        height: clamp(50px, 12vw, 72px);
    }
    .vs-divider {
        font-size: clamp(24px, 6vw, 40px);
    }
    .team-abbr {
        font-size: clamp(14px, 3.5vw, 20px);
    }
    .team-vs-wrapper {
        padding: clamp(16px, 4vw, 28px) clamp(12px, 3vw, 20px);
    }
}

@media (max-width: 480px) {
    section[data-testid="stSidebar"] {
        width: 200px !important;
    }
    .hero-wrapper {
        padding: clamp(16px, 5vw, 32px) clamp(10px, 4vw, 24px) clamp(16px, 4vw, 24px);
    }
    .hero-title {
        font-size: clamp(20px, 7vw, 40px);
        letter-spacing: -0.5px;
    }
    .hero-eyebrow {
        font-size: clamp(7px, 1.5vw, 8px);
        margin-bottom: clamp(8px, 2vw, 12px);
    }
    .stats-row {
        padding: clamp(10px, 2vw, 16px) clamp(10px, 4vw, 24px);
        flex-direction: column;
    }
    .stat-pill {
        min-width: 100%;
        flex: 1 1 100%;
    }
    .input-card {
        padding: clamp(10px, 2vw, 16px) clamp(10px, 2vw, 16px);
    }
    .prediction-card {
        padding: clamp(12px, 3vw, 20px) clamp(10px, 2vw, 16px);
    }
    .win-probability {
        font-size: clamp(32px, 7vw, 48px);
    }
    .win-team-name {
        font-size: clamp(14px, 4vw, 28px);
    }
    .main-pad {
        padding: 0 clamp(10px, 3vw, 20px) clamp(20px, 5vw, 32px);
    }
    .metrics-row {
        flex-direction: column;
        gap: clamp(6px, 2vw, 8px);
    }
    .metric-chip {
        min-width: 100%;
        flex: 1 1 100%;
    }
}

@media (min-width: 1440px) {
    .stats-row {
        gap: 20px;
    }
    .stat-pill {
        min-width: unset;
        flex: 1;
    }
}

</style>
""", unsafe_allow_html=True)

# -----------------------------------
# TEAM DATA
# -----------------------------------
team_data = {
    "Chennai Super Kings": {
        "logo": "http://assets.designhill.com/design-blog/wp-content/uploads/2025/03/1-5.jpg",
        "abbr": "CSK", "color": "#facc15"
    },
    "Delhi Capitals": {
        "logo": "https://sp-ao.shortpixel.ai/client/to_webp,q_glossy,ret_img,w_700/https://assets.designhill.com/design-blog/wp-content/uploads/2025/03/2-4.jpg",
        "abbr": "DC", "color": "#3b82f6"
    },
    "Punjab Kings": {
        "logo": "https://sp-ao.shortpixel.ai/client/to_webp,q_glossy,ret_img,w_700/https://assets.designhill.com/design-blog/wp-content/uploads/2025/03/5-4.jpg",
        "abbr": "PBKS", "color": "#ef4444"
    },
    "Kolkata Knight Riders": {
        "logo": "http://assets.designhill.com/design-blog/wp-content/uploads/2025/03/3-4.jpg",
        "abbr": "KKR", "color": "#7c3aed"
    },
    "Mumbai Indians": {
        "logo": "http://assets.designhill.com/design-blog/wp-content/uploads/2025/03/4-4.jpg",
        "abbr": "MI", "color": "#3b82f6"
    },
    "Rajasthan Royals": {
        "logo": "https://sp-ao.shortpixel.ai/client/to_webp,q_glossy,ret_img,w_700/https://assets.designhill.com/design-blog/wp-content/uploads/2025/03/6-4.jpg",
        "abbr": "RR", "color": "#ec4899"
    },
    "Royal Challengers Bangalore": {
        "logo": "https://assets.designhill.com/design-blog/wp-content/uploads/2025/03/Untitled-4.jpg",
        "abbr": "RCB", "color": "#dc2626"
    },
    "Sunrisers Hyderabad": {
        "logo": "http://assets.designhill.com/design-blog/wp-content/uploads/2025/03/8-4.jpg",
        "abbr": "SRH", "color": "#f97316"
    }
}

# -----------------------------------
# WIN RATE STATS
# -----------------------------------
@st.cache_data
def compute_win_rates():
    matches = pd.read_csv("matches.csv")
    valid_teams = set(team_data.keys())

    name_map = {
        "Delhi Daredevils": "Delhi Capitals",
        "Kings XI Punjab": "Punjab Kings",
        "Royal Challengers Bengaluru": "Royal Challengers Bangalore",
    }
    matches["team1"] = matches["team1"].replace(name_map)
    matches["team2"] = matches["team2"].replace(name_map)
    matches["winner"] = matches["winner"].replace(name_map)

    stats = {}
    for team in valid_teams:
        played = matches[(matches["team1"] == team) | (matches["team2"] == team)]
        played = played[played["result"] == "normal"]
        wins = played[played["winner"] == team].shape[0]
        total = played.shape[0]
        stats[team] = {"wins": wins, "total": total, "rate": round((wins / total * 100), 1) if total > 0 else 0}
    return stats

win_stats = compute_win_rates()

# -----------------------------------
# MODEL
# -----------------------------------
@st.cache_resource
def train_model():
    matches = pd.read_csv("matches.csv")
    deliveries = pd.read_csv("deliveries.csv")

    df = deliveries.merge(matches, left_on='match_id', right_on='id')

    total_df = df[df['inning'] == 1].groupby('match_id')['total_runs'].sum().reset_index()
    total_df.rename(columns={'total_runs': 'target'}, inplace=True)

    df = df.merge(total_df, on='match_id')
    df = df[df['inning'] == 2]

    df['current_score'] = df.groupby('match_id')['total_runs'].cumsum()
    df['runs_left'] = df['target'] - df['current_score']
    df['balls_left'] = 120 - (df['over'] * 6 + df['ball'])

    df['player_dismissed'] = df['player_dismissed'].notna().astype(int)
    df['wickets'] = df.groupby('match_id')['player_dismissed'].cumsum()
    df['wickets'] = 10 - df['wickets']

    df['over'] = df['over'].replace(0, 0.1)

    df['crr'] = df['current_score'] / (df['over'] + df['ball'] / 6)
    df['rrr'] = (df['runs_left'] * 6) / df['balls_left']

    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df['result'] = np.where(df['batting_team'] == df['winner'], 1, 0)

    final_df = df[['batting_team', 'bowling_team', 'city',
                   'runs_left', 'balls_left', 'wickets',
                   'target', 'crr', 'rrr', 'result']]
    final_df.dropna(inplace=True)

    X = final_df.drop('result', axis=1)
    y = final_df['result']

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['batting_team', 'bowling_team', 'city']),
        ('num', 'passthrough', ['runs_left', 'balls_left', 'wickets', 'target', 'crr', 'rrr'])
    ])

    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(max_iter=1000))
    ])

    pipe.fit(X, y)
    return pipe

pipe = train_model()

# -----------------------------------
# SIDEBAR
# -----------------------------------
with st.sidebar:
    # FIX 2: added f prefix so T() calls work
    st.markdown(f"""
        <div class="sidebar-brand">
            <span class="sidebar-logo-text" style="font-family: {'Noto Sans Devanagari, sans-serif' if st.session_state.lang == 'hi' else 'Cormorant Garamond, serif'};">{T("brand_name")}</span>
            <span class="sidebar-tagline">{T("brand_tagline")}</span>
        </div>
    """, unsafe_allow_html=True)

    # FIX 2: added f prefix
    st.markdown(f'<div class="sidebar-section-label">{T("nav_label")}</div>', unsafe_allow_html=True)

    # FIX 3: T() passed directly, no quotes around it
    if st.button(T("nav_dashboard"), key="nav_dash"):
        st.session_state.page = "Dashboard"

    if st.button(T("nav_analysis"), key="nav_analysis"):
        st.session_state.page = "Analysis"

    st.markdown('<div style="height:1px; background:rgba(212,175,55,0.08); margin:16px 0;"></div>', unsafe_allow_html=True)

    # FIX 5: wrong key "built_by_label" → correct key "built_by"
    st.markdown(f'<div class="sidebar-section-label">{T("built_by")}</div>', unsafe_allow_html=True)

    st.markdown("""
        <div style="padding:0 18px 8px;">
            <div style="background:rgba(255,255,255,0.025);border:1px solid rgba(212,175,55,0.12);
                        border-radius:16px;padding:20px 18px 14px;position:relative;overflow:hidden;">
                <div style="position:absolute;top:0;left:0;right:0;height:60px;
                            background:radial-gradient(ellipse at 50% 0%,rgba(212,175,55,0.08) 0%,transparent 70%);
                            pointer-events:none;"></div>
                <div style="width:44px;height:44px;border-radius:50%;
                            background:linear-gradient(135deg,#c9a227,#f0d060);
                            display:flex;align-items:center;justify-content:center;
                            font-size:16px;font-weight:700;color:#0a0800;
                            margin-bottom:12px;box-shadow:0 0 18px rgba(212,175,55,0.25);">AS</div>
                <div style="font-size:17px;font-weight:600;color:#f0e8cc;
                            letter-spacing:0.5px;margin-bottom:3px;">Arnav Singh</div>
                <div style="font-size:9px;letter-spacing:2px;text-transform:uppercase;
                            color:rgba(212,175,55,0.4);margin-bottom:18px;font-weight:500;">ML · Data · Analytics</div>
                <div style="height:1px;background:linear-gradient(90deg,transparent,rgba(212,175,55,0.15),transparent);margin-bottom:12px;"></div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div style="padding:0 18px;">
            <div style="background:rgba(255,255,255,0.025);border:1px solid rgba(212,175,55,0.12);
                        border-top:none;border-radius:0 0 16px 16px;padding:4px 10px 14px;">
                <p style="margin:0 0 2px 0;padding:8px 8px;">
                    <span style="color:rgba(212,175,55,0.6);margin-right:8px;font-size:12px;">✉</span>
                    <a href="mailto:itsarnav.singh80@gmail.com"
                       style="color:rgba(200,185,140,0.6);font-size:11px;text-decoration:none;letter-spacing:0.2px;">
                        itsarnav.singh80@gmail.com
                    </a>
                </p>
                <p style="margin:0 0 2px 0;padding:8px 8px;">
                    <span style="color:rgba(212,175,55,0.6);margin-right:8px;font-size:12px;">in</span>
                    <a href="https://www.linkedin.com/in/arnav-singh-a87847351" target="_blank"
                       style="color:rgba(200,185,140,0.6);font-size:11px;text-decoration:none;letter-spacing:0.2px;">
                        linkedin.com/in/arnav-singh
                    </a>
                </p>
                <p style="margin:0;padding:8px 8px;">
                    <span style="color:rgba(212,175,55,0.6);margin-right:8px;font-size:12px;">&#9670;</span>
                    <a href="https://github.com/Arnav-Singh-5080" target="_blank"
                       style="color:rgba(200,185,140,0.6);font-size:11px;text-decoration:none;letter-spacing:0.2px;">
                        Arnav-Singh-5080
                    </a>
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # FIX 2: added f prefix
    st.markdown(f"""
        <div style="text-align:center;margin-top:16px;padding-bottom:24px;font-size:9px;
                    letter-spacing:1.5px;text-transform:uppercase;color:rgba(200,185,140,0.18);">
            {T("version")}
        </div>
    """, unsafe_allow_html=True)

# FIX 1: moved OUTSIDE sidebar block (no indentation)
_spacer, _toggle_col = st.columns([7,1])
with _toggle_col:
    st.markdown('<div class="lang-toggle-wrapper">', unsafe_allow_html=True)
    if st.button(T("toggle_btn"), key="lang_toggle"):
     new_lang = "hi" if st.session_state.lang == "en" else "en"
     st.session_state.lang = new_lang
     st.query_params["lang"] = new_lang
     st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------------
# DASHBOARD PAGE
# -----------------------------------
if st.session_state.page == "Dashboard":

    # FIX 2: added f prefix to all st.markdown blocks containing T()
    st.markdown(f"""
        <div class="hero-wrapper">
            <div class="hero-eyebrow">{T("hero_eyebrow")}</div>
            <div class="hero-badge">
                <div class="hero-dot"></div>
                {T("hero_badge")}
            </div>
            <div class="hero-title" style="font-family: {'Noto Sans Devanagari, sans-serif' if st.session_state.lang == 'hi' else 'Cormorant Garamond, serif'};">{T("hero_title")}</div>
            <div class="hero-subtitle">
                {T("hero_subtitle")}
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div class="stats-row">
            <div class="stat-pill">
                <div class="stat-value">8</div>
                <div class="stat-label">{T("stat_teams")}</div>
            </div>
            <div class="stat-pill">
                <div class="stat-value">ML</div>
                <div class="stat-label">{T("stat_model")}</div>
            </div>
            <div class="stat-pill">
                <div class="stat-value">120</div>
                <div class="stat-label">{T("stat_balls")}</div>
            </div>
            <div class="stat-pill">
                <div class="stat-value">6+</div>
                <div class="stat-label">{T("stat_signals")}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div style="padding: clamp(24px, 6vw, 48px) clamp(16px, 8vw, 60px);">
            <div style="font-family:'Cormorant Garamond',serif; font-size: clamp(11px, 2vw, 13px); letter-spacing:3px;
                        text-transform:uppercase; color:rgba(212,175,55,0.4); margin-bottom: clamp(16px, 4vw, 28px);">
                {T("teams_label")}
            </div>
            <div style="display:grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); gap: clamp(8px, 2vw, 12px);">
    """, unsafe_allow_html=True)

    team_cols = st.columns(4)
    for i, (team_name, tdata) in enumerate(team_data.items()):
        with team_cols[i % 4]:
            st.markdown(f"""
                <div style="
                    background:rgba(255,255,255,0.025);
                    border:1px solid rgba(255,255,255,0.07);
                    border-radius:16px;
                    padding: clamp(12px, 3vw, 20px);
                    text-align:center;
                    transition:all 0.25s ease;
                    margin-bottom:12px;
                ">
                    <div style="width: clamp(50px, 12vw, 72px); height: clamp(50px, 12vw, 72px); border-radius:50%; margin:0 auto;
                                overflow:hidden;background:#111;
                                box-shadow:0 0 20px {tdata['color']}50;
                                display:flex;align-items:center;justify-content:center;">
                        <img src="{tdata['logo']}"
                             style="width:100%;height:100%;object-fit:cover;
                                    mix-blend-mode:screen;border-radius:50%;" />
                    </div>
                    <div style="font-family:'Cormorant Garamond',serif; font-size: clamp(14px, 3vw, 18px); font-weight:600;
                                color:{tdata['color']}; letter-spacing:2px; margin-top: clamp(8px, 2vw, 12px);">
                        {tdata['abbr']}
                    </div>
                    <div style="font-size: clamp(8px, 1.5vw, 10px); color:rgba(200,185,140,0.35); margin-top:4px;
                                letter-spacing:0.5px;">
                        {team_name}
                    </div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
        <div style="padding: 40px 60px 0;">
            <div style="font-family:'Cormorant Garamond',serif; font-size:13px; letter-spacing:3px;
                        text-transform:uppercase; color:rgba(212,175,55,0.4); margin-bottom:20px;">
                {T("win_rates_label")}
            </div>
        </div>
    """, unsafe_allow_html=True)

    wr_cols = st.columns(4)
    for i, (team_name, tdata) in enumerate(team_data.items()):
        s = win_stats.get(team_name, {"wins": 0, "total": 0, "rate": 0})
        bar_pct = s["rate"]
        with wr_cols[i % 4]:
            st.markdown(f"""
                <div style="
                    background:rgba(255,255,255,0.025);
                    border:1px solid rgba(255,255,255,0.07);
                    border-radius:16px;
                    padding:18px 20px;
                    margin-bottom:12px;
                ">
                    <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
                        <div style="width:32px;height:32px;border-radius:50%;overflow:hidden;
                                    background:#111;box-shadow:0 0 12px {tdata['color']}50;
                                    display:flex;align-items:center;justify-content:center;flex-shrink:0;">
                            <img src="{tdata['logo']}" style="width:100%;height:100%;object-fit:cover;mix-blend-mode:screen;" />
                        </div>
                        <div>
                            <div style="font-family:'Cormorant Garamond',serif;font-size:15px;
                                        font-weight:600;color:{tdata['color']};letter-spacing:1.5px;">{tdata['abbr']}</div>
                            <div style="font-size:9px;color:rgba(200,185,140,0.3);letter-spacing:0.5px;">{s['wins']}W / {s['total']}M</div>
                        </div>
                        <div style="margin-left:auto;font-family:'DM Mono',monospace;
                                    font-size:20px;font-weight:500;color:#e8d89a;">{bar_pct}%</div>
                    </div>
                    <div style="height:4px;background:rgba(255,255,255,0.05);border-radius:100px;overflow:hidden;">
                        <div style="height:100%;width:{bar_pct}%;border-radius:100px;
                                    background:linear-gradient(90deg,{tdata['color']}80,{tdata['color']});
                                    box-shadow:0 0 8px {tdata['color']}60;"></div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown(f"""
        <div style="padding: clamp(16px, 4vw, 32px) clamp(16px, 8vw, 60px); text-align:center;">
            <div style="display:inline-block; background:rgba(212,175,55,0.06); border:1px solid rgba(212,175,55,0.15);
                        border-radius:14px; padding: clamp(12px, 3vw, 20px) clamp(20px, 5vw, 36px);">
                <div style="font-size: clamp(8px, 1.5vw, 10px); letter-spacing:2px; text-transform:uppercase;
                            color:rgba(212,175,55,0.5); margin-bottom:8px;">{T("cta_label")}</div>
                <div style="font-family:'Cormorant Garamond',serif; font-size: clamp(14px, 3vw, 20px); color:#f0e8cc; font-weight:500;">
                    {T("cta_text")}
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# -----------------------------------
# ANALYSIS PAGE
# -----------------------------------
if st.session_state.page == "Analysis":

    # FIX 2: added f prefix
    st.markdown(f"""
        <div class="hero-wrapper" style="padding-bottom: clamp(20px, 5vw, 32px);">
            <div class="hero-eyebrow">{T("analysis_eyebrow")}</div>
            <div class="hero-title" style="font-size: clamp(28px, 6vw, 56px); margin-bottom: clamp(8px, 2vw, 10px); font-family: {'Noto Sans Devanagari, sans-serif' if st.session_state.lang == 'hi' else 'Cormorant Garamond, serif'};">{T("analysis_title")}</div>
            <div class="hero-subtitle">{T("analysis_subtitle")}</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="main-pad">', unsafe_allow_html=True)
    st.markdown('<div style="height: clamp(16px, 4vw, 32px);"></div>', unsafe_allow_html=True)

    teams = list(team_data.keys())

    # FIX 2: added f prefix
    st.markdown(f"""
        <div style="font-size: clamp(8px, 1.5vw, 10px); letter-spacing:3px; text-transform:uppercase;
                    color:rgba(212,175,55,0.4); margin-bottom: clamp(12px, 3vw, 20px); font-weight:500;">
            {T("match_config_label")}
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        # FIX 2: added f prefix
        st.markdown(f'<div class="input-label">{T("teams_section_label")}</div>', unsafe_allow_html=True)
        # FIX 3+4: T() passed directly, no quotes
        batting_team = st.selectbox(T("batting_team_label"), teams, key="bat")
        bowling_team = st.selectbox(T("bowling_team_label"), [t for t in teams if t != batting_team], key="bowl")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        # FIX 2: added f prefix
        st.markdown(f'<div class="input-label">{T("match_state_label")}</div>', unsafe_allow_html=True)
        # FIX 4+5: correct key names, T() passed directly
        target = st.number_input(T("target_label"), min_value=50, max_value=300, value=180, step=1)
        score = st.number_input(T("score_label"), min_value=0, max_value=target - 1, value=50, step=1)
        col_ov, col_wk = st.columns(2)
        with col_ov:
            overs = st.slider(T("overs_label"), min_value=1, max_value=19, value=10)
        with col_wk:
            wickets = st.number_input(T("wickets_label"), min_value=0, max_value=9, value=2)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height: clamp(16px, 4vw, 28px);"></div>', unsafe_allow_html=True)

    t1 = team_data[batting_team]
    if bowling_team in team_data:
        t2 = team_data[bowling_team]
    else:
        t2 = team_data[teams[1]]

    # FIX 2: added f prefix
    st.markdown(f"""
        <div style="font-size: clamp(8px, 1.5vw, 10px); letter-spacing:3px; text-transform:uppercase;
                    color:rgba(212,175,55,0.4); margin-bottom: clamp(12px, 3vw, 16px); font-weight:500;">
            {T("fixture_label")}
        </div>
    """, unsafe_allow_html=True)

    vs_col1, vs_col2, vs_col3 = st.columns([2, 1, 2])

    with vs_col1:
        st.markdown(f"""
            <div style="background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.07);
                        border-radius:20px; padding: clamp(16px, 4vw, 28px); text-align:center;
                        box-shadow:0 0 40px {t1['color']}12;">
                <div style="width: clamp(60px, 15vw, 100px); height: clamp(60px, 15vw, 100px); border-radius:50%; margin:0 auto;
                            overflow:hidden;background:#111;
                            box-shadow:0 0 28px {t1['color']}60;
                            display:flex;align-items:center;justify-content:center;">
                    <img src="{t1['logo']}"
                         style="width:100%;height:100%;object-fit:cover;
                                mix-blend-mode:screen;" />
                </div>
                <div style="font-family:'Cormorant Garamond',serif; font-size: clamp(18px, 4vw, 26px); font-weight:600;
                            color:{t1['color']}; letter-spacing:3px; margin-top: clamp(10px, 2vw, 14px);">
                    {t1['abbr']}
                </div>
                <div style="font-size: clamp(8px, 1.5vw, 10px); color:rgba(200,185,140,0.3); margin-top:4px; letter-spacing:0.5px;">
                    {T("batting_badge")}
                </div>
            </div>
        """, unsafe_allow_html=True)

    with vs_col2:
        st.markdown("""
            <div style="display:flex;align-items:center;justify-content:center;height:100%;
                        font-family:'Cormorant Garamond',serif; font-size: clamp(24px, 8vw, 52px); font-weight:300;
                        color:rgba(212,175,55,0.2); letter-spacing:-2px; padding: clamp(16px, 4vw, 28px) 0;">
                vs
            </div>
        """, unsafe_allow_html=True)

    with vs_col3:
        st.markdown(f"""
            <div style="background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.07);
                        border-radius:20px; padding: clamp(16px, 4vw, 28px); text-align:center;
                        box-shadow:0 0 40px {t2['color']}12;">
                <div style="width: clamp(60px, 15vw, 100px); height: clamp(60px, 15vw, 100px); border-radius:50%; margin:0 auto;
                            overflow:hidden;background:#111;
                            box-shadow:0 0 28px {t2['color']}60;
                            display:flex;align-items:center;justify-content:center;">
                    <img src="{t2['logo']}"
                         style="width:100%;height:100%;object-fit:cover;
                                mix-blend-mode:screen;" />
                </div>
                <div style="font-family:'Cormorant Garamond',serif; font-size: clamp(18px, 4vw, 26px); font-weight:600;
                            color:{t2['color']}; letter-spacing:3px; margin-top: clamp(10px, 2vw, 14px);">
                    {t2['abbr']}
                </div>
                <div style="font-size: clamp(8px, 1.5vw, 10px); color:rgba(200,185,140,0.3); margin-top:4px; letter-spacing:0.5px;">
                    {T("bowling_badge")}
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height: clamp(16px, 4vw, 28px);"></div>', unsafe_allow_html=True)

    st.markdown('<div class="analyze-btn">', unsafe_allow_html=True)
    # FIX 4: T() passed directly, no quotes
    analyze = st.button(T("analyze_btn"), key="analyze_btn", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if analyze:
        runs_left = target - score
        balls_left = 120 - (overs * 6)
        crr = score / overs if overs > 0 else 0
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else 0

        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': ['Mumbai'],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [10 - wickets],
            'target': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        with st.spinner(""):
            time.sleep(0.4)
            proba = pipe.predict_proba(input_df)[0]

        win = proba[1]
        lose = proba[0]
        st.session_state.prob_history.append(round(win * 100, 2))

        st.markdown('<div style="height: clamp(16px, 4vw, 28px);"></div>', unsafe_allow_html=True)

        # FIX 2+11: added f prefix
        st.markdown(f"""
            <div style="font-size: clamp(8px, 1.5vw, 10px); letter-spacing:3px; text-transform:uppercase;
                        color:rgba(212,175,55,0.4); margin-bottom: clamp(12px, 3vw, 16px); font-weight:500;">
                {T("prediction_label")}
            </div>
        """, unsafe_allow_html=True)

        res_col1, res_col2 = st.columns(2, gap="large")

        with res_col1:
            bat_pct = round(win * 100)
            st.markdown(f"""
                <div class="prediction-card">
                    <div class="prediction-label">{T("batting_badge")} · {t1['abbr']}</div>
                    <div style="font-family:'Cormorant Garamond',serif; font-size: clamp(16px, 4vw, 22px);
                                font-weight:500;color:#c8b870;margin-bottom: clamp(12px, 3vw, 16px);">
                        {batting_team}
                    </div>
                    <!-- FIX 6: restored correct CSS class name -->
                    <div class="win-probability">{bat_pct}%</div>
                    <div class="win-prob-label">{T("win_prob_label")}</div>
                    <div class="prob-bar-track">
                        <div class="prob-bar-fill" style="width:{bat_pct}%;"></div>
                    </div>
                    <div class="prob-bar-labels">
                        <span>0%</span><span>{bat_pct}%</span><span>100%</span>
                    </div>
                    <div class="metrics-row">
                        <div class="metric-chip">
                            <div class="metric-chip-value">{score}</div>
                            <!-- FIX 7: added closing </div> tags -->
                            <div class="metric-chip-label">{T("score_chip")}</div>
                        </div>
                        <div class="metric-chip">
                            <div class="metric-chip-value">{runs_left}</div>
                            <div class="metric-chip-label">{T("needed_chip")}</div>
                        </div>
                        <div class="metric-chip">
                            <div class="metric-chip-value">{balls_left}</div>
                            <div class="metric-chip-label">{T("balls_chip")}</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with res_col2:
            bowl_pct = round(lose * 100)
            st.markdown(f"""
                <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.07);
                            border-radius:24px; padding: clamp(20px, 5vw, 36px) clamp(16px, 4vw, 32px); position:relative;overflow:hidden;">
                    <div class="prediction-label">{T("bowling_badge")} · {t2['abbr']}</div>
                    <div style="font-family:'Cormorant Garamond',serif; font-size: clamp(16px, 4vw, 22px);
                                font-weight:500;color:#c8b870;margin-bottom: clamp(12px, 3vw, 16px);">
                        {bowling_team}
                    </div>
                    <div style="font-family:'DM Mono',monospace; font-size: clamp(36px, 8vw, 72px); font-weight:500;
                                color:rgba(200,185,140,0.55);line-height:1;margin-bottom:4px;">
                        {bowl_pct}%
                    </div>
                    <!-- FIX 10: was hardcoded "Win Probability", now uses T() -->
                    <div class="win-prob-label">{T("win_prob_label")}</div>
                    <div class="prob-bar-track">
                        <div style="height:100%;border-radius:100px;
                                    background:rgba(200,185,140,0.2);
                                    width:{bowl_pct}%;transition:width 0.8s ease;"></div>
                    </div>
                    <div class="prob-bar-labels">
                        <span>0%</span><span>{bowl_pct}%</span><span>100%</span>
                    </div>
                    <div class="metrics-row">
                        <div class="metric-chip">
                            <div class="metric-chip-value">{round(crr, 2)}</div>
                            <!-- FIX 7: added closing </div> tags -->
                            <div class="metric-chip-label">{T("crr_chip")}</div>
                        </div>
                        <div class="metric-chip">
                            <div class="metric-chip-value">{round(rrr, 2)}</div>
                            <div class="metric-chip-label">{T("rrr_chip")}</div>
                        </div>
                        <div class="metric-chip">
                            <div class="metric-chip-value">{10 - wickets}</div>
                            <div class="metric-chip-label">{T("inhand_chip")}</div>
                        </div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown('<div style="height: clamp(8px, 2vw, 16px);"></div>', unsafe_allow_html=True)
        verdict = batting_team if win > 0.5 else bowling_team
        conf = max(win, lose)
        conf_label = T("conf_high") if conf > 0.75 else T("conf_mod") if conf > 0.55 else T("conf_close")

        st.markdown(f"""
            <div style="background:rgba(212,175,55,0.03);border:1px solid rgba(212,175,55,0.1);
                        border-radius:16px; padding: clamp(12px, 3vw, 20px) clamp(16px, 4vw, 28px); display:flex;
                        flex-direction: column;
                        align-items:flex-start;justify-content:space-between;
                        gap: clamp(12px, 3vw, 20px);">
                <div>
                    <!-- FIX 8: added closing </div> to verdict label -->
                    <div style="font-size: clamp(7px, 1.5vw, 9px); letter-spacing:2px; text-transform:uppercase;
                                color:rgba(212,175,55,0.35);margin-bottom:6px;">{T("verdict_label")}</div>
                    <div style="font-family:'Cormorant Garamond',serif; font-size: clamp(16px, 4vw, 22px);
                                font-weight:500;color:#f0e8cc;">
                        {verdict} {T("verdict_suffix")}
                    </div>
                </div>
                <div style="text-align:left;width:100%;">
                    <!-- FIX 8: added closing </div> to confidence label -->
                    <div style="font-size: clamp(7px, 1.5vw, 9px); letter-spacing:2px; text-transform:uppercase;
                                color:rgba(212,175,55,0.35);margin-bottom:6px;">{T("confidence_label")}</div>
                    <div style="font-family:'DM Mono',monospace; font-size: clamp(14px, 3vw, 20px); color:#d4af37;">
                        {conf_label} · {round(conf*100)}%
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

        if len(st.session_state.prob_history) > 1:
            st.markdown('<div style="height:24px;"></div>', unsafe_allow_html=True)

            fig = px.line(
                x=list(range(1, len(st.session_state.prob_history)+1)),
                y=st.session_state.prob_history,
                # FIX 9: added missing comma after labels
                labels={'x': T("chart_x"), 'y': T("chart_y")},
                title=T("chart_title")
            )

            fig.update_layout(
                template="plotly_dark",
                plot_bgcolor="#080808",
                paper_bgcolor="#080808",
                font=dict(color="#e2dfd8")
            )

            st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)  # close main-pad