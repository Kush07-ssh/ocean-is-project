import streamlit as st
import pandas as pd
import altair as alt
from ReportGeneration import generate_html_report
from OceanModel import llm_analysis
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os

#  1. PAGE CONFIGURATION
st.set_page_config(page_title="Personality Assessment", layout="wide")

#  2. CUSTOM CSS (Kept exactly as per your requirement)
st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    h1, h2, h3, h4, h5, h6, p, li, div { color: #000000; }
    div[role="radiogroup"] label div p {
        color: #000000 !important;
        font-size: 18px !important;
        font-weight: 500 !important;
    }
    div[role="radiogroup"] { gap: 10px; }
    .question-box {
        background-color: #ffffff;
        padding: 40px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
        text-align: center;
        margin-bottom: 30px;
    }
    div.stButton > button, div.stDownloadButton > button {
        background-color: #2c3e50;
        color: #ffffff !important;
        border: none;
        height: 50px;
        font-size: 16px;
        font-weight: 600;
        border-radius: 8px;
        width: 100%;
        transition: all 0.3s ease;
    }
    div.stButton > button:hover, div.stDownloadButton > button:hover {
        background-color: #34495e;
        color: #ffffff !important;
        transform: translateY(-2px);
    }
    div.stButton > button p, div.stDownloadButton > button p { color: #ffffff !important; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)


# --- 3. HELPER FUNCTIONS FOR VIDEO/PLOTS ---

def get_frame(video_path, timestamp_sec, is_thermal=False):
    if not os.path.exists(video_path): return None
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fps * timestamp_sec))
    success, frame = cap.read()
    cap.release()
    if success:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if is_thermal:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    return None


def create_emotion_plot(valence_data, arousal_data, current_idx):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(4, 4))
    fig.patch.set_facecolor('#161b22')
    ax.set_facecolor('#161b22')
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, linestyle='--', alpha=0.1, color='#8b949e')
    ax.axhline(0, color='#30363d', linewidth=1.5)
    ax.axvline(0, color='#30363d', linewidth=1.5)
    ax.plot(valence_data, arousal_data, color='#58a6ff', alpha=0.15, linewidth=1)
    cur_val, cur_ar = valence_data[current_idx], arousal_data[current_idx]
    ax.scatter([cur_val], [cur_ar], color='#ff7b72', s=120, zorder=5, edgecolors='white')
    labels = [("Excited", 0.7, 0.7, "#7ee787"), ("Stressed", -0.7, 0.7, "#ff7b72"),
              ("Depressed", -0.7, -0.7, "#a5d6ff"), ("Relaxed", 0.7, -0.7, "#d2a8ff")]
    for text, x, y, col in labels:
        ax.text(x, y, text, fontsize=9, color=col, alpha=0.7, ha='center', weight='bold')
    plt.tight_layout()
    return fig


# --- 4. QUESTION DATA ---
questions = [
    "Is talkative", "Tends to find fault with others", "Does a thorough job",
    "Is depressed, blue", "Is original, comes up with new ideas", "Is reserved",
    "Is helpful and unselfish with others", "Can be somewhat careless",
    "Is relaxed, handles stress well", "Is curious about many different things",
    "Is full of energy", "Starts quarrels with others", "Is a reliable worker",
    "Can be tense", "Is ingenious, a deep thinker", "Generates a lot of enthusiasm",
    "Has a forgiving nature", "Tends to be disorganized", "Worries a lot",
    "Has an active imagination", "Tends to be quiet", "Is generally trusting",
    "Tends to be lazy", "Is emotionally stable, not easily upset", "Is inventive",
    "Has an assertive personality", "Can be cold and aloof", "Perseveres until the task is finished",
    "Can be moody", "Values artistic, aesthetic experiences", "Is sometimes shy, inhibited",
    "Is considerate and kind to almost everyone", "Does things efficiently",
    "Remains calm in tense situations", "Prefers work that is routine", "Is outgoing, sociable",
    "Is sometimes rude to others", "Makes plans and follows through with them", "Gets nervous easily",
    "Likes to reflect, play with ideas", "Has few artistic interests", "Likes to cooperate with others",
    "Is easily distracted", "Is sophisticated in art, music, or literature"
]

scoring_map = {
    "Extraversion": [1, "6R", 11, 16, "21R", 26, "31R", 36],
    "Agreeableness": ["2R", 7, "12R", 17, 22, "27R", 32, "37R", 42],
    "Conscientiousness": [3, "8R", 13, "18R", "23R", 28, 33, 38, "43R"],
    "Neuroticism": [4, "9R", 14, 19, "24R", 29, "34R", 39],
    "Openness": [5, 10, 15, 20, 25, 30, "35R", 40, "41R", 44]
}

options = ["Disagree Strongly", "Disagree a Little", "Neither Agree nor Disagree", "Agree a Little", "Agree Strongly"]
values = [1, 2, 3, 4, 5]

if 'current_step' not in st.session_state: st.session_state.current_step = 0
if 'answers' not in st.session_state: st.session_state.answers = {}


def get_score(idx):
    if isinstance(idx, str) and 'R' in idx:
        return 6 - st.session_state.answers.get(int(idx.replace('R', '')), 3)
    elif isinstance(idx, int):
        return st.session_state.answers.get(idx, 3)
    return 0


# 5. MAIN LAYOUT
def main():
    col_l, col_center, col_r = st.columns([1, 6, 1])
    with col_center:
        st.markdown("<h1 style='text-align: center; color: black;'>Personality Assessment</h1>", unsafe_allow_html=True)
        total_q = len(questions)
        current_idx = st.session_state.current_step

        if current_idx < total_q:
            st.progress((current_idx) / total_q)
            st.markdown(f"""<div class='question-box'>
                <p style='color: #888;'>Statement {current_idx + 1} of {total_q}</p>
                <h2 style='color: #2c3e50; font-size: 32px;'>"{questions[current_idx]}"</h2>
            </div>""", unsafe_allow_html=True)
            choice = st.radio("Response", options, index=2, key=f"q_{current_idx}", horizontal=True,
                              label_visibility="collapsed")

            c1, c2, c3 = st.columns([1, 4, 1])
            with c1:
                if st.button("Previous") and st.session_state.current_step > 0:
                    st.session_state.current_step -= 1;
                    st.rerun()
            with c3:
                if st.button("Next"):
                    st.session_state.answers[current_idx + 1] = values[options.index(choice)]
                    st.session_state.current_step += 1;
                    st.rerun()
        else:
            display_results()


def display_results():
    normalized_results = {}
    for trait, items in scoring_map.items():
        raw_score = sum(get_score(i) for i in items)
        normalized_results[trait] = round(((raw_score - len(items)) / (len(items) * 4)) * 100, 1)

    st.session_state.final_results = normalized_results
    st.success("Assessment Complete.")
    df_scores = pd.DataFrame(list(normalized_results.items()), columns=['Trait', 'Score (%)'])

    col1, col2 = st.columns([3, 2])
    with col1:
        # Altair Chart - Style maintained as requested
        chart = alt.Chart(df_scores).mark_bar(color='#4A90E2').encode(
            x=alt.X('Trait', axis=alt.Axis(labelAngle=0, labelColor='black', titleColor='black')),
            y=alt.Y('Score (%)', scale=alt.Scale(domain=[0, 100]),
                    axis=alt.Axis(labelColor='black', titleColor='black')),
            tooltip=['Trait', 'Score (%)']
        ).properties(height=400).configure_axis(labelFontSize=12, titleFontSize=14).configure_view(
            strokeWidth=0).configure(background='#ffffff')
        st.altair_chart(chart, use_container_width=True)
    with col2:
        st.dataframe(df_scores.style.set_properties(
            **{'color': 'black', 'background-color': '#ffffff', 'border': '1px solid #eee'}), use_container_width=True,
                     hide_index=True)

    if st.button("Generate Full Analysis"):
        with st.spinner("Analyzing Behavior & Generating Report..."):
            ocean_order = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
            scores_array = [normalized_results.get(trait, 0) for trait in ocean_order]
            scores_dict = dict(zip(ocean_order, scores_array))

            # Temporal Data Processing
            arousal_data = np.load("Emotional_Behaviour/arousal.npy");
            valence_data = np.load("Emotional_Behaviour/valence.npy")
            min_len = min(len(arousal_data), len(valence_data))
            indices = sorted(random.sample(range(min_len), 10))
            cap = cv2.VideoCapture("Emotional_Behaviour/video.mp4");
            duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30);
            cap.release()

            report_snapshots = []
            for idx in indices:
                ts = (idx / min_len) * duration
                report_snapshots.append({
                    'time': f"{int(ts // 60)}:{int(ts % 60):02d}", 'valence': valence_data[idx],
                    'arousal': arousal_data[idx],
                    'rgb': get_frame("Emotional_Behaviour/video.mp4", ts), 'thermal': get_frame(
                        "Emotional_Behaviour/video.mp4", ts, is_thermal=True),
                    'plot': create_emotion_plot(valence_data, arousal_data, idx)
                })

            llm_analysis_text = llm_analysis(scores_dict)
            # This calls the updated function where personality is at the TOP
            html_content = generate_html_report("Candidate Name", scores_array, llm_analysis_text, report_snapshots)

            st.download_button(label="Download Full Report HTML", data=html_content,
                               file_name="Personality_Report.html", mime="text/html")
            st.components.v1.html(html_content, height=1200, scrolling=True)


if __name__ == "__main__":
    main()