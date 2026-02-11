import markdown2
import matplotlib.pyplot as plt
import numpy as np
from jinja2 import Template
import io
import base64
import cv2


def ndarray_to_base64(img_array):
    """Converts numpy video frames to base64 for the report."""
    if img_array is None: return ""
    bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', bgr)
    return base64.b64encode(buffer).decode('utf-8')


def fig_to_base64(fig):
    """Converts matplotlib emotion plots to base64 for the report."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# --- YOUR ORIGINAL RADAR CHART ENGINE ---

def create_radar_chart_base64(scores):
    labels = list(scores.keys())
    values = list(scores.values())
    values += values[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    plt.xticks(angles[:-1], labels, color='#2c3e50', size=11, fontweight='bold')
    ax.set_rlabel_position(30)
    plt.yticks([20, 40, 60, 80, 100], ["20", "40", "60", "80", "100"], color="grey", size=9)
    plt.ylim(0, 100)

    ax.plot(angles, values, linewidth=2.5, linestyle='solid', color='#2c3e50')
    ax.fill(angles, values, alpha=0.3, color='#3498db')
    plt.title("Normalized Personality Profile", y=1.08, fontsize=14, fontweight='bold', color='#2c3e50')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', transparent=True)
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# --- INTEGRATED REPORT GENERATION FUNCTION ---

def generate_html_report(candidate_name, scores_array, llm_analysis_text, temporal_snapshots=None):
    # 1. Map Big Five Scores
    scores_array = (scores_array + [0] * 5)[:5]
    traits = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
    scores_dict = {traits[i]: round(scores_array[i], 1) for i in range(5)}

    # 2. Generate Personality Chart
    chart_base64 = create_radar_chart_base64(scores_dict)

    # 3. Process Snapshots (Integrating your Thermal/Arousal/Valence data)
    processed_snaps = []
    if temporal_snapshots:
        for s in temporal_snapshots:
            processed_snaps.append({
                "time": s['time'],
                "v": round(s['valence'], 2),
                "a": round(s['arousal'], 2),
                "rgb": ndarray_to_base64(s['rgb']),
                "therm": ndarray_to_base64(s['thermal']),
                "plot": fig_to_base64(s['plot'])
            })

    # 4. Process Markdown Analysis
    html_details = markdown2.markdown(llm_analysis_text)

    # 5. YOUR ORIGINAL CSS (With Snapshot addition)
    css_style = """
    :root {
        --primary-color: #2c3e50;
        --secondary-color: #3498db;
        --accent-color: #e67e22;
        --bg-color: #f8f9fa;
        --card-bg: #ffffff;
        --text-color: #333333;
    }
    body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: var(--text-color); line-height: 1.6; max-width: 950px; margin: 0 auto; padding: 40px 20px; background-color: var(--bg-color); }
    .report-container { background-color: var(--card-bg); padding: 40px; border-radius: 12px; box-shadow: 0 8px 30px rgba(0,0,0,0.08); }
    .header { text-align: center; border-bottom: 2px solid #ecf0f1; padding-bottom: 25px; margin-bottom: 40px; }
    .header h1 { color: var(--primary-color); margin: 0; font-size: 32px; font-weight: 800; }
    .header h2 { color: var(--accent-color); margin-top: 10px; font-size: 18px; text-transform: uppercase; }

    .snapshot-section { margin-bottom: 40px; }
    .snapshot-card { display: flex; gap: 15px; border: 1px solid #ecf0f1; border-radius: 10px; padding: 15px; margin-bottom: 15px; align-items: center; page-break-inside: avoid; }
    .snap-info { flex: 0.7; border-right: 2px solid #ecf0f1; padding-right: 10px; font-size: 14px; color: var(--primary-color); }
    .snap-box { flex: 2; text-align: center; }
    .snap-box img { width: 100%; border-radius: 6px; border: 1px solid #eee; }
    .snap-box span { font-size: 10px; font-weight: bold; color: #7f8c8d; display: block; margin-top: 5px; }

    .top-layout { display: flex; align-items: center; gap: 40px; margin-bottom: 40px; }
    .chart-section { flex: 1; text-align: center; }
    .score-card { flex: 1; background-color: var(--bg-color); padding: 30px; border-radius: 10px; border-top: 4px solid var(--primary-color); }
    .score-row { display: flex; justify-content: space-between; margin-bottom: 12px; font-size: 16px; }
    .score-label { font-weight: 600; color: var(--primary-color); }
    .score-value { font-weight: bold; color: var(--secondary-color); }
    .analysis-section { margin-top: 40px; padding-top: 30px; border-top: 2px solid #ecf0f1; }
    @media print { .no-print { display: none; } .report-container { box-shadow: none; border: none; width: 100% !important; } }
    """

    # Inside your generate_html_report function in ReportGeneration.py

    template_str = """
    <!DOCTYPE html>
    <html>
    <head><style>{{ css }}</style></head>
    <body>
        <div class="report-container">
            <div class="header">
                <h1>Comprehensive Personality Assessment</h1>
                <h2>Prepared for: Harshit</h2>
            </div>

            <div class="top-layout">
                <div class="chart-section">
                    <img src="data:image/png;base64,{{ chart_data }}" alt="Radar Chart">
                </div>
                <div class="score-card">
                    <h3>Trait Breakdown (0-100%)</h3>
                    {% for trait, score in scores_dict.items() %}
                    <div class="score-row">
                        <span class="score-label">{{ trait }}</span>
                        <span class="score-value">{{ score }}%</span>
                    </div>
                    {% endfor %}
                </div>
            </div>

            <div class="analysis-section">
                {{ details_html }}
            </div>

            {% if snapshots %}
            <div class="snapshot-section" style="margin-top: 50px; border-top: 2px solid #ecf0f1; padding-top: 30px;">
                <h3 style="color: #2c3e50; margin-bottom: 20px;">Temporal Emotion & Behavioral Analysis</h3>
                {% for snap in snapshots %}
                <div class="snapshot-card" style="display: flex; gap: 15px; border: 1px solid #ecf0f1; border-radius: 10px; padding: 15px; margin-bottom: 15px; align-items: center;">
                    <div class="snap-info" style="flex: 0.7; border-right: 2px solid #ecf0f1; padding-right: 10px;">
                        <strong>{{ snap.time }}</strong><br>V: {{ snap.v }}<br>A: {{ snap.a }}
                    </div>
                    <div class="snap-box" style="flex: 2; text-align: center;"><img src="data:image/png;base64,{{ snap.rgb }}" style="width:100%;"><span>OPTICAL</span></div>
                    <div class="snap-box" style="flex: 2; text-align: center;"><img src="data:image/png;base64,{{ snap.therm }}" style="width:100%;"><span>THERMAL</span></div>
                    <div class="snap-box" style="flex: 2; text-align: center;"><img src="data:image/png;base64,{{ snap.plot }}" style="width:100%;"><span>MAPPING</span></div>
                </div>
                {% endfor %}
            </div>
            {% endif %}
        </div>
    </body>
    </html>
    """

    return Template(template_str).render(
        css=css_style, name=candidate_name, chart_data=chart_base64,
        scores_dict=scores_dict, details_html=html_details, snapshots=processed_snaps
    )