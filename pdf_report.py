# pdf_report.py — PDF Session Report Generator for ISL Project
# Generates professional PDF reports with charts, stats, and sign history
# Usage: python pdf_report.py (GUI) or import and call generate_report()
import os
import json
import time
from datetime import datetime
from collections import defaultdict

def generate_report(session_data=None, output_path=None, progress_file="isl_progress.json"):
    """
    Generate a professional HTML/PDF session report.

    Args:
        session_data: dict with keys: sentence, history, quiz_score, quiz_total,
                      model_type, total_signs, duration_minutes
        output_path: path for output file (.html)
        progress_file: path to progress tracker JSON
    """
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"ISL_Report_{timestamp}.html"

    # Load progress data
    progress = {}
    if os.path.exists(progress_file):
        try:
            with open(progress_file, 'r') as f:
                progress = json.load(f)
        except Exception:
            pass

    # Compute stats
    per_sign = progress.get('per_sign', {})
    sign_counts = {s: d.get('count', 0) for s, d in per_sign.items()}
    total_detected = progress.get('total_signs_detected', 0)
    total_sessions = progress.get('total_sessions', 0)
    total_minutes = progress.get('total_time_minutes', 0)
    streak = progress.get('daily_streak', 0)
    achievements = progress.get('achievements', [])
    quiz_history = progress.get('quiz_history', [])

    # Top signs
    sorted_signs = sorted(sign_counts.items(), key=lambda x: x[1], reverse=True)
    top_10 = sorted_signs[:10]

    # Mastered vs struggling
    mastered = [s for s, d in per_sign.items()
                if d.get('count', 0) >= 50 and (d.get('total_conf', 0) / max(d.get('count', 1), 1)) >= 0.7]
    struggling = [s for s, d in per_sign.items()
                  if d.get('count', 0) >= 10 and (d.get('total_conf', 0) / max(d.get('count', 1), 1)) < 0.5]

    # Session data
    if session_data is None:
        session_data = {
            'sentence': [],
            'history': [],
            'quiz_score': 0,
            'quiz_total': 0,
            'model_type': 'YOLO11',
            'total_signs': len(per_sign),
            'duration_minutes': 0
        }

    # Build chart data for JS
    chart_labels = json.dumps([s.upper() for s, _ in top_10])
    chart_values = json.dumps([c for _, c in top_10])

    quiz_dates = [q.get('date', '')[:10] for q in quiz_history[-20:]]
    quiz_scores = [q.get('accuracy', 0) for q in quiz_history[-20:]]

    # Daily activity
    daily_log = progress.get('daily_log', {})
    daily_dates = json.dumps(list(daily_log.keys())[-30:])
    daily_counts = json.dumps(list(daily_log.values())[-30:])

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ISL Recognition Report — {datetime.now().strftime('%B %d, %Y')}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }}

        .container {{ max-width: 1100px; margin: 0 auto; }}

        .header {{
            text-align: center;
            padding: 40px 20px;
            background: rgba(255,255,255,0.05);
            border-radius: 20px;
            border: 1px solid rgba(187,134,252,0.3);
            margin-bottom: 30px;
        }}
        .header h1 {{
            font-size: 2.5rem;
            background: linear-gradient(90deg, #bb86fc, #03dac6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 10px;
        }}
        .header .subtitle {{ color: #b0b0b0; font-size: 1rem; }}

        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }}
        .metric-card {{
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 20px;
            text-align: center;
            transition: transform 0.3s;
        }}
        .metric-card:hover {{ transform: translateY(-5px); border-color: rgba(187,134,252,0.5); }}
        .metric-value {{ font-size: 2.2rem; font-weight: 700; color: #03dac6; }}
        .metric-label {{ font-size: 0.85rem; color: #b0b0b0; margin-top: 5px; }}

        .section {{
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 16px;
            padding: 25px;
            margin-bottom: 20px;
        }}
        .section h2 {{
            font-size: 1.3rem;
            color: #bb86fc;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .chart-container {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 20px;
        }}
        .chart-box {{
            background: rgba(0,0,0,0.2);
            border-radius: 12px;
            padding: 15px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 10px 15px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.08);
        }}
        th {{ color: #bb86fc; font-weight: 600; font-size: 0.85rem; text-transform: uppercase; }}
        td {{ color: #e0e0e0; font-size: 0.9rem; }}
        tr:hover {{ background: rgba(255,255,255,0.03); }}

        .badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            margin: 2px;
        }}
        .badge-green {{ background: rgba(3,218,198,0.2); color: #03dac6; }}
        .badge-red {{ background: rgba(207,102,121,0.2); color: #cf6679; }}
        .badge-purple {{ background: rgba(187,134,252,0.2); color: #bb86fc; }}
        .badge-gold {{ background: rgba(251,192,45,0.2); color: #fbc02d; }}

        .progress-bar {{
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            height: 8px;
            overflow: hidden;
            margin-top: 5px;
        }}
        .progress-fill {{
            height: 100%;
            border-radius: 10px;
            background: linear-gradient(90deg, #bb86fc, #03dac6);
            transition: width 0.5s;
        }}

        .footer {{
            text-align: center;
            padding: 20px;
            color: #666;
            font-size: 0.8rem;
        }}

        @media print {{
            body {{ background: white; color: #333; }}
            .metric-card, .section {{ border: 1px solid #ddd; }}
            .metric-value {{ color: #00897b; }}
            .section h2 {{ color: #7b1fa2; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤟 ISL Recognition Report</h1>
            <p class="subtitle">
                Indian Sign Language Detection System — Generated {datetime.now().strftime('%B %d, %Y at %I:%M %p')}
            </p>
        </div>

        <div class="metrics">
            <div class="metric-card">
                <div class="metric-value">{total_detected}</div>
                <div class="metric-label">Total Signs Detected</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{total_sessions}</div>
                <div class="metric-label">Sessions Completed</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{round(total_minutes, 1)}</div>
                <div class="metric-label">Minutes Practiced</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{streak} 🔥</div>
                <div class="metric-label">Day Streak</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len(per_sign)}</div>
                <div class="metric-label">Signs Learned</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{len(mastered)}</div>
                <div class="metric-label">Signs Mastered</div>
            </div>
        </div>

        <div class="chart-container">
            <div class="chart-box">
                <canvas id="signChart"></canvas>
            </div>
            <div class="chart-box">
                <canvas id="dailyChart"></canvas>
            </div>
        </div>

        <div class="section">
            <h2>🏆 Achievements</h2>
            {''.join(f'<span class="badge badge-gold">🏅 {a}</span>' for a in achievements) if achievements else '<p style="color:#666">No achievements yet — keep practicing!</p>'}
        </div>

        <div class="section">
            <h2>✅ Mastered Signs</h2>
            {''.join(f'<span class="badge badge-green">{s.upper()}</span>' for s in mastered) if mastered else '<p style="color:#666">Keep practicing to master signs!</p>'}
        </div>

        <div class="section">
            <h2>⚠️ Signs to Practice</h2>
            {''.join(f'<span class="badge badge-red">{s.upper()}</span>' for s in struggling) if struggling else '<p style="color:#666">All signs looking great!</p>'}
        </div>

        <div class="section">
            <h2>📊 Per-Sign Statistics</h2>
            <table>
                <tr><th>Sign</th><th>Detections</th><th>Avg Confidence</th><th>Status</th></tr>
                {''.join(f"""<tr>
                    <td>{s.upper()}</td>
                    <td>{d.get('count',0)}</td>
                    <td>{(d.get('total_conf',0)/max(d.get('count',1),1))*100:.0f}%</td>
                    <td><span class="badge {'badge-green' if d.get('mastered') else 'badge-purple'}">
                        {'Mastered' if d.get('mastered') else 'Learning'}
                    </span></td>
                </tr>""" for s, d in sorted(per_sign.items(), key=lambda x: x[1].get('count',0), reverse=True))}
            </table>
        </div>

        <div class="section">
            <h2>📝 Quiz History (Recent)</h2>
            <table>
                <tr><th>Date</th><th>Score</th><th>Accuracy</th></tr>
                {''.join(f"""<tr>
                    <td>{q.get('date','')[:10]}</td>
                    <td>{q.get('score',0)}/{q.get('total',0)}</td>
                    <td>{q.get('accuracy',0):.0f}%</td>
                </tr>""" for q in reversed(quiz_history[-10:]))}
            </table>
        </div>

        <div class="footer">
            <p>Generated by ISL Recognition System | Model: {session_data.get('model_type', 'YOLO11')} | Python 3.11</p>
            <p>Indian Sign Language Detection — Final Year Project</p>
        </div>
    </div>

    <script>
        new Chart(document.getElementById('signChart'), {{
            type: 'doughnut',
            data: {{
                labels: {chart_labels},
                datasets: [{{ data: {chart_values},
                    backgroundColor: ['#03dac6','#bb86fc','#ffb74d','#64b5f6','#cf6679',
                                      '#fbc02d','#81c784','#ce93d8','#90caf9','#ffcc02'] }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{ display: true, text: 'Top Signs Detected', color: '#bb86fc' }},
                    legend: {{ labels: {{ color: '#b0b0b0' }} }}
                }}
            }}
        }});

        new Chart(document.getElementById('dailyChart'), {{
            type: 'bar',
            data: {{
                labels: {daily_dates},
                datasets: [{{ label: 'Signs per Day', data: {daily_counts},
                    backgroundColor: 'rgba(3,218,198,0.5)', borderColor: '#03dac6', borderWidth: 1 }}]
            }},
            options: {{
                responsive: true,
                plugins: {{
                    title: {{ display: true, text: 'Daily Activity', color: '#bb86fc' }},
                    legend: {{ labels: {{ color: '#b0b0b0' }} }}
                }},
                scales: {{
                    x: {{ ticks: {{ color: '#888' }} }},
                    y: {{ ticks: {{ color: '#888' }}, beginAtZero: true }}
                }}
            }}
        }});
    </script>
</body>
</html>"""

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"✅ Report generated: {output_path}")
    return output_path


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--open":
        path = generate_report()
        os.startfile(path)
    else:
        path = generate_report()
        print(f"Open in browser: {path}")
        print("Or run: python pdf_report.py --open")
