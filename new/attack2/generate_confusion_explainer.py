import os
import json

INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.json")

def load_index_data():
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def build_html_content(index_data, counts=None):
    scenarios = index_data["scenarios"]
    html_cfg = index_data["html_strings"]
    headers = index_data["plot_strings"]["section_headers"]

    scenarios_json_str = json.dumps(scenarios)
    counts_json_str = json.dumps(counts if counts else {})

    html = "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
    html += "<meta charset=\"UTF-8\">\n"
    html += "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">\n"
    html += "<title>" + html_cfg["page_title"] + "</title>\n"
    html += "<style>\n"
    html += "* { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }\n"
    html += "body { background: #0B1120; color: #F8FAFC; padding: 24px 20px; line-height: 1.5; }\n"
    html += ".container { max-width: 1200px; margin: 0 auto; }\n"
    html += "header { text-align: center; margin-bottom: 24px; }\n"
    html += "h1 { font-size: 26px; font-weight: 800; color: #F8FAFC; margin-bottom: 6px; }\n"
    html += ".subtitle { font-size: 14px; color: #94A3B8; max-width: 780px; margin: 0 auto 10px; }\n"
    html += ".instruction { font-size: 12.5px; color: #38BDF8; font-weight: 600; }\n"
    html += ".grid-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 14px; margin-bottom: 22px; }\n"
    html += ".card { background: #1E293B; border: 2px solid #334155; border-radius: 12px; padding: 16px; cursor: pointer; transition: all 0.2s ease; }\n"
    html += ".card:hover { transform: translateY(-3px); box-shadow: 0 8px 20px rgba(0,0,0,0.3); }\n"
    html += ".card.active { border-color: var(--accent); background: #1E293B; box-shadow: 0 0 0 2px var(--accent); }\n"
    html += ".card-badge { display: inline-block; font-size: 10.5px; font-weight: 800; padding: 3px 8px; border-radius: 6px; margin-bottom: 8px; }\n"
    html += ".card-title { font-size: 15px; font-weight: 700; color: #FFFFFF; margin-bottom: 4px; }\n"
    html += ".card-label { font-size: 12px; color: #94A3B8; margin-bottom: 10px; min-height: 32px; }\n"
    html += ".card-meta { display: flex; justify-content: space-between; font-size: 11px; color: #CBD5E1; padding-top: 8px; border-top: 1px solid #334155; }\n"
    html += ".stage-container { background: #111827; border: 1px solid #1F2937; border-radius: 14px; padding: 22px; margin-bottom: 24px; position: relative; overflow: hidden; }\n"
    html += ".stage-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }\n"
    html += ".stage-title { font-size: 13px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; color: #94A3B8; }\n"
    html += ".stage-replay-btn { background: #1E293B; border: 1px solid #475569; color: #F8FAFC; padding: 6px 14px; border-radius: 6px; font-size: 11.5px; font-weight: 700; cursor: pointer; transition: all 0.2s; }\n"
    html += ".stage-replay-btn:hover { background: #334155; border-color: #38BDF8; }\n"
    html += ".anim-canvas { position: relative; height: 180px; background: #0F172A; border-radius: 12px; border: 1px solid #1E293B; margin-bottom: 20px; overflow: hidden; display: flex; align-items: center; justify-content: space-between; padding: 0 40px; }\n"
    html += ".bus-track { position: absolute; left: 60px; right: 60px; height: 4px; background: #334155; z-index: 1; }\n"
    html += ".bus-track-glow { position: absolute; left: 60px; right: 60px; height: 2px; background: #38BDF8; opacity: 0.4; z-index: 2; }\n"
    html += ".anim-station { position: relative; z-index: 5; background: #1E293B; border: 2px solid #334155; border-radius: 10px; width: 140px; height: 120px; display: flex; flex-direction: column; align-items: center; justify-content: center; text-align: center; padding: 10px; }\n"
    html += ".station-icon { font-size: 11px; font-weight: 800; padding: 3px 8px; border-radius: 4px; margin-bottom: 6px; }\n"
    html += ".station-title { font-size: 12px; font-weight: 700; color: #F8FAFC; margin-bottom: 3px; }\n"
    html += ".station-sub { font-size: 10px; color: #94A3B8; }\n"
    html += ".scanner-beam { position: absolute; top: 0; bottom: 0; width: 3px; background: #38BDF8; box-shadow: 0 0 12px #38BDF8; opacity: 0.8; animation: sweep 2s infinite ease-in-out; }\n"
    html += "@keyframes sweep { 0% { left: 8px; } 50% { left: 128px; } 100% { left: 8px; } }\n"
    html += ".packet-carrier { position: absolute; top: 76px; left: 150px; z-index: 10; display: flex; align-items: center; gap: 6px; }\n"
    html += ".packet-box { padding: 5px 12px; border-radius: 6px; font-size: 11px; font-weight: 800; box-shadow: 0 4px 12px rgba(0,0,0,0.5); }\n"
    html += ".barrier-gate { position: absolute; top: 25px; bottom: 25px; width: 6px; border-radius: 3px; z-index: 8; transition: all 0.3s ease; }\n"
    html += ".score-meter-wrap { width: 100%; margin-top: 6px; background: #0B1120; border-radius: 4px; height: 8px; overflow: hidden; position: relative; }\n"
    html += ".score-meter-fill { height: 100%; border-radius: 4px; transition: width 0.6s ease; }\n"
    html += ".score-meter-thresh { position: absolute; top: 0; bottom: 0; left: 47%; width: 2px; background: #EF4444; z-index: 3; }\n"
    html += ".details-box { display: grid; grid-template-columns: repeat(auto-fit, minmax(270px, 1fr)); gap: 14px; background: #1E293B; border-radius: 10px; padding: 16px; border-left: 4px solid #38BDF8; }\n"
    html += ".detail-item h4 { font-size: 11px; text-transform: uppercase; color: #94A3B8; margin-bottom: 4px; font-weight: 700; }\n"
    html += ".detail-item p { font-size: 12.5px; color: #E2E8F0; line-height: 1.45; }\n"
    html += ".table-section { background: #111827; border: 1px solid #1F2937; border-radius: 14px; padding: 22px; margin-bottom: 24px; overflow-x: auto; }\n"
    html += "table { width: 100%; border-collapse: collapse; font-size: 12px; text-align: left; }\n"
    html += "th { background: #1E293B; color: #94A3B8; font-weight: 700; padding: 10px 12px; border-bottom: 2px solid #334155; text-transform: uppercase; font-size: 11px; }\n"
    html += "td { padding: 12px 12px; border-bottom: 1px solid #1F2937; vertical-align: top; color: #E2E8F0; }\n"
    html += "tr:hover td { background: #1E293B; }\n"
    html += ".tag { display: inline-block; font-size: 10px; font-weight: 700; padding: 2px 7px; border-radius: 4px; }\n"
    html += "footer { text-align: center; font-size: 11px; color: #64748B; padding: 10px; }\n"
    html += "</style>\n</head>\n<body>\n"

    html += "<div class=\"container\">\n"
    html += "<header>\n"
    html += "  <h1>" + html_cfg["headline"] + "</h1>\n"
    html += "  <p class=\"subtitle\">" + html_cfg["subheadline"] + "</p>\n"
    html += "  <p class=\"instruction\">" + html_cfg["tab_instruction"] + "</p>\n"
    html += "</header>\n"

    html += "<div class=\"grid-cards\">\n"
    order = ["true_negative", "false_positive", "false_negative", "true_positive"]
    for key in order:
        sc = scenarios[key]
        color = sc["color_theme"]
        bg = sc["bg_theme"]
        count_display = ""
        if counts and key in counts:
            count_display = "Count: " + str(counts[key])
        html += "  <div class=\"card\" id=\"card-" + key + "\" style=\"--accent: " + color + ";\" onclick=\"selectScenario('" + key + "')\">\n"
        html += "    <div class=\"card-badge\" style=\"background: " + bg + "; color: " + color + ";\">[" + sc["id"] + "] " + sc["outcome_type"].upper() + "</div>\n"
        html += "    <div class=\"card-title\">" + sc["name"] + "</div>\n"
        html += "    <div class=\"card-label\">" + sc["short_label"] + "</div>\n"
        html += "    <div class=\"card-meta\"><span>" + sc["vehicle_state"] + "</span><span>" + count_display + "</span></div>\n"
        html += "  </div>\n"
    html += "</div>\n"

    html += "<div class=\"stage-container\">\n"
    html += "  <div class=\"stage-header\">\n"
    html += "    <div class=\"stage-title\" id=\"stage-heading\">Live CAN Traffic Simulation</div>\n"
    html += "    <button class=\"stage-replay-btn\" onclick=\"replaySimulation()\">Replay Animation</button>\n"
    html += "  </div>\n"

    html += "  <div class=\"anim-canvas\" id=\"anim-canvas\">\n"
    html += "    <div class=\"bus-track\"></div>\n"
    html += "    <div class=\"bus-track-glow\"></div>\n"

    html += "    <div class=\"anim-station\" id=\"station-ecu\">\n"
    html += "      <div class=\"station-icon\" id=\"st-ecu-badge\" style=\"background: #064E3B; color: #10B981;\">NODE</div>\n"
    html += "      <div class=\"station-title\">Electronic Control Unit</div>\n"
    html += "      <div class=\"station-sub\" id=\"st-ecu-sub\">Transmitting Telemetry</div>\n"
    html += "    </div>\n"

    html += "    <div class=\"anim-station\" style=\"position: relative; overflow: hidden;\">\n"
    html += "      <div class=\"scanner-beam\"></div>\n"
    html += "      <div class=\"station-icon\" style=\"background: #0C4A6E; color: #38BDF8;\">AI DETECTOR</div>\n"
    html += "      <div class=\"station-title\">Graph Transformer</div>\n"
    html += "      <div class=\"station-sub\" id=\"sim-score-text\">Score: 0.24</div>\n"
    html += "      <div class=\"score-meter-wrap\">\n"
    html += "        <div class=\"score-meter-thresh\"></div>\n"
    html += "        <div class=\"score-meter-fill\" id=\"sim-score-fill\" style=\"width: 25%; background: #10B981;\"></div>\n"
    html += "      </div>\n"
    html += "    </div>\n"

    html += "    <div class=\"barrier-gate\" id=\"sim-barrier\" style=\"right: 215px; background: #10B981; height: 10px;\"></div>\n"

    html += "    <div class=\"anim-station\" id=\"station-gateway\">\n"
    html += "      <div class=\"station-icon\" id=\"st-gtw-badge\" style=\"background: #064E3B; color: #10B981;\">GATEWAY</div>\n"
    html += "      <div class=\"station-title\">Security Gateway</div>\n"
    html += "      <div class=\"station-sub\" id=\"sim-gateway-action\">Allowed (Normal)</div>\n"
    html += "    </div>\n"

    html += "    <div class=\"packet-carrier\" id=\"sim-packet\">\n"
    html += "      <div class=\"packet-box\" id=\"sim-packet-box\" style=\"background: #10B981; color: #064E3B;\">ID: 0x0D0 (Normal)</div>\n"
    html += "    </div>\n"
    html += "  </div>\n"

    html += "  <div class=\"details-box\" id=\"details-box\">\n"
    html += "    <div class=\"detail-item\">\n"
    html += "      <h4>" + headers["vehicle_status"] + "</h4>\n"
    html += "      <p id=\"sim-veh-status\">-</p>\n"
    html += "    </div>\n"
    html += "    <div class=\"detail-item\">\n"
    html += "      <h4>" + headers["model_decision"] + "</h4>\n"
    html += "      <p id=\"sim-decision\">-</p>\n"
    html += "    </div>\n"
    html += "    <div class=\"detail-item\">\n"
    html += "      <h4>" + headers["system_outcome"] + "</h4>\n"
    html += "      <p id=\"sim-consequence\">-</p>\n"
    html += "    </div>\n"
    html += "    <div class=\"detail-item\">\n"
    html += "      <h4>" + headers["model_role"] + "</h4>\n"
    html += "      <p id=\"sim-role\">-</p>\n"
    html += "    </div>\n"
    html += "  </div>\n"
    html += "</div>\n"

    html += "<div class=\"table-section\">\n"
    html += "  <div style=\"font-size: 13px; font-weight: 700; text-transform: uppercase; color: #94A3B8; margin-bottom: 14px;\">" + html_cfg["table_header_title"] + "</div>\n"
    html += "  <table>\n"
    html += "    <thead>\n"
    html += "      <tr>\n"
    html += "        <th>Scenario</th>\n"
    html += "        <th>" + headers["vehicle_status"] + "</th>\n"
    html += "        <th>" + headers["model_decision"] + "</th>\n"
    html += "        <th>" + headers["system_outcome"] + "</th>\n"
    html += "        <th>" + headers["model_role"] + "</th>\n"
    html += "      </tr>\n"
    html += "    </thead>\n"
    html += "    <tbody>\n"
    for key in order:
        sc = scenarios[key]
        color = sc["color_theme"]
        bg = sc["bg_theme"]
        html += "      <tr>\n"
        html += "        <td><span class=\"tag\" style=\"background: " + bg + "; color: " + color + ";\">[" + sc["id"] + "] " + sc["name"] + "</span></td>\n"
        html += "        <td>" + sc["vehicle_state"] + "</td>\n"
        html += "        <td><span style=\"color: " + color + "; font-weight: 700;\">" + sc["model_verdict"] + "</span></td>\n"
        html += "        <td>" + sc["real_world_consequence"] + "</td>\n"
        html += "        <td>" + sc["role_of_model"] + "</td>\n"
        html += "      </tr>\n"
    html += "    </tbody>\n"
    html += "  </table>\n"
    html += "</div>\n"

    html += "<footer>" + html_cfg["footer_note"] + "</footer>\n"
    html += "</div>\n"

    html += "<script>\n"
    html += "const scenarios = " + scenarios_json_str + ";\n"
    html += "let currentKey = 'true_positive';\n"
    html += "let animTimer = null;\n"

    html += "function runPacketAnimation(key) {\n"
    html += "  const packet = document.getElementById('sim-packet');\n"
    html += "  const pbox = document.getElementById('sim-packet-box');\n"
    html += "  const barrier = document.getElementById('sim-barrier');\n"
    html += "  const fill = document.getElementById('sim-score-fill');\n"
    html += "  const scoreText = document.getElementById('sim-score-text');\n"
    html += "  const gtwAction = document.getElementById('sim-gateway-action');\n"
    html += "  const data = scenarios[key];\n"
    html += "  if (!data) return;\n"

    html += "  if (animTimer) clearInterval(animTimer);\n"
    html += "  packet.style.transition = 'none';\n"
    html += "  packet.style.left = '160px';\n"
    html += "  pbox.textContent = data.simulation.packet_label;\n"
    html += "  pbox.style.background = data.color_theme;\n"
    html += "  pbox.style.color = '#FFFFFF';\n"
    html += "  fill.style.width = '10%';\n"
    html += "  fill.style.background = '#10B981';\n"
    html += "  scoreText.textContent = 'Scanning packet...';\n"
    html += "  gtwAction.textContent = 'Evaluating...';\n"
    html += "  barrier.style.height = '10px';\n"
    html += "  barrier.style.background = '#334155';\n"

    html += "  setTimeout(() => {\n"
    html += "    packet.style.transition = 'left 1.2s ease-in';\n"
    html += "    packet.style.left = '480px';\n"
    html += "  }, 80);\n"

    html += "  setTimeout(() => {\n"
    html += "    scoreText.textContent = data.simulation.score_comparison;\n"
    html += "    scoreText.style.color = data.color_theme;\n"
    html += "    let pct = Math.min(100, Math.max(15, parseFloat(data.simulation.score_value) * 60));\n"
    html += "    fill.style.width = pct + '%';\n"
    html += "    fill.style.background = data.color_theme;\n"

    html += "    if (key === 'true_negative') {\n"
    html += "      barrier.style.height = '10px';\n"
    html += "      barrier.style.background = '#10B981';\n"
    html += "      gtwAction.textContent = 'ALLOWED (NORMAL)';\n"
    html += "      gtwAction.style.color = '#10B981';\n"
    html += "      packet.style.transition = 'left 1.1s ease-out';\n"
    html += "      packet.style.left = '860px';\n"
    html += "    } else if (key === 'false_positive') {\n"
    html += "      barrier.style.height = '120px';\n"
    html += "      barrier.style.background = '#F59E0B';\n"
    html += "      gtwAction.textContent = 'CAUTION: REVIEW';\n"
    html += "      gtwAction.style.color = '#F59E0B';\n"
    html += "      packet.style.transition = 'left 0.7s ease-out';\n"
    html += "      packet.style.left = '760px';\n"
    html += "    } else if (key === 'false_negative') {\n"
    html += "      barrier.style.height = '10px';\n"
    html += "      barrier.style.background = '#334155';\n"
    html += "      gtwAction.textContent = 'BYPASSED (MISSED)';\n"
    html += "      gtwAction.style.color = '#EF4444';\n"
    html += "      packet.style.transition = 'left 1.2s ease-out';\n"
    html += "      packet.style.left = '860px';\n"
    html += "    } else if (key === 'true_positive') {\n"
    html += "      barrier.style.height = '120px';\n"
    html += "      barrier.style.background = '#EF4444';\n"
    html += "      barrier.style.boxShadow = '0 0 16px #EF4444';\n"
    html += "      gtwAction.textContent = 'BLOCKED & ISOLATED';\n"
    html += "      gtwAction.style.color = '#38BDF8';\n"
    html += "      packet.style.transition = 'left 0.6s ease-out';\n"
    html += "      packet.style.left = '740px';\n"
    html += "    }\n"
    html += "  }, 1350);\n"
    html += "}\n"

    html += "function selectScenario(key) {\n"
    html += "  currentKey = key;\n"
    html += "  document.querySelectorAll('.card').forEach(c => c.classList.remove('active'));\n"
    html += "  const card = document.getElementById('card-' + key);\n"
    html += "  if (card) card.classList.add('active');\n"
    html += "  const data = scenarios[key];\n"
    html += "  if (!data) return;\n"
    html += "  document.getElementById('sim-veh-status').textContent = data.vehicle_state;\n"
    html += "  document.getElementById('sim-decision').innerHTML = '<span style=\"color:' + data.color_theme + '; font-weight:700;\">' + data.model_verdict + '</span>';\n"
    html += "  document.getElementById('sim-consequence').textContent = data.real_world_consequence;\n"
    html += "  document.getElementById('sim-role').textContent = data.role_of_model;\n"
    html += "  document.getElementById('details-box').style.borderLeftColor = data.color_theme;\n"
    html += "  document.getElementById('stage-heading').textContent = 'Simulation: ' + data.name + ' [' + data.id + ']';\n"
    html += "  runPacketAnimation(key);\n"
    html += "}\n"

    html += "function replaySimulation() {\n"
    html += "  runPacketAnimation(currentKey);\n"
    html += "}\n"

    html += "window.addEventListener('DOMContentLoaded', () => { selectScenario('true_positive'); });\n"
    html += "</script>\n"
    html += "</body>\n</html>\n"

    return html

def generate_confusion_explainer_html(counts=None, save_path=None):
    index_data = load_index_data()
    content = build_html_content(index_data, counts)

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(content)
        print("  [Visual Explainer] Saved Interactive Scenario Explainer HTML to: " + save_path)

    return content
