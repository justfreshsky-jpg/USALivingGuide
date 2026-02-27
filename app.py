import logging
from logging.handlers import RotatingFileHandler
import os
from collections import deque
import requests
import threading
import time
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, render_template_string
os.environ['HTTPX_PROXIES'] = 'null'

# ─── LOGGING ────────────────────────────────────────────────
_log_dir = os.environ.get('LOG_DIR', 'logs')
os.makedirs(_log_dir, exist_ok=True)
_log_file = os.path.join(_log_dir, 'app.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    handlers=[
        RotatingFileHandler(
            _log_file, maxBytes=5 * 1024 * 1024, backupCount=3
        ),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
GOOGLE_CLOUD_PROJECT = os.environ.get('GOOGLE_CLOUD_PROJECT') or os.environ.get('GCP_PROJECT')
VERTEX_LOCATION = os.environ.get('VERTEX_LOCATION', 'us-central1')
GEMINI_MODEL = os.environ.get('GEMINI_MODEL', 'gemini-1.5-flash')
_token_cache = {'value': '', 'expires_at': 0}

def get_access_token():
    now = time.time()
    if _token_cache['value'] and _token_cache['expires_at'] > now:
        return _token_cache['value']

    env_token = os.environ.get('GOOGLE_OAUTH_ACCESS_TOKEN')
    if env_token:
        _token_cache['value'] = env_token
        _token_cache['expires_at'] = now + 3300
        return env_token

    metadata_url = 'http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token'
    try:
        r = requests.get(metadata_url, headers={'Metadata-Flavor': 'Google'}, timeout=(1.5, 2.0))
        if not r.ok:
            return ''
        data = r.json()
        token = data.get('access_token', '')
        expires_in = max(30, int(data.get('expires_in', 300)) - 30)
        _token_cache['value'] = token
        _token_cache['expires_at'] = now + expires_in
        return token
    except Exception:
        return ''

def call_vertex(prompt):
    if not GOOGLE_CLOUD_PROJECT:
        return ''

    token = get_access_token()
    if not token:
        return ''

    endpoint = (
        f"https://{VERTEX_LOCATION}-aiplatform.googleapis.com/v1/"
        f"projects/{GOOGLE_CLOUD_PROJECT}/locations/{VERTEX_LOCATION}/"
        f"publishers/google/models/{GEMINI_MODEL}:generateContent"
    )
    payload = {
        'contents': [{'role': 'user', 'parts': [{'text': prompt}]}],
        'generationConfig': {'maxOutputTokens': 2000, 'temperature': 0.6}
    }
    headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}

    try:
        r = requests.post(endpoint, headers=headers, json=payload, timeout=(3, 30))
    except Exception:
        return ''

    if not r.ok:
        return ''

    try:
        data = r.json()
        return data['candidates'][0]['content']['parts'][0]['text']
    except Exception:
        return ''


# ─── BACKGROUND BLOG CONTENT ─────────────────────────────
_cache = {"content": "", "last": 0}

FALLBACK = """
[TAX] Rideshare tax forms are released at the end of January. 1099-K, 1099-NEC required.
Don't write "exempt" on W-4, you'll lose your refund.
[VISA] F-1 holders can travel to neighboring countries (Automatic Visa Revalidation).
J-1 visa application: Get DS-2019, pay SEVIS, schedule consulate appointment.
[PHONE] You can get a free line through the Lifeline program.
Get a US number without SSN using Google Voice.
[HEALTH] NJ Medicaid is free for low income. Free clinics available in NY.
[BANK] Chase and BofA open accounts with passport. Start credit score with a secured card.
[RIDESHARE] Uber/Lyft requires SSN + driver's license + car insurance. Expect 1099 form in January.
[HOUSING] NJ Newark/Paterson 1BR $900-1200. Try Craigslist, Zillow, Facebook Marketplace.
[WISE] International transfer limits $50k/year. Wise > Western Union.
[LICENSE] NJ has 6 Points of ID system. Even undocumented can get a license.
[FLIGHTS] International flights from NJ $400-700. Pay excess baggage 24 hours before flight for cheaper rate.
"""

def _fetch_blog():
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.google.com/"
        }
        urls = [
            "https://abdyasam.blogspot.com/",
            "https://abdyasam.blogspot.com/search?max-results=20"
        ]
        combined = ""
        for url in urls:
            r = requests.get(url, headers=headers, timeout=8)
            if not r.ok:
                logger.warning("Blog fetch returned status %s for %s", r.status_code, url)
                continue
            soup = BeautifulSoup(r.text, "html.parser")
            for tag in soup(["script", "style", "nav", "header", "footer"]):
                tag.decompose()
            posts = soup.find_all("div", class_=lambda c: c and "post" in c.lower())
            for p in posts[:15]:
                text = p.get_text(separator=" ", strip=True)
                if len(text) > 100:
                    combined += text[:800] + "\n---\n"
        if combined:
            _cache["content"] = combined[:6000]
            _cache["last"] = time.time()
    except Exception:
        logger.exception("Blog fetch failed; using fallback content")
        _cache["content"] = FALLBACK

def _bg_refresh():
    while True:
        _fetch_blog()
        time.sleep(3600)

threading.Thread(target=_bg_refresh, daemon=True).start()

def get_context():
    if not _cache["content"]:
        return FALLBACK
    return _cache["content"]

# ─── AI ─────────────────────────────────────────────
def local_fallback_reply(user):
    sample = "\n".join(
        line for line in get_context().splitlines()
        if line.strip() and line.strip() != "---"
    )
    sample = "\n".join(sample.splitlines()[:8])
    return (
        "⚠️ Vertex AI configuration is missing, so showing a quick guide summary instead of an AI response.\n\n"
        f"📌 Question: {user or 'General question'}\n"
        "✅ Full AI answers will return once you add GOOGLE_CLOUD_PROJECT and VERTEX_LOCATION to Cloud Run env variables.\n"
        "✅ Grant the Vertex AI User (roles/aiplatform.user) role to the service account.\n"
        "\nQuick Info:\n"
        f"{sample}"
    )

def llm(system, user):
    if not GOOGLE_CLOUD_PROJECT:
        return local_fallback_reply(user)

    usa_prompt = """
    🇺🇸 ONLY ANSWER ABOUT USA-RELATED TOPICS
    ✅ USA VISA / SSN / BANK / HOUSING / UBER / TAX / HEALTH
    • Add emoji to each step: ✅ 🚀 💰 📱 🏠 🪪 ✈️ 🏥 💳
    • CAPITALIZE important words
    • Short paragraphs, long lists
    • USE OUTPUT TEMPLATE:
      1) Quick Summary (3 items)
      2) Step-by-Step Checklist
      3) Common Mistakes / Risks
      4) Official Links (if available)
      5) Next Step (one clear recommendation)
    ⚠️ USA / NJ / NY ONLY!
    """

    full_system = system + "\n\n" + usa_prompt + "\n\nReference data:\n" + get_context()

    text = call_vertex(f"{full_system}\n\nUser question:\n{user}")
    if not text:
        return local_fallback_reply(user)
    text = text.replace('**', '')

    text = ''.join(c for c in text
                   if (ord(c) < 128 or
                       0x1F600 <= ord(c) <= 0x1F64F or
                       0x1F300 <= ord(c) <= 0x1F5FF))

    return text.strip()


class BadRequestError(Exception):
    """Raised when the request body is not in the expected format."""


_MAX_FIELD_LENGTH = 2000

def require_json(required_fields=None):
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        raise BadRequestError("JSON body required.")

    required_fields = required_fields or []
    missing = [field for field in required_fields if not str(data.get(field, '')).strip()]
    if missing:
        raise BadRequestError(f"Missing field(s): {', '.join(missing)}")

    for key, value in data.items():
        if isinstance(value, str) and len(value) > _MAX_FIELD_LENGTH:
            raise BadRequestError(f"Request field exceeds maximum length ({_MAX_FIELD_LENGTH} characters).")

    return data

def llm_json(system_prompt, user_prompt):
    return jsonify(result=llm(system_prompt, user_prompt))

@app.errorhandler(BadRequestError)
def handle_bad_request(error):
    return jsonify(error=str(error)), 400

@app.errorhandler(Exception)
def handle_unexpected_error(_error):
    logger.exception("Unhandled exception")
    return jsonify(error="An error occurred during processing."), 500

# ─── HTML ─────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>USA Living Guide</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:Segoe UI,Arial,sans-serif;background:#f0f4ff;color:#1e293b}
.hero{background:linear-gradient(135deg,#1e3a8a,#3b82f6);color:#fff;padding:40px 20px;text-align:center}
.hero h1{font-size:2.2em;font-weight:800;margin-bottom:10px}
.hero p{font-size:1.1em;opacity:.9;max-width:600px;margin:0 auto 16px}
.steps{display:flex;justify-content:center;flex-wrap:wrap;gap:10px;margin-top:12px}
.step{background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.3);border-radius:20px;padding:6px 16px;font-size:.9em}
.features{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:16px;max-width:1000px;margin:30px auto;padding:0 20px}
.feat{background:#fff;border-radius:14px;padding:20px;text-align:center;box-shadow:0 4px 20px rgba(0,0,0,.08);border-top:4px solid #3b82f6;transition:transform .3s}
.feat:hover{transform:translateY(-6px)}
.feat i{font-size:2em;color:#1e40af;margin-bottom:8px}
.feat h3{font-size:1em;margin-bottom:4px}
.feat p{font-size:.82em;color:#64748b}
.container{max-width:900px;margin:0 auto;padding:20px}
.tabs{display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin:24px 0}
@media(max-width:600px){.tabs{grid-template-columns:repeat(2,1fr)}}
.tabs button{background:#fff;border:2px solid #e2e8f0;padding:12px 8px;border-radius:12px;cursor:pointer;font-size:12px;font-weight:600;color:#1e293b;transition:all .2s;display:flex;flex-direction:column;align-items:center;gap:4px}
.tabs button i{font-size:1.4em;color:#3b82f6}
.tabs button.active{background:#1e3a8a;color:#fff;border-color:#3b82f6}
.tabs button.active i{color:#fff}
.tabs button:hover:not(.active){background:#f0f4ff;transform:translateY(-2px)}
.tab{display:none}
.tab.active{display:block;animation:fadeIn .4s}
@keyframes fadeIn{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
.card{background:#fff;border-radius:16px;padding:28px;box-shadow:0 4px 20px rgba(0,0,0,.08)}
.card h2{color:#1e3a8a;font-size:1.5em;margin-bottom:12px;display:flex;align-items:center;gap:10px}
.hint{background:#f0f9ff;border-left:4px solid #10b981;padding:14px 16px;border-radius:0 10px 10px 0;margin-bottom:20px;font-size:.9em;color:#0f4c75}
.form-row{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}
@media(max-width:500px){.form-row{grid-template-columns:1fr}}
.field{display:flex;flex-direction:column;gap:6px;margin-bottom:12px}
label{font-weight:600;font-size:.9em;color:#334155}
input,select,textarea{padding:12px 14px;border:2px solid #e2e8f0;border-radius:10px;font-size:15px;transition:border .2s;background:#fafbfc;width:100%}
input:focus,select:focus,textarea:focus{border-color:#3b82f6;outline:none;background:#fff}
textarea{resize:vertical;min-height:90px}
.btn{background:linear-gradient(135deg,#1e3a8a,#3b82f6);color:#fff;border:none;padding:14px;width:100%;border-radius:12px;font-size:15px;font-weight:700;cursor:pointer;margin:16px 0 8px;box-shadow:0 4px 15px rgba(30,64,175,.3);transition:all .2s}
.btn:hover{transform:translateY(-2px);box-shadow:0 8px 25px rgba(30,64,175,.4)}
.btn:disabled{opacity:.65;cursor:not-allowed;transform:none}
.output-wrap{position:relative;margin-top:8px}
.output{background:#f8fafc;border:2px solid #e2e8f0;border-radius:12px;padding:20px;min-height:100px;white-space:pre-wrap;font-size:14px;line-height:1.75}
.copy-btn{position:absolute;top:10px;right:10px;background:#10b981;color:#fff;border:none;border-radius:8px;padding:6px 14px;font-size:12px;cursor:pointer;opacity:0;transition:opacity .2s}
.output-wrap:hover .copy-btn{opacity:1}
.spinner{display:inline-block;width:16px;height:16px;border:3px solid rgba(255,255,255,.3);border-top-color:#fff;border-radius:50%;animation:spin .8s linear infinite;vertical-align:middle;margin-right:6px}
@keyframes spin{to{transform:rotate(360deg)}}
.trust-row{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0 10px}.trust-chip{background:#fff;border:1px solid #dbeafe;color:#1e3a8a;padding:8px 12px;border-radius:999px;font-size:.82em;font-weight:600}.goal-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:8px 0 18px}.goal-card{background:#fff;border:2px solid #e2e8f0;border-radius:12px;padding:12px;cursor:pointer;transition:.2s}.goal-card:hover{border-color:#3b82f6;transform:translateY(-2px)}.goal-card h4{font-size:.95em;color:#1e3a8a;margin-bottom:4px}.goal-card p{font-size:.8em;color:#64748b}.hero-cta{display:flex;justify-content:center;gap:10px;flex-wrap:wrap;margin-top:14px}.hero-cta button{background:#fff;color:#1e3a8a;border:none;padding:10px 14px;border-radius:10px;font-weight:700;cursor:pointer}.footer{text-align:center;padding:32px 20px;color:#64748b;font-size:.88em;line-height:2;background:#fff;margin-top:20px;border-radius:16px}
</style>
</head>
<body>
<div class="hero">
  <h1>🇺🇸 USA Living Guide</h1>
  <p>Create your personal roadmap for your first 30 days in the USA in 2-3 minutes.</p>
  <div class="steps">
    <span class="step">1️⃣ Pick a Topic</span>
    <span class="step">2️⃣ Enter Your Info</span>
    <span class="step">3️⃣ Get Your Checklist</span>
  </div>
  <div class="hero-cta">
    <button onclick="quickStart('ssn')">Start with SSN</button>
    <button onclick="quickStart('visa')">Visa Plan</button>
    <button onclick="quickStart('ask')">Quick Question</button>
  </div>
</div>
<div class="container">
  <div class="trust-row">
    <span class="trust-chip">🔐 No personal data stored</span>
    <span class="trust-chip">🧭 Step-by-step checklist</span>
    <span class="trust-chip">🗓 Updated: 2026</span>
  </div>
  <div class="hint" style="margin-top:8px">
    🍎 <strong>Usage tip:</strong> First pick a goal card, then generate a guide based on your situation.
  </div>
  <div class="goal-grid">
    <div class="goal-card" onclick="quickStart('ssn')"><h4>SSN Application</h4><p>Documents + office steps</p></div>
    <div class="goal-card" onclick="quickStart('bank')"><h4>Bank Account</h4><p>Options without SSN</p></div>
    <div class="goal-card" onclick="quickStart('housing')"><h4>Rent an Apartment</h4><p>Budget + lease checklist</p></div>
    <div class="goal-card" onclick="quickStart('tax')"><h4>Tax Guide</h4><p>Forms + deadline summary</p></div>
  </div>
  <div class="tabs" id="topicTabs">
    <button class="active" onclick="show('visa',this)"><i class="fas fa-passport"></i>Visa</button>
    <button onclick="show('tax',this)"><i class="fas fa-calculator"></i>Tax</button>
    <button onclick="show('rideshare',this)"><i class="fas fa-car"></i>Gig Work</button>
    <button onclick="show('housing',this)"><i class="fas fa-home"></i>Housing</button>
    <button onclick="show('health',this)"><i class="fas fa-heartbeat"></i>Health</button>
    <button onclick="show('license',this)"><i class="fas fa-id-card"></i>License</button>
    <button onclick="show('ssn',this)"><i class="fas fa-id-card-alt"></i>SSN</button>
    <button onclick="show('bank',this)"><i class="fas fa-university"></i>Bank</button>
    <button onclick="show('phone',this)"><i class="fas fa-phone"></i>Phone</button>
    <button onclick="show('car',this)"><i class="fas fa-car-side"></i>Car</button>
    <button onclick="show('transfer',this)"><i class="fas fa-exchange-alt"></i>Money Transfer</button>
    <button onclick="show('flights',this)"><i class="fas fa-plane"></i>Flights</button>
    <button onclick="show('ask',this)"><i class="fas fa-question-circle"></i>Ask</button>
    <button onclick="show('feedback',this)"><i class="fas fa-comment-dots"></i>Feedback</button>
  </div>

  <div id="visa" class="tab active"><div class="card">
    <h2><i class="fas fa-passport"></i> Visa & Green Card</h2>
    <div class="hint">🎯 <strong>For newcomers:</strong> Start with J-1, switch to H-1B once you find a job.</div>
    <div class="form-row">
      <div class="field"><label>Visa Type</label><select id="v1"><option>J-1 Student</option><option>H-1B Work</option><option>E-2 Investor</option><option>Green Card (EB)</option><option>F-1 Student</option><option>Visitor B-2</option></select></div>
      <div class="field"><label>State</label><input id="v2" placeholder="e.g. New Jersey"></div>
    </div>
    <div class="field"><label>Special Situation</label><input id="v3" placeholder="e.g. First application, extension, denied"></div>
    <button class="btn" id="vb" onclick="call('/visa',{type:g('v1'),state:g('v2'),situation:g('v3')},'vo','vb','Generate Visa Plan')">Generate Visa Plan</button>
    <div class="output-wrap"><div id="vo" class="output">Results will appear here...</div><button class="copy-btn" onclick="cp('vo')">Copy</button></div>
  </div></div>

  <div id="tax" class="tab"><div class="card">
    <h2><i class="fas fa-calculator"></i> Tax Refund & Forms</h2>
    <div class="hint">💰 <strong>Tip:</strong> File 1040NR in your first year. Add 1099 if you do rideshare.</div>
    <div class="form-row">
      <div class="field"><label>Form Type</label><select id="t1"><option>W-4 (Payroll)</option><option>1040NR (International)</option><option>1099-K (Rideshare)</option><option>W-2 (Employee)</option></select></div>
      <div class="field"><label>Annual Income ($)</label><input id="t2" type="number" placeholder="e.g. 35000"></div>
    </div>
    <div class="form-row">
      <div class="field"><label>Visa Type</label><select id="t3"><option>F-1 / J-1</option><option>H-1B</option><option>Green Card</option><option>Citizen</option></select></div>
      <div class="field"><label>State</label><input id="t4" placeholder="New Jersey"></div>
    </div>
    <button class="btn" id="tb" onclick="call('/tax',{form:g('t1'),income:g('t2'),visa:g('t3'),state:g('t4')},'to','tb','Generate Tax Checklist')">Generate Tax Checklist</button>
    <div class="output-wrap"><div id="to" class="output">Results will appear here...</div><button class="copy-btn" onclick="cp('to')">Copy</button></div>
  </div></div>

  <div id="rideshare" class="tab"><div class="card">
    <h2><i class="fas fa-car"></i> Earn with Uber / Lyft</h2>
    <div class="hint">🚗 <strong>For newcomers:</strong> License + car + SSN is enough. $800-1500/week.</div>
    <div class="form-row">
      <div class="field"><label>App</label><select id="r1"><option>Uber</option><option>Lyft</option><option>Both</option></select></div>
      <div class="field"><label>State</label><input id="r2" placeholder="New Jersey"></div>
    </div>
    <div class="field"><label>Topic</label><select id="r3"><option>How do I get started?</option><option>1099 form / taxes</option><option>How much can I earn per week?</option><option>Expense deductions</option></select></div>
    <button class="btn" id="rb" onclick="call('/rideshare',{app:g('r1'),state:g('r2'),topic:g('r3')},'ro','rb','Rideshare Guide')">Generate Plan</button>
    <div class="output-wrap"><div id="ro" class="output">Results will appear here...</div><button class="copy-btn" onclick="cp('ro')">Copy</button></div>
  </div></div>

  <div id="housing" class="tab"><div class="card">
    <h2><i class="fas fa-home"></i> Apartment Rental</h2>
    <div class="hint">🏠 <strong>Tip:</strong> NJ Newark/Paterson 1BR $900-1200. Try Craigslist and Zillow.</div>
    <div class="form-row">
      <div class="field"><label>City / Area</label><input id="e1" placeholder="e.g. Newark NJ, Jersey City"></div>
      <div class="field"><label>Budget ($/month)</label><input id="e2" type="number" placeholder="1200"></div>
    </div>
    <div class="field"><label>Special Situation</label><input id="e3" placeholder="e.g. No SSN, no credit score, have pets"></div>
    <button class="btn" id="eb" onclick="call('/housing',{city:g('e1'),budget:g('e2'),situation:g('e3')},'eo','eb','Housing Guide')">Generate Plan</button>
    <div class="output-wrap"><div id="eo" class="output">Results will appear here...</div><button class="copy-btn" onclick="cp('eo')">Copy</button></div>
  </div></div>

  <div id="health" class="tab"><div class="card">
    <h2><i class="fas fa-heartbeat"></i> Free Health Insurance</h2>
    <div class="hint">🏥 <strong>Tip:</strong> Medicaid is free in NJ for low income. Some clinics don't require documents.</div>
    <div class="form-row">
      <div class="field"><label>State</label><input id="h1" placeholder="New Jersey"></div>
      <div class="field"><label>Situation</label><select id="h2"><option>No insurance, how do I get it?</option><option>How do I apply for Medicaid?</option><option>Where are free clinics?</option><option>Can I get insurance without SSN?</option></select></div>
    </div>
    <button class="btn" id="hb" onclick="call('/health',{state:g('h1'),situation:g('h2')},'ho','hb','Health Guide')">Generate Guide</button>
    <div class="output-wrap"><div id="ho" class="output">Results will appear here...</div><button class="copy-btn" onclick="cp('ho')">Copy</button></div>
  </div></div>

  <div id="license" class="tab"><div class="card">
    <h2><i class="fas fa-id-card"></i> Driver's License (DMV)</h2>
    <div class="hint">🪪 <strong>Tip:</strong> NJ has the 6 Points of ID system. Even undocumented can get a license.</div>
    <div class="form-row">
      <div class="field"><label>State</label><input id="l1" placeholder="New Jersey"></div>
      <div class="field"><label>Situation</label><select id="l2"><option>Getting it for the first time</option><option>Converting a foreign license</option><option>No SSN / ITIN</option><option>Need Real ID</option></select></div>
    </div>
    <button class="btn" id="lb" onclick="call('/license',{state:g('l1'),situation:g('l2')},'lo','lb','License Guide')">Generate Guide</button>
    <div class="output-wrap"><div id="lo" class="output">Results will appear here...</div><button class="copy-btn" onclick="cp('lo')">Copy</button></div>
  </div></div>

<div id="ssn" class="tab">
  <div class="card">
    <h2><i class="fas fa-id-card-alt"></i> SSN Application Guide</h2>
    <div class="hint">🆔 <strong>For newcomers:</strong> F-1/J-1 students can get it with CPT/OPT. SSN is required for on-campus jobs.</div>
    <div class="form-row">
      <div class="field">
        <label>Visa Type</label>
        <select id="ss1">
          <option>F-1 Student (CPT/OPT)</option>
          <option>J-1 Student</option>
          <option>H-1B Work Visa</option>
          <option>Green Card Pending</option>
          <option>On-campus job</option>
          <option>No SSN, how do I get one?</option>
        </select>
      </div>
      <div class="field">
        <label>State</label>
        <input id="ss2" placeholder="New Jersey">
      </div>
    </div>
    <div class="field">
      <label>Situation</label>
      <input id="ss3" placeholder="e.g. CPT approved, waiting for OPT, do I need ITIN?">
    </div>
    <button class="btn" id="ssb" onclick="call('/ssn',{visa:g('ss1'),state:g('ss2'),situation:g('ss3')},'sso','ssb','Generate SSN Guide')">Generate SSN Guide</button>
    <div class="output-wrap">
      <div id="sso" class="output">Results will appear here...</div>
      <button class="copy-btn" onclick="cp('sso')">Copy</button>
    </div>
  </div>
</div>

  <div id="bank" class="tab"><div class="card">
    <h2><i class="fas fa-university"></i> Open a Bank Account</h2>
    <div class="hint">💳 <strong>Tip:</strong> Chase/BofA open accounts with passport. Start credit score with a secured card.</div>
    <div class="field"><label>Situation</label><select id="ba1"><option>Open bank account without SSN</option><option>Get a credit card</option><option>Build credit score from scratch</option><option>Best free bank?</option></select></div>
    <button class="btn" id="bb" onclick="call('/bank',{situation:g('ba1')},'bo','bb','Bank Guide')">Generate Guide</button>
    <div class="output-wrap"><div id="bo" class="output">Results will appear here...</div><button class="copy-btn" onclick="cp('bo')">Copy</button></div>
  </div></div>

  <div id="phone" class="tab"><div class="card">
    <h2><i class="fas fa-phone"></i> US Phone Number</h2>
    <div class="hint">📱 <strong>Tip:</strong> Get a free US number without SSN using Google Voice.</div>
    <div class="field"><label>Topic</label><select id="p1"><option>Free number (Google Voice)</option><option>Cheap plans (Mint, Visible, T-Mobile)</option><option>Contract plan without SSN</option><option>Cheap international calls</option></select></div>
    <button class="btn" id="pb" onclick="call('/phone',{topic:g('p1')},'po','pb','Phone Guide')">Generate Guide</button>
    <div class="output-wrap"><div id="po" class="output">Results will appear here...</div><button class="copy-btn" onclick="cp('po')">Copy</button></div>
  </div></div>

  <div id="car" class="tab"><div class="card">
    <h2><i class="fas fa-car-side"></i> Car Rental / Purchase</h2>
    <div class="hint">🚗 <strong>Tip:</strong> You can buy a car without SSN. Start with CarMax/Carvana.</div>
    <div class="form-row">
      <div class="field"><label>State</label><input id="ar1" placeholder="New Jersey"></div>
      <div class="field"><label>Topic</label><select id="ar2"><option>Buy a used car</option><option>Rent a car</option><option>Get car insurance</option><option>Can I buy without SSN?</option></select></div>
    </div>
    <button class="btn" id="arb" onclick="call('/car',{state:g('ar1'),topic:g('ar2')},'aro','arb','Car Guide')">Generate Guide</button>
    <div class="output-wrap"><div id="aro" class="output">Results will appear here...</div><button class="copy-btn" onclick="cp('aro')">Copy</button></div>
  </div></div>

  <div id="transfer" class="tab"><div class="card">
    <h2><i class="fas fa-exchange-alt"></i> Money Transfer (Wise / Zelle)</h2>
    <div class="hint">💸 <strong>Tip:</strong> Send money internationally with the lowest fees using Wise.</div>
    <div class="field"><label>Topic</label><select id="w1"><option>Send money abroad with Wise</option><option>Wise limits and fees</option><option>How to use Zelle?</option><option>Venmo / CashApp guide</option></select></div>
    <button class="btn" id="wb" onclick="call('/transfer',{topic:g('w1')},'wo','wb','Money Transfer Guide')">Generate Guide</button>
    <div class="output-wrap"><div id="wo" class="output">Results will appear here...</div><button class="copy-btn" onclick="cp('wo')">Copy</button></div>
  </div></div>

  <div id="flights" class="tab"><div class="card">
    <h2><i class="fas fa-plane"></i> Flights & Baggage</h2>
    <div class="hint">✈️ <strong>Tip:</strong> International flights from NJ $400-700. Pay excess baggage 24 hours before for cheaper rates.</div>
    <div class="form-row">
      <div class="field"><label>Airline</label><select id="u1"><option>Turkish Airlines</option><option>American Airlines</option><option>United</option><option>Delta</option></select></div>
      <div class="field"><label>Topic</label><select id="u2"><option>Baggage fees and rules</option><option>How to find cheapest tickets?</option><option>Check-in guide</option><option>Refund / cancellation policy</option></select></div>
    </div>
    <button class="btn" id="ub" onclick="call('/flights',{airline:g('u1'),topic:g('u2')},'uo','ub','Flight Guide')">Generate Guide</button>
    <div class="output-wrap"><div id="uo" class="output">Results will appear here...</div><button class="copy-btn" onclick="cp('uo')">Copy</button></div>
  </div></div>

  <div id="ask" class="tab"><div class="card">
    <h2><i class="fas fa-question-circle"></i> Ask Any Question</h2>
    <div class="hint">🤖 Ask anything about life in the USA. You'll get a detailed answer.</div>
    <div class="field"><label>What's your question?</label><textarea id="q1" rows="4" placeholder="e.g. Can I find a job without SSN? What should I do in my first month?"></textarea></div>
    <button class="btn" id="qb" onclick="call('/ask',{question:g('q1')},'qo','qb','Answer')">Answer</button>
    <div class="output-wrap"><div id="qo" class="output">Answer will appear here...</div><button class="copy-btn" onclick="cp('qo')">Copy</button></div>
  </div></div>

  <div id="feedback" class="tab"><div class="card">
    <h2><i class="fas fa-comment-dots"></i> Site Feedback</h2>
    <div class="hint">💬 Share your experience: what worked, what's missing, what should we improve?</div>
    <div class="field"><label>Your Message</label><textarea id="fb1" rows="4" placeholder="E.g. Tab transitions could be faster, output should be PDF, more official links needed..."></textarea></div>
    <div class="field"><label>Optional Email</label><input id="fb2" placeholder="name@example.com"></div>
    <button class="btn" id="fbb" onclick="sendFeedback()">Submit Feedback</button>
    <div class="output-wrap"><div id="fbo" class="output">Feedback status message will appear here...</div></div>
  </div></div>

</div>
<div class="footer">
  No personal data is stored<br>
  <span style="font-size:.8em;color:#94a3b8">
    ⚠️ This tool is for informational purposes only. AI may make mistakes.
    For legal, financial, or medical matters, always consult a professional.
    No liability is accepted for decisions made based on AI output.
  </span>
</div>
<script>
function g(id){return document.getElementById(id).value;}
const lastAnswers = {};
function quickStart(tab){
  const target=document.getElementById(tab);
  if(!target) return;
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tabs button').forEach(b=>b.classList.remove('active'));
  target.classList.add('active');
  const match=[...document.querySelectorAll('.tabs button')].find(b=>{const h=b.getAttribute('onclick'); return h && h.indexOf("'"+tab+"'")>-1;});
  if(match) match.classList.add('active');
  const firstInput=target.querySelector('input,select,textarea');
  if(firstInput) firstInput.focus({preventScroll:true});
  target.scrollIntoView({behavior:'smooth',block:'start'});
}
function show(tab,btn){
  const target=document.getElementById(tab);
  if(!target) return;
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tabs button').forEach(b=>b.classList.remove('active'));
  target.classList.add('active');
  if(btn) btn.classList.add('active');
}
function cp(id){
  navigator.clipboard.writeText(document.getElementById(id).innerText).then(()=>{
    const btn=document.querySelector('#'+id).parentNode.querySelector('.copy-btn');
    btn.textContent='Copied!';
    setTimeout(()=>btn.textContent='Copy',2000);
  });
}
async function call(endpoint,data,outId,btnId,label){
  const out=document.getElementById(outId);
  const btn=document.getElementById(btnId);
  btn.disabled=true;
  btn.innerHTML='<span class="spinner"></span>Step 1/3: Preparing info';
  out.textContent='Step 2/3: Generating response with Vertex AI...';
  try{
    const r=await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
    const j=await r.json().catch(()=>({}));
    if(!r.ok){
      out.textContent='Error: '+(j.error || 'Request could not be processed.');
      return;
    }
    out.textContent='Step 3/3: Result ready ✅\n\n'+(j.result || 'Could not generate result.');
    lastAnswers[outId]=j.result || '';
    ensureFollowupBox(outId);
  }catch(e){
    out.textContent='Connection error: '+e.message;
  }finally{
    btn.disabled=false;
    btn.textContent=label;
  }
}

function ensureFollowupBox(outId){
  const out=document.getElementById(outId);
  const wrap=out && out.closest('.output-wrap');
  if(!wrap) return;
  const ns=wrap.nextElementSibling;
  if(ns && ns.classList && ns.classList.contains('followup-wrap')) return;

  const box=document.createElement('div');
  box.className='followup-wrap';
  box.style.marginTop='10px';

  const field=document.createElement('div');
  field.className='field';
  const label=document.createElement('label');
  label.textContent='Dig deeper (follow-up question)';
  const textarea=document.createElement('textarea');
  textarea.id='fu-'+outId;
  textarea.rows=2;
  textarea.placeholder='Explain this part of the answer in more detail...';
  field.appendChild(label);
  field.appendChild(textarea);

  const btn=document.createElement('button');
  btn.id='fub-'+outId;
  btn.className='btn';
  btn.style.margin='6px 0 0';
  btn.textContent='Ask Follow-up';
  btn.addEventListener('click', function(){
    followup(outId, textarea.id, btn.id);
  });

  box.appendChild(field);
  box.appendChild(btn);
  wrap.parentNode.insertBefore(box, wrap.nextSibling);
}

function followup(outId,inputId,btnId){
  const el=document.getElementById(inputId);
  const q=((el && el.value) || '').trim();
  if(!q) return;
  const previous=lastAnswers[outId] || '';
  const prompt='Previous answer:\n'+previous+'\n\nFollow-up question:\n'+q+'\n\nPlease explain more clearly, step by step, with examples.';
  call('/ask',{question:prompt},outId,btnId,'Ask Follow-up');
}

async function sendFeedback(){
  const out=document.getElementById('fbo');
  const btn=document.getElementById('fbb');
  btn.disabled=true;
  out.textContent='Sending...';
  try{
    const r=await fetch('/feedback',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:g('fb1'),contact:g('fb2')})});
    const j=await r.json().catch(()=>({}));
    out.textContent=r.ok ? (j.result || 'Thank you!') : ('Error: '+(j.error || 'Could not send'));
  }catch(e){
    out.textContent='Connection error: '+e.message;
  }finally{
    btn.disabled=false;
  }
}

</script>
</body>
</html>"""# ─── ROUTES ──────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/healthz')
def healthz():
    return jsonify(
        status='ok',
        ai_provider='vertex_ai_gemini',
        vertex_configured=bool(GOOGLE_CLOUD_PROJECT and get_access_token()),
        project=GOOGLE_CLOUD_PROJECT,
        location=VERTEX_LOCATION,
        model=GEMINI_MODEL
    )

@app.route('/visa', methods=['POST'])
def do_visa():
    d = require_json(["type"])
    return llm_json(
        "You are a US immigration expert. Provide practical English guidance.",
        f"{d['type']} visa. State: {d.get('state','')}. Situation: {d.get('situation','')}. Documents, forms, fees, common mistakes, links."
    )

@app.route('/tax', methods=['POST'])
def do_tax():
    d = require_json(["form"])
    return llm_json(
        "You are a US tax expert. Explain clearly in English.",
        f"Form: {d['form']}. Income: ${d.get('income',0)}. Visa: {d.get('visa','')}. State: {d.get('state','')}. Filing guide, refund estimate, deadlines."
    )

@app.route('/rideshare', methods=['POST'])
def do_rideshare():
    d = require_json(["app"])
    return llm_json(
        "You are a rideshare and gig economy expert. Write in English.",
        f"{d['app']} - {d.get('state','')}. Topic: {d.get('topic','')}. Documents, earnings, tax, tips."
    )

@app.route('/housing', methods=['POST'])
def do_housing():
    d = require_json()
    return llm_json(
        "You are a US real estate expert. Write in English.",
        f"{d.get('city','')} ${d.get('budget','')} budget. Situation: {d.get('situation','')}. Websites, documents, negotiation tips."
    )

@app.route('/health', methods=['POST'])
def do_health():
    d = require_json()
    return llm_json(
        "You are a US healthcare system expert. Write practical English guidance.",
        f"{d.get('state','')} - {d.get('situation','')}. Addresses, documents, Medicaid, free clinics."
    )

@app.route('/license', methods=['POST'])
def do_license():
    d = require_json()
    return llm_json(
        "You are a US DMV expert. Explain in English.",
        f"{d.get('state','')} driver's license: {d.get('situation','')}. 6 Points documents, exam, appointment, fees."
    )

@app.route('/ssn', methods=['POST'])
def do_ssn():
    d = require_json(["visa"])
    return llm_json(
        "You are a US SSN expert. Provide practical English guidance focused on NJ.",
        f"Visa: {d['visa']}. State: {d.get('state','NJ')}. Situation: {d.get('situation','')}. \
        Required documents for SSN, application steps, NJ SSA office addresses, \
        CPT/OPT requirements for F-1/J-1, ITIN alternative, common mistakes."
    )

@app.route('/bank', methods=['POST'])
def do_bank():
    d = require_json()
    return llm_json(
        "You are a US banking expert. Write in English.",
        f"Topic: {d.get('situation','')}. Which bank, documents, credit score, secured card."
    )

@app.route('/phone', methods=['POST'])
def do_phone():
    d = require_json()
    return llm_json(
        "You are a US telecom expert. Provide an English guide.",
        f"Topic: {d.get('topic','')}. Step-by-step setup, prices, alternatives."
    )

@app.route('/car', methods=['POST'])
def do_car():
    d = require_json()
    return llm_json(
        "You are a US automotive expert. Write in English.",
        f"{d.get('state','')} - {d.get('topic','')}. Documents, insurance, pricing, CarMax/Carvana."
    )

@app.route('/transfer', methods=['POST'])
def do_transfer():
    d = require_json()
    return llm_json(
        "You are a money transfer expert. Explain in English.",
        f"Topic: {d.get('topic','')}. Steps, fees, limits, alternatives."
    )

@app.route('/flights', methods=['POST'])
def do_flights():
    d = require_json()
    return llm_json(
        "You are an aviation expert. Provide a practical English guide.",
        f"{d.get('airline','')} - {d.get('topic','')}. Detailed info, fees, tips."
    )

@app.route('/ask', methods=['POST'])
def do_ask():
    d = require_json()
    return llm_json(
        "You are a practical guide expert for people living in the USA. Give clear, step-by-step, safe answers in English.",
        d.get('question', '')
    )

_feedback_store = deque(maxlen=500)

@app.route('/feedback', methods=['POST'])
def do_feedback():
    d = require_json(['message'])
    _feedback_store.append({
        'message': d.get('message', '').strip(),
        'contact': d.get('contact', '').strip(),
        'ts': int(time.time())
    })
    return jsonify(result='Thank you! Your feedback has been received and added to the improvement list.', total_feedback=len(_feedback_store))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))