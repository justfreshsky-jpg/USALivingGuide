import hashlib
import logging
from logging.handlers import RotatingFileHandler
import os
import re
import unicodedata
from collections import deque, OrderedDict
import requests
import threading
import time
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify, make_response

# ─── LOGGING ────────────────────────────────────────────────
_log_handlers = [logging.StreamHandler()]
_log_dir = os.environ.get('LOG_DIR', '')
if _log_dir:
    os.makedirs(_log_dir, exist_ok=True)
    _log_file = os.path.join(_log_dir, 'app.log')
    _log_handlers.append(RotatingFileHandler(_log_file, maxBytes=5 * 1024 * 1024, backupCount=3))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    handlers=_log_handlers,
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# ─── RESPONSE CACHE ──────────────────────────────────────
_RESPONSE_CACHE_TTL = 3600       # 1 hour
_RESPONSE_CACHE_MAX = 500
_response_cache: 'OrderedDict[str, tuple]' = OrderedDict()
_response_cache_lock = threading.Lock()

def _cache_key(system: str, user: str) -> str:
    import json
    return hashlib.sha256(json.dumps([system, user], ensure_ascii=False).encode()).hexdigest()

def _response_cache_get(key: str):
    with _response_cache_lock:
        entry = _response_cache.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if time.time() > expires_at:
            del _response_cache[key]
            return None
        # Move to end (LRU)
        _response_cache.move_to_end(key)
        return value

def _response_cache_set(key: str, value: str):
    with _response_cache_lock:
        if key in _response_cache:
            _response_cache.move_to_end(key)
        _response_cache[key] = (value, time.time() + _RESPONSE_CACHE_TTL)
        while len(_response_cache) > _RESPONSE_CACHE_MAX:
            _response_cache.popitem(last=False)

# ─── RATE LIMITING ────────────────────────────────────────
_RATE_LIMIT_MAX = 20
_RATE_LIMIT_WINDOW = 60          # seconds
_rate_limit_store: 'dict[str, deque]' = {}
_rate_limit_lock = threading.Lock()

def _is_rate_limited(ip: str) -> bool:
    now = time.time()
    cutoff = now - _RATE_LIMIT_WINDOW
    with _rate_limit_lock:
        q = _rate_limit_store.setdefault(ip, deque())
        # Drop timestamps outside the window
        while q and q[0] < cutoff:
            q.popleft()
        # Clean up empty deques for inactive IPs
        if len(q) == 0:
            del _rate_limit_store[ip]
        if len(q) >= _RATE_LIMIT_MAX:
            return True
        q.append(now)
        return False

@app.before_request
def check_rate_limit():
    if request.method == 'POST':
        ip = request.remote_addr
        if not ip:
            return  # No IP available; skip rate limiting
        if _is_rate_limited(ip):
            return jsonify(error='Too many requests. Please wait before trying again.'), 429

# ─── SECURITY HEADERS ─────────────────────────────────────
@app.after_request
def add_security_headers(response):
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "style-src 'self' 'unsafe-inline' https://cdnjs.cloudflare.com; "
        "font-src 'self' https://cdnjs.cloudflare.com; "
        "script-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "connect-src 'self';"
    )
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response

# ─── LLM PROVIDERS ───────────────────────────────────────

def _llm_via_groq(system: str, user: str) -> str:
    key = os.environ.get('GROQ_KEY', '')
    if not key:
        return ''
    payload = {
        'model': 'llama3-8b-8192',
        'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
        ],
        'max_tokens': 2000,
        'temperature': 0.6,
    }
    headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
    try:
        r = requests.post('https://api.groq.com/openai/v1/chat/completions',
                          headers=headers, json=payload, timeout=(5, 30))
        if not r.ok:
            logger.warning("Groq returned status %s", r.status_code)
            return ''
        return r.json()['choices'][0]['message']['content']
    except Exception:
        logger.warning("Groq call failed", exc_info=True)
        return ''


def _llm_via_cerebras(system: str, user: str) -> str:
    key = os.environ.get('CEREBRAS_KEY', '')
    if not key:
        return ''
    payload = {
        'model': 'llama3.1-8b',
        'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
        ],
        'max_tokens': 2000,
        'temperature': 0.6,
    }
    headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
    try:
        r = requests.post('https://api.cerebras.ai/v1/chat/completions',
                          headers=headers, json=payload, timeout=(5, 30))
        if not r.ok:
            logger.warning("Cerebras returned status %s", r.status_code)
            return ''
        return r.json()['choices'][0]['message']['content']
    except Exception:
        logger.warning("Cerebras call failed", exc_info=True)
        return ''


def _llm_via_gemini(system: str, user: str) -> str:
    key = os.environ.get('GEMINI_KEY', '')
    if not key:
        return ''
    payload = {
        'systemInstruction': {'parts': [{'text': system}]},
        'contents': [{'role': 'user', 'parts': [{'text': user}]}],
        'generationConfig': {'maxOutputTokens': 2000, 'temperature': 0.6},
    }
    url = (
        'https://generativelanguage.googleapis.com/v1beta/models/'
        f'gemini-1.5-flash:generateContent?key={key}'
    )
    try:
        r = requests.post(url, json=payload, timeout=(5, 30))
        if not r.ok:
            logger.warning("Gemini returned status %s", r.status_code)
            return ''
        data = r.json()
        return data['candidates'][0]['content']['parts'][0]['text']
    except Exception:
        logger.warning("Gemini call failed", exc_info=True)
        return ''


def _llm_via_cohere(system: str, user: str) -> str:
    key = os.environ.get('COHERE_KEY', '')
    if not key:
        return ''
    payload = {
        'model': 'command-r-plus',
        'preamble': system,
        'message': user,
        'max_tokens': 2000,
        'temperature': 0.6,
    }
    headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
    try:
        r = requests.post('https://api.cohere.ai/v1/chat',
                          headers=headers, json=payload, timeout=(5, 30))
        if not r.ok:
            logger.warning("Cohere returned status %s", r.status_code)
            return ''
        return r.json()['text']
    except Exception:
        logger.warning("Cohere call failed", exc_info=True)
        return ''


def _llm_via_mistral(system: str, user: str) -> str:
    key = os.environ.get('MISTRAL_KEY', '')
    if not key:
        return ''
    payload = {
        'model': 'mistral-small-latest',
        'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
        ],
        'max_tokens': 2000,
        'temperature': 0.6,
    }
    headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
    try:
        r = requests.post('https://api.mistral.ai/v1/chat/completions',
                          headers=headers, json=payload, timeout=(5, 30))
        if not r.ok:
            logger.warning("Mistral returned status %s", r.status_code)
            return ''
        return r.json()['choices'][0]['message']['content']
    except Exception:
        logger.warning("Mistral call failed", exc_info=True)
        return ''


def _llm_via_openrouter(system: str, user: str) -> str:
    key = os.environ.get('OPENROUTER_KEY', '')
    if not key:
        return ''
    payload = {
        'model': 'mistralai/mistral-7b-instruct',
        'messages': [
            {'role': 'system', 'content': system},
            {'role': 'user', 'content': user},
        ],
        'max_tokens': 2000,
        'temperature': 0.6,
    }
    headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
    try:
        r = requests.post('https://openrouter.ai/api/v1/chat/completions',
                          headers=headers, json=payload, timeout=(5, 30))
        if not r.ok:
            logger.warning("OpenRouter returned status %s", r.status_code)
            return ''
        return r.json()['choices'][0]['message']['content']
    except Exception:
        logger.warning("OpenRouter call failed", exc_info=True)
        return ''


def _llm_via_huggingface(system: str, user: str) -> str:
    key = os.environ.get('HF_KEY', '')
    if not key:
        return ''
    prompt = f"[INST] <<SYS>>\n{system}\n<</SYS>>\n\n{user} [/INST]"
    payload = {
        'inputs': prompt,
        'parameters': {'max_new_tokens': 2000, 'temperature': 0.6, 'return_full_text': False},
    }
    headers = {'Authorization': f'Bearer {key}'}
    try:
        r = requests.post(
            'https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf',
            headers=headers, json=payload, timeout=(5, 30),
        )
        if not r.ok:
            logger.warning("HuggingFace returned status %s", r.status_code)
            return ''
        data = r.json()
        if isinstance(data, list) and data:
            return data[0].get('generated_text', '')
        return ''
    except Exception:
        logger.warning("HuggingFace call failed", exc_info=True)
        return ''


_PROVIDERS = [
    _llm_via_groq,
    _llm_via_cerebras,
    _llm_via_gemini,
    _llm_via_cohere,
    _llm_via_mistral,
    _llm_via_openrouter,
    _llm_via_huggingface,
]


# ─── BACKGROUND BLOG CONTENT ─────────────────────────────
_blog_cache = {"content": "", "last": 0}
_blog_cache_lock = threading.Lock()
_refresh_thread_started = False
_refresh_thread_lock = threading.Lock()

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
            with _blog_cache_lock:
                _blog_cache["content"] = combined[:6000]
                _blog_cache["last"] = time.time()
    except requests.RequestException as exc:
        logger.warning("Blog fetch failed (%s); using fallback content", exc.__class__.__name__)
        with _blog_cache_lock:
            _blog_cache["content"] = FALLBACK
    except Exception:
        logger.exception("Blog fetch failed unexpectedly; using fallback content")
        with _blog_cache_lock:
            _blog_cache["content"] = FALLBACK

def _bg_refresh():
    while True:
        try:
            _fetch_blog()
        except Exception:
            logger.exception("Unexpected error in background blog refresh")
        time.sleep(3600)

def ensure_bg_refresh_started():
    global _refresh_thread_started
    if _refresh_thread_started:
        return
    with _refresh_thread_lock:
        if _refresh_thread_started:
            return
        threading.Thread(target=_bg_refresh, daemon=True).start()
        _refresh_thread_started = True

def get_context():
    with _blog_cache_lock:
        content = _blog_cache["content"]
    return content if content else FALLBACK

# ─── TOPIC GUIDE DATA ─────────────────────────────────────
TOPIC_GUIDES = [
    (
        ['uber', 'lyft', 'rideshare', 'gig'],
        [
            'Start with driver signup: SSN, valid driver license, vehicle registration, and insurance.',
            'Track every trip expense (gas, maintenance, phone, tolls) for tax deductions.',
            'Expect year-end forms (1099-K/1099-NEC) and set aside tax money weekly.'
        ],
        [
            'Create Uber/Lyft account and upload documents for approval.',
            'Complete required background check and vehicle inspection.',
            'Drive during peak hours and keep a weekly earnings + expense log.',
        ],
    ),
    (
        ['visa', 'f-1', 'j-1', 'h-1b', 'green card', 'opt', 'cpt'],
        [
            'Keep passport, I-94, and visa documents valid and stored in one folder.',
            'Follow status-specific rules (F-1/J-1 work limits, H-1B employer restrictions).',
            'Use official USCIS/State Department pages for forms and deadlines.'
        ],
        [
            'Confirm your current status and expiration dates.',
            'Prepare required forms and supporting documents before filing.',
            'Book appointments early and keep receipt numbers for tracking.',
        ],
    ),
    (
        ['rent', 'housing', 'apartment', 'lease', 'landlord'],
        [
            'Search on trusted listing sites and compare commute + safety + total monthly cost.',
            'Prepare ID, proof of income, and references before applying.',
            'Read lease terms carefully (deposit, maintenance, renewal, penalties).'
        ],
        [
            'Set your budget including utilities and internet.',
            'Tour multiple units and document apartment condition before move-in.',
            'Sign lease only after confirming all fees and move-out terms.',
        ],
    ),
    (
        ['tax', '1099', 'w-2', 'w-4', 'refund', 'irs'],
        [
            'Collect all forms first (W-2/1099/1098) before filing.',
            'Use the correct filing status and include state return if required.',
            'File before deadlines and keep PDF copies of all submissions.'
        ],
        [
            'Create IRS account and gather identity/tax documents.',
            'Prepare federal return, then state return, then submit.',
            'Track refund status and respond quickly to IRS letters.',
        ],
    ),
]

# ─── AI ─────────────────────────────────────────────
def local_fallback_reply(user=None):
    return (
        "✅ USA Living Guide\n\n"
        "Our AI assistant is temporarily unavailable. Please try again in a few moments.\n\n"
        "In the meantime, visit these official resources:\n"
        "- uscis.gov — Immigration and visa information\n"
        "- irs.gov — Tax filing and forms\n"
        "- ssa.gov — Social Security Administration\n"
        "- healthcare.gov — Health insurance marketplace\n"
        "- dmv.org — Driver's license and DMV info\n\n"
        "If the issue persists, please use the Feedback tab to let us know."
    )

def _clean_ai_text(text):
    """Remove markdown formatting and filter to safe Unicode characters."""
    text = text.replace('**', '')
    # Strip leading # header markers from lines
    text = re.sub(r'(?m)^#+\s*', '', text)
    # Normalize excessive blank lines (3+ newlines → 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    cleaned = []
    for c in text:
        if ord(c) < 128 or unicodedata.category(c).startswith(('L', 'M', 'N', 'P', 'S', 'Z')):
            cleaned.append(c)
    return ''.join(cleaned).strip()

def llm(system, user):
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

    key = _cache_key(system, user)
    cached = _response_cache_get(key)
    if cached is not None:
        return cached

    text = ''
    for provider in _PROVIDERS:
        text = provider(full_system, user)
        if text:
            break

    if not text:
        return local_fallback_reply(user)

    result = _clean_ai_text(text)
    _response_cache_set(key, result)
    return result


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
    logger.warning("Bad request: %s", error)
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
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ctext y='.9em' font-size='90'%3E%F0%9F%87%BA%F0%9F%87%B8%3C/text%3E%3C/svg%3E">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css" integrity="sha512-iecdLmaskl7CVkqkXNQ/ZH/XLlvWZOJyj7Yy7tcenmpD1ypASozpmT/E0iPtmFIB46ZmdtAc9eNBvH0H/ZpiBw==" crossorigin="anonymous" referrerpolicy="no-referrer">
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:Segoe UI,Arial,sans-serif;background:#f0f4ff;color:#1e293b}
.hero{background:linear-gradient(135deg,#1e3a8a,#3b82f6);color:#fff;padding:40px 20px;text-align:center;isolation:isolate}
.hero h1{font-size:2.2em;font-weight:800;margin-bottom:10px}
.hero p{font-size:1.1em;opacity:.9;max-width:600px;margin:0 auto 16px}
.steps{display:flex;justify-content:center;flex-wrap:wrap;gap:10px;margin-top:12px;pointer-events:none}
.step{background:rgba(255,255,255,.15);border:1px solid rgba(255,255,255,.3);border-radius:20px;padding:6px 16px;font-size:.9em;pointer-events:none}
.features{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:16px;max-width:1000px;margin:30px auto;padding:0 20px}
.feat{background:#fff;border-radius:14px;padding:20px;text-align:center;box-shadow:0 4px 20px rgba(0,0,0,.08);border-top:4px solid #3b82f6;transition:transform .3s}
.feat:hover{transform:translateY(-6px)}
.feat i{font-size:2em;color:#1e40af;margin-bottom:8px}
.feat h3{font-size:1em;margin-bottom:4px}
.feat p{font-size:.82em;color:#64748b}
.container{max-width:900px;margin:0 auto;padding:20px}
.tabs{display:grid;grid-template-columns:repeat(auto-fit,minmax(80px,1fr));gap:8px;margin:24px 0}
.tabs button{background:#fff;border:2px solid #e2e8f0;padding:12px 8px;border-radius:12px;cursor:pointer;font-size:11px;font-weight:600;color:#1e293b;transition:all .2s;display:flex;flex-direction:column;align-items:center;gap:4px;position:relative;z-index:1}
.tabs button i{font-size:1.4em;color:#3b82f6}
.tabs button.active{background:#1e3a8a;color:#fff;border-color:#3b82f6}
.tabs button.active i{color:#fff}
.tabs button:hover:not(.active){background:#f0f4ff;transform:translateY(-2px);z-index:2}
.tab{display:none}
.tab.active{display:block;animation:fadeIn .4s;position:relative;z-index:1}
@keyframes fadeIn{from{opacity:0;transform:translateY(12px)}to{opacity:1;transform:translateY(0)}}
.card{background:#fff;border-radius:16px;padding:28px;box-shadow:0 4px 20px rgba(0,0,0,.08)}
.card h2{color:#1e3a8a;font-size:1.5em;margin-bottom:12px;display:flex;align-items:center;gap:10px}
.hint{background:#f0f9ff;border-left:4px solid #3b82f6;padding:14px 16px;border-radius:0 10px 10px 0;margin-bottom:20px;font-size:.9em;color:#0f4c75}
.form-row{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px}
@media(max-width:500px){.form-row{grid-template-columns:1fr}}
.field{display:flex;flex-direction:column;gap:6px;margin-bottom:12px}
label{font-weight:600;font-size:.9em;color:#334155}
input,select,textarea{padding:12px 14px;border:2px solid #e2e8f0;border-radius:10px;font-size:15px;transition:border .2s;background:#fafbfc;width:100%}
input:focus,select:focus,textarea:focus{border-color:#3b82f6;outline:none;background:#fff}
textarea{resize:vertical;min-height:90px}
.btn{background:linear-gradient(135deg,#1e3a8a,#3b82f6);color:#fff;border:none;padding:14px;width:100%;border-radius:12px;font-size:15px;font-weight:700;cursor:pointer;margin:16px 0 8px;box-shadow:0 4px 15px rgba(30,64,175,.3);transition:all .2s;position:relative;z-index:1}
.btn:hover{transform:translateY(-2px);box-shadow:0 8px 25px rgba(30,64,175,.4)}
.btn:disabled{opacity:.65;cursor:not-allowed;transform:none}
.output-wrap{position:relative;margin-top:8px}
.output{background:#f8fafc;border:2px solid #e2e8f0;border-radius:12px;padding:20px;min-height:100px;white-space:pre-wrap;font-size:14px;line-height:1.75}
.output.error{border-color:#ef4444;background:#fff5f5;color:#b91c1c}
.output.loading{animation:pulse 1.5s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.5}}
.copy-btn{position:absolute;top:10px;right:10px;background:#10b981;color:#fff;border:none;border-radius:8px;padding:6px 14px;font-size:12px;cursor:pointer;opacity:0;transition:opacity .2s}
.output-wrap:hover .copy-btn{opacity:1}
@media(hover:none){.copy-btn{opacity:1}}
.spinner{display:inline-block;width:16px;height:16px;border:3px solid rgba(255,255,255,.3);border-top-color:#fff;border-radius:50%;animation:spin .8s linear infinite;vertical-align:middle;margin-right:6px}
@keyframes spin{to{transform:rotate(360deg)}}
.trust-row{display:flex;gap:10px;flex-wrap:wrap;margin:16px 0 10px}.trust-chip{background:#fff;border:1px solid #dbeafe;color:#1e3a8a;padding:8px 12px;border-radius:999px;font-size:.82em;font-weight:600}.goal-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px;margin:8px 0 18px}.goal-card{background:#fff;border:2px solid #e2e8f0;border-radius:12px;padding:12px;cursor:pointer;transition:.2s;position:relative;z-index:1}.goal-card:hover{border-color:#3b82f6;transform:translateY(-2px);z-index:2}.goal-card h4{font-size:.95em;color:#1e3a8a;margin-bottom:4px}.goal-card p{font-size:.8em;color:#64748b}.hero-cta{display:flex;justify-content:center;gap:10px;flex-wrap:wrap;margin-top:14px;position:relative;z-index:10}.hero-cta button{background:#fff;color:#1e3a8a;border:none;padding:10px 14px;border-radius:10px;font-weight:700;cursor:pointer;position:relative;z-index:1}.footer{text-align:center;padding:32px 20px;color:#64748b;font-size:.88em;line-height:2;background:#fff;margin-top:20px;border-radius:16px}.char-count{font-size:.8em;color:#94a3b8;text-align:right;margin-top:2px}
.hero-subtitle{font-size:1em;color:#e0e7ff;margin:6px 0 0;opacity:.9;max-width:520px;text-align:center}
</style>
</head>
<body>
<div class="hero">
  <h1 style="cursor:pointer" title="Click to reload">🇺🇸 USA Living Guide</h1>
  <p class="hero-subtitle">Your AI-powered guide to navigating life in the USA — covering visas, taxes, banking, housing, healthcare, and more. Pick a topic below to get a personalized, step-by-step answer.</p>
  <div class="steps">
    <span class="step">1️⃣ Pick a Topic</span>
    <span class="step">2️⃣ Enter Your Info</span>
    <span class="step">3️⃣ Get Your Checklist</span>
  </div>
</div>
<div class="container">

  <div class="tabs" id="topicTabs" role="tablist">
    <button type="button" class="active" data-tab="visa" role="tab" aria-selected="true" aria-controls="visa"><i class="fas fa-passport"></i>Visa</button>
    <button type="button" data-tab="tax" role="tab" aria-selected="false" aria-controls="tax"><i class="fas fa-calculator"></i>Tax</button>
    <button type="button" data-tab="rideshare" role="tab" aria-selected="false" aria-controls="rideshare"><i class="fas fa-car"></i>Gig Work</button>
    <button type="button" data-tab="housing" role="tab" aria-selected="false" aria-controls="housing"><i class="fas fa-home"></i>Housing</button>
    <button type="button" data-tab="health" role="tab" aria-selected="false" aria-controls="health"><i class="fas fa-heartbeat"></i>Health</button>
    <button type="button" data-tab="license" role="tab" aria-selected="false" aria-controls="license"><i class="fas fa-id-card"></i>License</button>
    <button type="button" data-tab="ssn" role="tab" aria-selected="false" aria-controls="ssn"><i class="fas fa-id-card-alt"></i>SSN</button>
    <button type="button" data-tab="bank" role="tab" aria-selected="false" aria-controls="bank"><i class="fas fa-university"></i>Bank</button>
    <button type="button" data-tab="phone" role="tab" aria-selected="false" aria-controls="phone"><i class="fas fa-phone"></i>Phone</button>
    <button type="button" data-tab="car" role="tab" aria-selected="false" aria-controls="car"><i class="fas fa-car-side"></i>Car</button>
    <button type="button" data-tab="transfer" role="tab" aria-selected="false" aria-controls="transfer"><i class="fas fa-exchange-alt"></i>Money Transfer</button>
    <button type="button" data-tab="flights" role="tab" aria-selected="false" aria-controls="flights"><i class="fas fa-plane"></i>Flights</button>
    <button type="button" data-tab="ask" role="tab" aria-selected="false" aria-controls="ask"><i class="fas fa-question-circle"></i>Ask</button>
    <button type="button" data-tab="feedback" role="tab" aria-selected="false" aria-controls="feedback"><i class="fas fa-comment-dots"></i>Feedback</button>
  </div>

  <div id="visa" class="tab active"><div class="card">
    <h2><i class="fas fa-passport"></i> Visa & Green Card</h2>
    <div class="hint">🎯 <strong>For newcomers:</strong> Start with J-1, switch to H-1B once you find a job.</div>
    <div class="form-row">
      <div class="field"><label for="v1">Visa Type</label><select id="v1"><option>J-1 Student</option><option>H-1B Work</option><option>E-2 Investor</option><option>Green Card (EB)</option><option>F-1 Student</option><option>Visitor B-2</option></select></div>
      <div class="field"><label for="v2">State</label><input id="v2" placeholder="e.g. New Jersey" maxlength="2000"></div>
    </div>
    <div class="field"><label for="v3">Special Situation</label><input id="v3" placeholder="e.g. First application, extension, denied" maxlength="2000"></div>
    <button type="button" class="btn" id="vb" data-action="visa">Generate Visa Plan</button>
    <div class="output-wrap"><div id="vo" class="output">Results will appear here...</div><button type="button" class="copy-btn" data-copy-target="vo">Copy</button></div>
  </div></div>

  <div id="tax" class="tab"><div class="card">
    <h2><i class="fas fa-calculator"></i> Tax Refund & Forms</h2>
    <div class="hint">💰 <strong>Tip:</strong> File 1040NR in your first year. Add 1099 if you do rideshare.</div>
    <div class="form-row">
      <div class="field"><label for="t1">Form Type</label><select id="t1"><option>W-4 (Payroll)</option><option>1040NR (International)</option><option>1099-K (Rideshare)</option><option>W-2 (Employee)</option></select></div>
      <div class="field"><label for="t2">Annual Income ($)</label><input id="t2" type="number" min="0" placeholder="e.g. 35000"></div>
    </div>
    <div class="form-row">
      <div class="field"><label for="t3">Visa Type</label><select id="t3"><option>F-1 / J-1</option><option>H-1B</option><option>Green Card</option><option>Citizen</option></select></div>
      <div class="field"><label for="t4">State</label><input id="t4" placeholder="New Jersey" maxlength="2000"></div>
    </div>
    <button type="button" class="btn" id="tb" data-action="tax">Generate Tax Checklist</button>
    <div class="output-wrap"><div id="to" class="output">Results will appear here...</div><button type="button" class="copy-btn" data-copy-target="to">Copy</button></div>
  </div></div>

  <div id="rideshare" class="tab"><div class="card">
    <h2><i class="fas fa-car"></i> Earn with Uber / Lyft</h2>
    <div class="hint">🚗 <strong>For newcomers:</strong> License + car + SSN is enough. $800-1500/week.</div>
    <div class="form-row">
      <div class="field"><label for="r1">App</label><select id="r1"><option>Uber</option><option>Lyft</option><option>Both</option></select></div>
      <div class="field"><label for="r2">State</label><input id="r2" placeholder="New Jersey" maxlength="2000"></div>
    </div>
    <div class="field"><label for="r3">Topic</label><select id="r3"><option>How do I get started?</option><option>1099 form / taxes</option><option>How much can I earn per week?</option><option>Expense deductions</option></select></div>
    <button type="button" class="btn" id="rb" data-action="rideshare">Generate Plan</button>
    <div class="output-wrap"><div id="ro" class="output">Results will appear here...</div><button type="button" class="copy-btn" data-copy-target="ro">Copy</button></div>
  </div></div>

  <div id="housing" class="tab"><div class="card">
    <h2><i class="fas fa-home"></i> Apartment Rental</h2>
    <div class="hint">🏠 <strong>Tip:</strong> NJ Newark/Paterson 1BR $900-1200. Try Craigslist and Zillow.</div>
    <div class="form-row">
      <div class="field"><label for="e1">City / Area</label><input id="e1" placeholder="e.g. Newark NJ, Jersey City" maxlength="2000"></div>
      <div class="field"><label for="e2">Budget ($/month)</label><input id="e2" type="number" min="0" placeholder="1200"></div>
    </div>
    <div class="field"><label for="e3">Special Situation</label><input id="e3" placeholder="e.g. No SSN, no credit score, have pets" maxlength="2000"></div>
    <button type="button" class="btn" id="eb" data-action="housing">Generate Plan</button>
    <div class="output-wrap"><div id="eo" class="output">Results will appear here...</div><button type="button" class="copy-btn" data-copy-target="eo">Copy</button></div>
  </div></div>

  <div id="health" class="tab"><div class="card">
    <h2><i class="fas fa-heartbeat"></i> Free Health Insurance</h2>
    <div class="hint">🏥 <strong>Tip:</strong> Medicaid is free in NJ for low income. Some clinics don't require documents.</div>
    <div class="form-row">
      <div class="field"><label for="h1">State</label><input id="h1" placeholder="New Jersey" maxlength="2000"></div>
      <div class="field"><label for="h2">Situation</label><select id="h2"><option>No insurance, how do I get it?</option><option>How do I apply for Medicaid?</option><option>Where are free clinics?</option><option>Can I get insurance without SSN?</option></select></div>
    </div>
    <button type="button" class="btn" id="hb" data-action="health">Generate Guide</button>
    <div class="output-wrap"><div id="ho" class="output">Results will appear here...</div><button type="button" class="copy-btn" data-copy-target="ho">Copy</button></div>
  </div></div>

  <div id="license" class="tab"><div class="card">
    <h2><i class="fas fa-id-card"></i> Driver's License (DMV)</h2>
    <div class="hint">🪪 <strong>Tip:</strong> NJ has the 6 Points of ID system. Even undocumented can get a license.</div>
    <div class="form-row">
      <div class="field"><label for="l1">State</label><input id="l1" placeholder="New Jersey" maxlength="2000"></div>
      <div class="field"><label for="l2">Situation</label><select id="l2"><option>Getting it for the first time</option><option>Converting a foreign license</option><option>No SSN / ITIN</option><option>Need Real ID</option></select></div>
    </div>
    <button type="button" class="btn" id="lb" data-action="license">Generate Guide</button>
    <div class="output-wrap"><div id="lo" class="output">Results will appear here...</div><button type="button" class="copy-btn" data-copy-target="lo">Copy</button></div>
  </div></div>

<div id="ssn" class="tab">
  <div class="card">
    <h2><i class="fas fa-id-card-alt"></i> SSN Application Guide</h2>
    <div class="hint">🆔 <strong>For newcomers:</strong> F-1/J-1 students can get it with CPT/OPT. SSN is required for on-campus jobs.</div>
    <div class="form-row">
      <div class="field">
        <label for="ss1">Visa Type</label>
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
        <label for="ss2">State</label>
        <input id="ss2" placeholder="New Jersey" maxlength="2000">
      </div>
    </div>
    <div class="field">
      <label for="ss3">Situation</label>
      <input id="ss3" placeholder="e.g. CPT approved, waiting for OPT, do I need ITIN?" maxlength="2000">
    </div>
    <button type="button" class="btn" id="ssb" data-action="ssn">Generate SSN Guide</button>
    <div class="output-wrap">
      <div id="sso" class="output">Results will appear here...</div>
      <button type="button" class="copy-btn" data-copy-target="sso">Copy</button>
    </div>
  </div>
</div>

  <div id="bank" class="tab"><div class="card">
    <h2><i class="fas fa-university"></i> Open a Bank Account</h2>
    <div class="hint">💳 <strong>Tip:</strong> Chase/BofA open accounts with passport. Start credit score with a secured card.</div>
    <div class="field"><label for="ba1">Situation</label><select id="ba1"><option>Open bank account without SSN</option><option>Get a credit card</option><option>Build credit score from scratch</option><option>Best free bank?</option></select></div>
    <button type="button" class="btn" id="bb" data-action="bank">Generate Guide</button>
    <div class="output-wrap"><div id="bo" class="output">Results will appear here...</div><button type="button" class="copy-btn" data-copy-target="bo">Copy</button></div>
  </div></div>

  <div id="phone" class="tab"><div class="card">
    <h2><i class="fas fa-phone"></i> US Phone Number</h2>
    <div class="hint">📱 <strong>Tip:</strong> Get a free US number without SSN using Google Voice.</div>
    <div class="field"><label for="p1">Topic</label><select id="p1"><option>Free number (Google Voice)</option><option>Cheap plans (Mint, Visible, T-Mobile)</option><option>Contract plan without SSN</option><option>Cheap international calls</option></select></div>
    <button type="button" class="btn" id="pb" data-action="phone">Generate Guide</button>
    <div class="output-wrap"><div id="po" class="output">Results will appear here...</div><button type="button" class="copy-btn" data-copy-target="po">Copy</button></div>
  </div></div>

  <div id="car" class="tab"><div class="card">
    <h2><i class="fas fa-car-side"></i> Car Rental / Purchase</h2>
    <div class="hint">🚗 <strong>Tip:</strong> You can buy a car without SSN. Start with CarMax/Carvana.</div>
    <div class="form-row">
      <div class="field"><label for="ar1">State</label><input id="ar1" placeholder="New Jersey" maxlength="2000"></div>
      <div class="field"><label for="ar2">Topic</label><select id="ar2"><option>Buy a used car</option><option>Rent a car</option><option>Get car insurance</option><option>Can I buy without SSN?</option></select></div>
    </div>
    <button type="button" class="btn" id="arb" data-action="car">Generate Guide</button>
    <div class="output-wrap"><div id="aro" class="output">Results will appear here...</div><button type="button" class="copy-btn" data-copy-target="aro">Copy</button></div>
  </div></div>

  <div id="transfer" class="tab"><div class="card">
    <h2><i class="fas fa-exchange-alt"></i> Money Transfer (Wise / Zelle)</h2>
    <div class="hint">💸 <strong>Tip:</strong> Send money internationally with the lowest fees using Wise.</div>
    <div class="field"><label for="w1">Topic</label><select id="w1"><option>Send money abroad with Wise</option><option>Wise limits and fees</option><option>How to use Zelle?</option><option>Venmo / CashApp guide</option></select></div>
    <button type="button" class="btn" id="wb" data-action="transfer">Generate Guide</button>
    <div class="output-wrap"><div id="wo" class="output">Results will appear here...</div><button type="button" class="copy-btn" data-copy-target="wo">Copy</button></div>
  </div></div>

  <div id="flights" class="tab"><div class="card">
    <h2><i class="fas fa-plane"></i> Flights & Baggage</h2>
    <div class="hint">✈️ <strong>Tip:</strong> International flights from NJ $400-700. Pay excess baggage 24 hours before for cheaper rates.</div>
    <div class="form-row">
      <div class="field"><label for="u1">Airline</label><select id="u1"><option>Turkish Airlines</option><option>American Airlines</option><option>United</option><option>Delta</option></select></div>
      <div class="field"><label for="u2">Topic</label><select id="u2"><option>Baggage fees and rules</option><option>How to find cheapest tickets?</option><option>Check-in guide</option><option>Refund / cancellation policy</option></select></div>
    </div>
    <button type="button" class="btn" id="ub" data-action="flights">Generate Guide</button>
    <div class="output-wrap"><div id="uo" class="output">Results will appear here...</div><button type="button" class="copy-btn" data-copy-target="uo">Copy</button></div>
  </div></div>

  <div id="ask" class="tab" role="tabpanel"><div class="card">
    <h2><i class="fas fa-question-circle"></i> Ask Any Question</h2>
    <div class="hint">🤖 Ask anything about life in the USA. You'll get a detailed answer.</div>
    <div class="field"><label for="q1">What's your question?</label><textarea id="q1" rows="4" placeholder="e.g. Can I find a job without SSN? What should I do in my first month?" maxlength="2000"></textarea><div class="char-count" id="q1_count" aria-live="polite" aria-atomic="true">0 / 2000</div></div>
    <button type="button" class="btn" id="qb" data-action="ask">Answer</button>
    <div class="output-wrap"><div id="qo" class="output">Answer will appear here...</div><button type="button" class="copy-btn" data-copy-target="qo">Copy</button></div>
  </div></div>

  <div id="feedback" class="tab" role="tabpanel"><div class="card">
    <h2><i class="fas fa-comment-dots"></i> Site Feedback</h2>
    <div class="hint">💬 Share your experience: what worked, what's missing, what should we improve?</div>
    <p>Send your feedback directly to: <a href="mailto:admin@usalivingguide.com">admin@usalivingguide.com</a></p>
  </div></div>

</div>
<div class="footer">
  <span style="font-size:.85em;color:#64748b">
    ⚠️ This tool is for informational purposes only. AI may make mistakes.
    For legal, financial, or medical matters, always consult a professional.
    No liability is accepted for decisions made based on AI output.
  </span>
</div>
<script>
function g(id){return document.getElementById(id).value;}
function quickStart(tab){
  const target=document.getElementById(tab);
  if(!target) return;
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tabs button').forEach(b=>b.classList.remove('active'));
  target.classList.add('active');
  const match=[...document.querySelectorAll('.tabs button')].find(b=>b.dataset.tab===tab);
  if(match) match.classList.add('active');
  const firstInput=target.querySelector('input,select,textarea');
  if(firstInput) firstInput.focus({preventScroll:true});
  // Delay scroll to allow tab content to render before scrolling into view
  setTimeout(()=>target.scrollIntoView({behavior:'smooth',block:'start'}),50);
}
function show(tab,btn){
  const target=document.getElementById(tab);
  if(!target) return;
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.querySelectorAll('.tabs button').forEach(b=>{b.classList.remove('active');b.setAttribute('aria-selected','false');});
  target.classList.add('active');
  if(btn){btn.classList.add('active');btn.setAttribute('aria-selected','true');}
}
function cp(id){
  navigator.clipboard.writeText(document.getElementById(id).innerText).then(()=>{
    const btn=document.querySelector('#'+id).parentNode.querySelector('.copy-btn');
    btn.textContent='Copied!';
    setTimeout(()=>btn.textContent='Copy',2000);
  });
}
function formatResult(text){
  function escape(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');}
  const paragraphs=text.split(/\\n{2,}/);
  return paragraphs.map(para=>{
    const lines=para.split('\\n').filter(l=>l.trim());
    if(!lines.length) return '';
    const isOrdered=lines.every(l=>/^\\d+\\.\\s/.test(l.trim()));
    const isBullet=lines.every(l=>/^[-\u2022]\\s/.test(l.trim()));
    if(isOrdered){
      const items=lines.map(l=>'<li>'+escape(l.trim().replace(/^\\d+\\.\\s/,''))+'</li>').join('');
      return '<ol>'+items+'</ol>';
    }
    if(isBullet){
      const items=lines.map(l=>'<li>'+escape(l.trim().replace(/^[-\u2022]\\s/,''))+'</li>').join('');
      return '<ul>'+items+'</ul>';
    }
    return lines.map(l=>{
      const t=l.trim();
      if(!t) return '';
      if(/^[A-Z][^a-z]{3,}$/.test(t)||(/[:\uFF1A]$/.test(t)&&t.length<80)){
        return '<p><strong>'+escape(t)+'</strong></p>';
      }
      return '<p>'+escape(t)+'</p>';
    }).join('');
  }).join('');
}
async function call(endpoint,data,outId,btnId,label){
  const out=document.getElementById(outId);
  const btn=document.getElementById(btnId);
  btn.disabled=true;
  btn.innerHTML='<span class="spinner"></span> Loading...';
  out.classList.remove('error');
  out.classList.add('loading');
  out.textContent='Generating your guide...';
  try{
    const r=await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
    const j=await r.json().catch(()=>({}));
    out.classList.remove('loading');
    if(!r.ok){
      out.classList.add('error');
      out.textContent='⚠️ Error: '+(j.error || 'Request could not be processed.');
      return;
    }
    out.innerHTML=formatResult(j.result || 'Could not generate result.');
  }catch(e){
    out.classList.remove('loading');
    out.classList.add('error');
    out.textContent='⚠️ Connection error: '+e.message;
  }finally{
    btn.disabled=false;
    btn.textContent=label;
    out.scrollIntoView({behavior:'smooth',block:'nearest'});
  }
}

const ACTIONS = {
  visa: () => call('/visa',{type:g('v1'),state:g('v2'),situation:g('v3')},'vo','vb','Generate Visa Plan'),
  tax: () => call('/tax',{form:g('t1'),income:g('t2'),visa:g('t3'),state:g('t4')},'to','tb','Generate Tax Checklist'),
  rideshare: () => call('/rideshare',{app:g('r1'),state:g('r2'),topic:g('r3')},'ro','rb','Rideshare Guide'),
  housing: () => call('/housing',{city:g('e1'),budget:g('e2'),situation:g('e3')},'eo','eb','Housing Guide'),
  health: () => call('/health',{state:g('h1'),situation:g('h2')},'ho','hb','Health Guide'),
  license: () => call('/license',{state:g('l1'),situation:g('l2')},'lo','lb','License Guide'),
  ssn: () => call('/ssn',{visa:g('ss1'),state:g('ss2'),situation:g('ss3')},'sso','ssb','Generate SSN Guide'),
  bank: () => call('/bank',{situation:g('ba1')},'bo','bb','Bank Guide'),
  phone: () => call('/phone',{topic:g('p1')},'po','pb','Phone Guide'),
  car: () => call('/car',{state:g('ar1'),topic:g('ar2')},'aro','arb','Car Guide'),
  transfer: () => call('/transfer',{topic:g('w1')},'wo','wb','Money Transfer Guide'),
  flights: () => call('/flights',{airline:g('u1'),topic:g('u2')},'uo','ub','Flight Guide'),
  ask: () => call('/ask',{question:g('q1')},'qo','qb','Answer')
};

document.addEventListener('DOMContentLoaded', () => {
  document.querySelector('.hero h1').addEventListener('click', () => location.reload());
  document.querySelectorAll('[data-tab]').forEach(btn => {
    btn.addEventListener('click', () => show(btn.dataset.tab, btn));
  });
  document.querySelectorAll('[data-quickstart]').forEach(el => {
    el.addEventListener('click', () => quickStart(el.dataset.quickstart));
  });
  document.querySelectorAll('[data-copy-target]').forEach(btn => {
    btn.addEventListener('click', () => cp(btn.dataset.copyTarget));
  });
  document.querySelectorAll('[data-action]').forEach(btn => {
    btn.addEventListener('click', () => {
      const fn = ACTIONS[btn.dataset.action];
      if (fn) fn();
    });
  });
  // Character counters for textareas with maxlength
  document.querySelectorAll('textarea[maxlength]').forEach(ta => {
    const countEl = document.getElementById(ta.id + '_count');
    if (!countEl) return;
    const max = ta.getAttribute('maxlength');
    const update = () => { countEl.textContent = ta.value.length + ' / ' + max; };
    ta.addEventListener('input', update);
    update();
  });
});

</script>
</body>
</html>"""# ─── ROUTES ──────────────────────────────────────────
@app.route('/')
def index():
    response = make_response(HTML)
    response.headers['Content-Type'] = 'text/html'
    return response

@app.route('/healthz')
def healthz():
    active_providers = [
        name for name, key_env in [
            ('groq', 'GROQ_KEY'), ('cerebras', 'CEREBRAS_KEY'), ('gemini', 'GEMINI_KEY'),
            ('cohere', 'COHERE_KEY'), ('mistral', 'MISTRAL_KEY'),
            ('openrouter', 'OPENROUTER_KEY'), ('huggingface', 'HF_KEY'),
        ] if os.environ.get(key_env)
    ]
    return jsonify(status='ok', providers_configured=active_providers)

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
    message = d.get('message', '').strip()
    if not message:
        raise BadRequestError("Message cannot be empty.")
    _feedback_store.append({
        'message': message,
        'contact': d.get('contact', '').strip(),
        'ts': int(time.time())
    })
    return jsonify(result='Thank you! Your feedback has been received.')

# Ensure background refresh starts on import (for gunicorn)
ensure_bg_refresh_started()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
