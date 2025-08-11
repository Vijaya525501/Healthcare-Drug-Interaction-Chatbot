# app.py
# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT SECRETS (don’t change names):
# NEO4J_URI = "neo4j+s://<your-db>.databases.neo4j.io"
# NEO4J_USERNAME = "neo4j"
# NEO4J_PASS = "********"
# NEO4J_DATABASE = "neo4j"
# ─────────────────────────────────────────────────────────────────────────────

import re
from datetime import datetime
from difflib import get_close_matches
from zoneinfo import ZoneInfo

import streamlit as st
from neo4j import GraphDatabase

# Try transformers; app still works without it
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    _HAS_TRF = True
except Exception:
    AutoTokenizer = AutoModelForCausalLM = None
    _HAS_TRF = False

import torch

ATL = ZoneInfo("America/Glace_Bay")  # Atlantic time for all timestamps

st.set_page_config(page_title="Healthcare – Drug Interaction Checker", page_icon="💊", layout="centered")

# ── Minimal styling (solid title color, badges, tidy layout)
st.markdown(
    """
    <style>
      :root{ --border:#e5e7eb; --muted:#6b7280; --soft:#f8fafc; --chip:#f9fafb; }
      .title-wrap{display:flex;align-items:center;gap:.6rem;margin:.25rem 0 .25rem}
      .title-icon{font-size:1.4rem;line-height:1}
      .app-title{font-size:1.35rem;font-weight:800;letter-spacing:.2px;color:#111827}
      .advice{color:var(--muted);font-size:0.92rem;margin:.15rem 0 .9rem}
      .stTextInput>div>div>input{font-size:0.95rem}
      .stButton>button{width:100%}
      .section-h{font-size:1.05rem;font-weight:700;margin:1rem 0 .35rem}
      .pair-title{font-weight:700;margin:.6rem 0 .25rem}
      .pill{display:inline-block;padding:4px 10px;margin:4px 6px 0 0;border-radius:9999px;
            font-size:0.85rem;border:1px solid var(--border);background:var(--chip)}
      .badge{display:inline-block;padding:2px 8px;border-radius:9999px;
             font-size:.80rem;font-weight:700;border:1px solid var(--border)}
      .badge-hit{background:#fef2f2;color:#7f1d1d;border-color:#fecaca}
      .badge-ok{background:#ecfdf5;color:#065f46;border-color:#a7f3d0}
      .types-line{margin:.35rem 0 .2rem;color:#111827;font-size:.95rem}
      .type-chip{display:inline-block;margin-right:.4rem;margin-top:.2rem;
                 padding:2px 8px;border-radius:9999px;border:1px solid var(--border);background:#fff;font-size:.80rem}
      .para-card{border:1px solid var(--border);background:#f6f7f9;border-radius:10px;padding:10px;margin-top:.35rem}
      .para-head{font-weight:700;margin-bottom:4px;color:#111827}
      .conv-row{display:flex;gap:.6rem;align-items:flex-start;margin:.5rem 0 .8rem}
      .conv-ico{font-size:1.1rem}
      .conv-card{flex:1;border:1px solid var(--border);background:var(--soft);
                 border-radius:10px;padding:10px}
      .conv-meta{color:#374151;font-size:.8rem;margin-bottom:4px}
    </style>
    """,
    unsafe_allow_html=True,
)

# Title + subline
st.markdown(
    """
    <div class="title-wrap">
      <span class="title-icon">💊</span>
      <div class="app-title">Healthcare — Drug–Drug Interaction Checker</div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="advice">Use the drug checker to find interactions and get simple guidance.</div>', unsafe_allow_html=True)

# ── Secrets
NEO4J_URI  = st.secrets["NEO4J_URI"]
NEO4J_USER = st.secrets["NEO4J_USERNAME"]
NEO4J_PASS = st.secrets["NEO4J_PASS"]
DB_NAME    = st.secrets["NEO4J_DATABASE"]

# ── Cache: driver + (optional) GPT-2
@st.cache_resource(show_spinner=False)
def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

@st.cache_resource(show_spinner=False)
def get_llm():
    if not _HAS_TRF:
        return None, None
    model_id = "gpt2"
    try:
        tok = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        mdl = AutoModelForCausalLM.from_pretrained(model_id)
        return tok, mdl
    except Exception:
        return None, None

driver = get_driver()
tokenizer, model = get_llm()

# ── Helpers
STOPWORDS = {
    "a","an","and","are","as","at","be","by","for","from","has","he","in","is","it","its",
    "of","on","that","the","to","was","were","will","with","you","your","yours","we","us",
    "can","i","what","how","why","reason","when","now","take","am","having","together",
    "safe","safely","use","using","mix","combine","combined","are","do","does","okay","ok",
    "there","any","interactions","between"
}
WORD_RE = re.compile(r"\b[a-zA-Z][a-zA-Z]+\b")
ALIASES = {
    "acetaminophen":"paracetamol","tylenol":"paracetamol","panadol":"paracetamol",
    "advil":"ibuprofen","motrin":"ibuprofen","asa":"aspirin"
}
EXIT_PATTERN = re.compile(r"\b(bye|goodbye|exit|quit|stop|thanks|thank you)\b", re.I)

GET_KNOWN_DRUGS = "MATCH (d:Drug) RETURN toLower(d.name) AS name"
GET_DETAILS = """
MATCH (d:Drug)
WHERE toLower(d.name) = $drug
OPTIONAL MATCH (d)-[:TREATS]->(c:Condition)
OPTIONAL MATCH (d)-[:HAS_SIDE_EFFECT]->(s:SideEffect)
OPTIONAL MATCH (d)-[:HAS_WARNING]->(w:Warning)
OPTIONAL MATCH (d)-[:HAS_PRECAUTION]->(p:Precaution)
OPTIONAL MATCH (d)-[:CAUSES]->(ca:Cause)
RETURN d.name AS Drug,
       collect(DISTINCT coalesce(c.normalized_name, c.condition_name)) AS Treats,
       collect(DISTINCT s.sideeffect_name) AS SideEffects,
       collect(DISTINCT w.warning_name) AS Warnings,
       collect(DISTINCT p.precaution_name) AS Precautions,
       collect(DISTINCT ca.cause_name) AS Causes
"""
GET_INTERACTION = """
MATCH (d1:Drug)-[r:INTERACTS_WITH]-(d2:Drug)
WHERE (toLower(d1.name) = $drug1 AND toLower(d2.name) = $drug2)
   OR (toLower(d1.name) = $drug2 AND toLower(d2.name) = $drug1)
RETURN d1.name AS Drug1, d2.name AS Drug2, r.reason AS reason, r.type AS type
"""

@st.cache_data(show_spinner=False, ttl=300)
def get_known_drug_names():
    with driver.session(database=DB_NAME) as s:
        return {rec["name"] for rec in s.run(GET_KNOWN_DRUGS)}

def flatten_unique(seq):
    seen, out = set(), []
    for x in (seq or []):
        if not x: continue
        x = str(x).strip()
        if x and x not in seen:
            seen.add(x); out.append(x)
    return out

def extract_drugs(text: str):
    known = get_known_drug_names()
    words = WORD_RE.findall(text or "")
    found, ignored = [], []
    for w in words:
        lw = w.lower()
        if lw in STOPWORDS:
            ignored.append(w); continue
        lw = ALIASES.get(lw, lw)
        if lw in known:
            found.append(lw)
        else:
            m = get_close_matches(lw, known, n=1, cutoff=0.88)
            if m: found.append(m[0])
            else: ignored.append(w)
    seen=set(); out=[]
    for d in found:
        if d not in seen:
            out.append(d); seen.add(d)
    return out, ignored

def _escape_html(s: str) -> str:
    return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

# ---------- Guidance generation (anti-repetition) ----------

_RISK_MAP = [
    (re.compile(r"\bbleed\w*|gi bleed|ulcer", re.I), "bleeding risk"),
    (re.compile(r"\bsedat\w*|drowsi\w*|cns depress", re.I), "drowsiness and slowed reactions"),
    (re.compile(r"\bserotonin", re.I), "serotonin syndrome risk"),
    (re.compile(r"\bqt\b|qt prolong|long qt|torsad", re.I), "heart rhythm problems"),
    (re.compile(r"\bhypo?glyc", re.I), "blood sugar changes"),
    (re.compile(r"\bhyperk", re.I), "high potassium"),
    (re.compile(r"\brenal|\bkidney", re.I), "kidney strain"),
    (re.compile(r"\bhepatic|\bliver", re.I), "liver stress"),
    (re.compile(r"\bblood pressure|hypotens|hypertens", re.I), "blood pressure changes"),
    (re.compile(r"\bgi|stomach|gastric", re.I), "stomach/intestinal side effects"),
]

def _pick_risks(reason_text: str, d1_info: dict, d2_info: dict):
    blob = " ".join([
        reason_text or "",
        " ".join(d1_info.get("Warnings") or []),
        " ".join(d1_info.get("Precautions") or []),
        " ".join(d1_info.get("SideEffects") or []),
        " ".join(d2_info.get("Warnings") or []),
        " ".join(d2_info.get("Precautions") or []),
        " ".join(d2_info.get("SideEffects") or []),
    ])
    hits = []
    seen = set()
    for rx, label in _RISK_MAP:
        if rx.search(blob) and label not in seen:
            hits.append(label); seen.add(label)
        if len(hits) >= 2:
            break
    return hits

def _severity_from_types(types):
    s = "may interact"
    joined = " ".join([t.lower() for t in types])
    if "contra" in joined:
        s = "should generally not be used together"
    elif "major" in joined or "severe" in joined:
        s = "can cause serious problems"
    elif "moderate" in joined:
        s = "may increase risk"
    return s

def _clean_guidance_text(txt: str) -> str:
    if not txt: return ""
    # remove "Advice:" prefixes and collapse repeats (a a a → a)
    txt = re.sub(r"(?i)\badvice\s*:\s*", "", txt).strip()
    txt = re.sub(r"\b(\w+)(\s+\1\b){1,}", r"\1", txt)  # collapse repeated words
    # keep one sentence, end with period
    txt = txt.split("\n")[-1].strip()
    txt = txt.split(".")[0].strip()
    if txt and not txt.endswith("."):
        txt += "."
    return txt

def _guidance_with_gpt2(reason_text: str, types, d1_info: dict, d2_info: dict) -> str:
    if not (tokenizer and model):
        return ""
    ctx_bits = []
    for label, key in [("Warnings", "Warnings"), ("Precautions", "Precautions"), ("Side effects", "SideEffects")]:
        vals = (d1_info.get(key) or []) + (d2_info.get(key) or [])
        uniq = []
        seen = set()
        for v in vals:
            v = str(v).strip()
            if v and v not in seen:
                uniq.append(v); seen.add(v)
        if uniq:
            ctx_bits.append(f"{label}: {', '.join(uniq[:6])}")
    ctx = " ".join(ctx_bits)[:300]
    type_line = ", ".join(types) if types else "Unknown"

    prompt = (
        "Using only the information provided, write one short, patient-friendly sentence of guidance. "
        "Do not add new facts. Keep it under 25 words.\n"
        f"Interaction types: {type_line}\n"
        f"Reason: {reason_text if reason_text else 'n/a'}\n"
        f"Context: {ctx if ctx else 'n/a'}\n"
        "Advice:"
    )

    eos_id = getattr(tokenizer, "eos_token_id", None)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=48,
            do_sample=True,
            temperature=0.7,
            top_p=0.92,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            pad_token_id=eos_id,
            eos_token_id=eos_id,
        )
    txt = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return _clean_guidance_text(txt)

def guidance_sentence(reason_text: str, types, d1_info: dict, d2_info: dict) -> str:
    """Hybrid: try GPT-2 with anti-repetition; if weak, fall back to deterministic guidance."""
    # 1) try model
    g = _guidance_with_gpt2(reason_text, types, d1_info, d2_info) if (tokenizer and model) else ""
    # if too short or still weird, fall back
    if len(g.split()) < 4:
        risks = _pick_risks(reason_text, d1_info, d2_info)
        sev = _severity_from_types(types)
        if risks:
            risk_text = " or ".join(risks)
            g = f"This combination {sev} due to potential {risk_text}. Talk to your healthcare provider."
        else:
            g = f"This combination {sev}. Talk to your healthcare provider."
    return g

# ── Session state for logs (not shown on screen)
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

# ── INPUT (form clears on submit)
with st.form("qform", clear_on_submit=True):
    query = st.text_input(
        "Question",
        placeholder="Enter drug names (e.g., Ibuprofen with Paracetamol?)",
        key="query",
        label_visibility="collapsed",
    )
    go = st.form_submit_button("Check interactions", type="primary")

# ── PROCESS
if go:
    user_text = (query or "").strip()
    when = datetime.now(ATL).strftime("%Y-%m-%d %H:%M:%S %Z")

    # show the user's question card
    st.markdown(
        f"""
        <div class="conv-row">
          <div class="conv-ico">🧑‍💬</div>
          <div class="conv-card">
            <div class="conv-meta">{when} — User</div>
            {_escape_html(user_text)}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # store in log
    if user_text:
        st.session_state.chat_log.append({"role":"user","text":user_text,"time":when})

    # exit intent (bye/exit/thanks)
    if user_text and EXIT_PATTERN.search(user_text):
        goodbye = "Goodbye! Have a nice day."
        st.session_state.chat_log.append({"role":"bot","text":goodbye,"time":when})
        st.markdown(
            f"""
            <div class="conv-row">
              <div class="conv-ico">🤖</div>
              <div class="conv-card">
                <div class="conv-meta">{when} — Bot</div>
                {goodbye}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # normal processing
        def get_known_drug_names_cached():
            return get_known_drug_names()

        drugs, ignored_terms = extract_drugs(user_text)
        # keep ignored terms only in logs (no on-screen warning)
        if ignored_terms:
            st.session_state.chat_log.append(
                {"role":"system","text":"ignored_terms: "+", ".join(ignored_terms), "time": when}
            )

        # Drugs detected (keep)
        if drugs:
            st.markdown('<div class="section-h">Drugs detected</div>', unsafe_allow_html=True)
            st.markdown("".join([f'<span class="pill">{d}</span>' for d in drugs]), unsafe_allow_html=True)

        if len(drugs) < 2:
            st.info("Please include at least two valid drug names.")
        else:
            st.markdown('<div class="section-h">Results</div>', unsafe_allow_html=True)

            bot_lines = []
            for i in range(len(drugs)):
                for j in range(i+1, len(drugs)):
                    d1, d2 = drugs[i], drugs[j]
                    with driver.session(database=DB_NAME) as s:
                        d1_info = get_drug_details(d1)
                        d2_info = get_drug_details(d2)
                        recs    = s.run(GET_INTERACTION, drug1=d1, drug2=d2).data()

                    pair_title = f"{d1_info.get('Drug', d1)} × {d2_info.get('Drug', d2)}"
                    st.markdown(f'<div class="pair-title">{pair_title}</div>', unsafe_allow_html=True)

                    if recs:
                        # collect types and internal reasons (not shown)
                        types = sorted({(r.get("type") or "Unknown") for r in recs})
                        combined_reason = " ".join([r.get("reason","").strip() for r in recs if r.get("reason")]).strip()

                        st.markdown('<span class="badge badge-hit">Interaction in database</span>', unsafe_allow_html=True)

                        # show types as chips
                        if types:
                            st.markdown('<div class="types-line">Interaction type(s): ' +
                                        "".join([f'<span class="type-chip">{_escape_html(t)}</span>' for t in types]) +
                                        '</div>', unsafe_allow_html=True)

                        # details
                        with st.expander(f"Details — {d1_info.get('Drug', d1)}"):
                            if d1_info.get("Treats"): st.write("Treats: " + ", ".join(d1_info["Treats"]))
                            if d1_info.get("SideEffects"): st.write("Side effects: " + ", ".join(d1_info["SideEffects"]))
                            if d1_info.get("Warnings"): st.write("Warnings: " + ", ".join(d1_info["Warnings"]))
                            if d1_info.get("Precautions"): st.write("Precautions: " + ", ".join(d1_info["Precautions"]))
                        with st.expander(f"Details — {d2_info.get('Drug', d2)}"):
                            if d2_info.get("Treats"): st.write("Treats: " + ", ".join(d2_info["Treats"]))
                            if d2_info.get("SideEffects"): st.write("Side effects: " + ", ".join(d2_info["SideEffects"]))
                            if d2_info.get("Warnings"): st.write("Warnings: " + ", ".join(d2_info["Warnings"]))
                            if d2_info.get("Precautions"): st.write("Precautions: " + ", ".join(d2_info["Precautions"]))

                        # Bot response (now robust)
                        guidance = guidance_sentence(combined_reason, types, d1_info, d2_info)
                        st.markdown(f'<div class="para-card"><div class="para-head">Bot response</div>{_escape_html(guidance)}</div>', unsafe_allow_html=True)
                        bot_lines.append(f"{pair_title} — {guidance}")
                    else:
                        st.markdown('<span class="badge badge-ok">No known interaction found</span>', unsafe_allow_html=True)

                        with st.expander(f"Details — {d1_info.get('Drug', d1)}"):
                            if d1_info.get("Treats"): st.write("Treats: " + ", ".join(d1_info["Treats"]))
                            if d1_info.get("SideEffects"): st.write("Side effects: " + ", ".join(d1_info["SideEffects"]))
                            if d1_info.get("Warnings"): st.write("Warnings: " + ", ".join(d1_info["Warnings"]))
                            if d1_info.get("Precautions"): st.write("Precautions: " + ", ".join(d1_info["Precautions"]))
                        with st.expander(f"Details — {d2_info.get('Drug', d2)}"):
                            if d2_info.get("Treats"): st.write("Treats: " + ", ".join(d2_info["Treats"]))
                            if d2_info.get("SideEffects"): st.write("Side effects: " + ", ".join(d2_info["SideEffects"]))
                            if d2_info.get("Warnings"): st.write("Warnings: " + ", ".join(d2_info["Warnings"]))
                            if d2_info.get("Precautions"): st.write("Precautions: " + ", ".join(d2_info["Precautions"]))

                        st.markdown(f'<div class="para-card"><div class="para-head">Bot response</div>No interaction found for this pair.</div>', unsafe_allow_html=True)
                        bot_lines.append(f"{pair_title} — no interaction found")

            if bot_lines:
                st.session_state.chat_log.append({"role":"bot","text":"\n".join(bot_lines),"time":when})
