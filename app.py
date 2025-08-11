# app.py
# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT SECRETS (don’t change names):
# NEO4J_URI = "neo4j+s://<your-db>.databases.neo4j.io"
# NEO4J_USERNAME = "neo4j"
# NEO4J_PASS = "********"
# NEO4J_DATABASE = "neo4j"
# ─────────────────────────────────────────────────────────────────────────────

import re
from pathlib import Path
from datetime import datetime
from difflib import get_close_matches
from zoneinfo import ZoneInfo

import streamlit as st
from neo4j import GraphDatabase

# LLM (GPT-2; CPU friendly)
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    _HAS_TRF = True
except Exception:
    AutoTokenizer = AutoModelForCausalLM = None
    torch = None
    _HAS_TRF = False

ATL = ZoneInfo("America/Glace_Bay")  # Atlantic time
ALWAYS_LLM = True  # force LLM-only guidance (no deterministic rewrite)

st.set_page_config(page_title="Healthcare – Drug Interaction Checker", page_icon="💊", layout="centered")

# ── Load external CSS (styles.css must sit next to app.py)
def load_css(file_path: str = "styles.css"):
    p = Path(file_path)
    if p.exists():
        st.markdown(f"<style>{p.read_text()}</style>", unsafe_allow_html=True)

load_css("styles.css")

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

# ── Caches
@st.cache_resource(show_spinner=False)
def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

@st.cache_resource(show_spinner=False)
def get_llm():
    if not _HAS_TRF:
        return None, None
    try:
        tok = AutoTokenizer.from_pretrained("gpt2", use_fast=False)  # avoid Rust fast tokenizer
        mdl = AutoModelForCausalLM.from_pretrained("gpt2")
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
    """No aliases. Exact match first, then soft fuzzy match to the DB names."""
    known = get_known_drug_names()
    words = WORD_RE.findall(text or "")
    found, ignored = [], []
    for w in words:
        lw = w.lower()
        if lw in STOPWORDS:
            ignored.append(w); continue
        if lw in known:
            found.append(lw)
        else:
            m = get_close_matches(lw, known, n=1, cutoff=0.80)  # helpful for minor typos
            if m: found.append(m[0])
            else: ignored.append(w)
    seen=set(); out=[]
    for d in found:
        if d not in seen:
            out.append(d); seen.add(d)
    return out, ignored

def get_drug_details(drug_lc: str):
    with driver.session(database=DB_NAME) as s:
        rec = s.run(GET_DETAILS, drug=drug_lc).single()
        if not rec:
            return {"Drug": drug_lc}
        return {
            "Drug": rec["Drug"] or drug_lc,
            "Treats": flatten_unique(rec["Treats"]),
            "SideEffects": flatten_unique(rec["SideEffects"]),
            "Warnings": flatten_unique(rec["Warnings"]),
            "Precautions": flatten_unique(rec["Precautions"]),
            "Causes": flatten_unique(rec["Causes"]),
        }

def get_interactions(d1: str, d2: str):
    with driver.session(database=DB_NAME) as s:
        return s.run(GET_INTERACTION, drug1=d1, drug2=d2).data()

def _escape_html(s: str) -> str:
    return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

# ---------- Interaction type definitions ----------
TYPE_DEFS = {
    "contraindicated": "Do not use together; risks outweigh benefits.",
    "major": "High-risk combination; significant chance of serious side effects.",
    "severe": "High-risk combination; significant chance of serious side effects.",
    "moderate": "May increase side effects; use only with clinical advice.",
    "minor": "Low risk; usually safe, but monitor for symptoms.",
    "antagonistic": "One drug may reduce the effect of the other.",
    "synergistic": "Effects may add up or become stronger together.",
    "pharmacokinetic": "One drug may change the level of the other (absorption/metabolism).",
    "pharmacodynamic": "Similar actions increase effect when combined.",
    "duplicate": "Drugs with the same effect increase overall exposure.",
    "additive": "Effects add together, increasing overall impact.",
}

def type_definitions_line(types):
    defs, seen = [], set()
    for t in types:
        key = t.lower().strip()
        d = TYPE_DEFS.get(key)
        if d and d not in seen:
            defs.append(d); seen.add(d)
        # partial matches
        elif "contra" in key and "Do not use together; risks outweigh benefits." not in seen:
            defs.append("Do not use together; risks outweigh benefits."); seen.add("Do not use together; risks outweigh benefits.")
        elif "major" in key and "High-risk combination; significant chance of serious side effects." not in seen:
            defs.append("High-risk combination; significant chance of serious side effects."); seen.add("High-risk combination; significant chance of serious side effects.")
        elif "moderate" in key and "May increase side effects; use only with clinical advice." not in seen:
            defs.append("May increase side effects; use only with clinical advice."); seen.add("May increase side effects; use only with clinical advice.")
        elif "minor" in key and "Low risk; usually safe, but monitor for symptoms." not in seen:
            defs.append("Low risk; usually safe, but monitor for symptoms."); seen.add("Low risk; usually safe, but monitor for symptoms.")
        elif "antagon" in key and "One drug may reduce the effect of the other." not in seen:
            defs.append("One drug may reduce the effect of the other."); seen.add("One drug may reduce the effect of the other.")
        elif "synerg" in key and "Effects may add up or become stronger together." not in seen:
            defs.append("Effects may add up or become stronger together."); seen.add("Effects may add up or become stronger together.")
    return " ".join(defs[:2]) if defs else ""

# ---------- LLM guidance (primary, single sentence) ----------
def _clean_llm(txt: str) -> str:
    if not txt: return ""
    # remove odd prefixes and words GPT-2 sometimes invents
    banned = ["video", "recording", "above", "guidelines", "elements"]
    txt = re.sub(r"(?i)\b(advice|note|suggestion)\s*:\s*", "", txt).strip()
    for b in banned:
        txt = re.sub(rf"(?i)\b{re.escape(b)}\b", "", txt)
    # collapse repetition: "bleeding bleeding" -> "bleeding"
    txt = re.sub(r"\b(\w+)(\s+\1\b){1,}", r"\1", txt)
    # single short sentence
    txt = txt.replace("\n", " ").strip()
    if "." in txt:
        txt = txt.split(".")[0]
    words = txt.split()
    if len(words) > 24:
        txt = " ".join(words[:24])
    txt = txt.strip()
    if txt and not txt.endswith("."):
        txt += "."
    txt = re.sub(r"\s{2,}", " ", txt)
    return txt

def guidance_with_llm(reason_text: str, types, d1_info: dict, d2_info: dict) -> str:
    if not (tokenizer and model):
        return ""
    # compact context from details (warnings/precautions/side effects)
    def pick(key):
        vals = (d1_info.get(key) or []) + (d2_info.get(key) or [])
        uniq, seen = [], set()
        for v in vals:
            v = str(v).strip()
            if v and v not in seen:
                uniq.append(v); seen.add(v)
        return ", ".join(uniq[:6])

    ctx_bits = []
    for label, key in [("Warnings","Warnings"),("Precautions","Precautions"),("Side effects","SideEffects")]:
        s = pick(key)
        if s:
            ctx_bits.append(f"{label}: {s}")
    ctx = " ".join(ctx_bits)[:300]
    type_line = ", ".join(types) if types else "Unknown"
    reason_line = reason_text if reason_text else "n/a"

    # tight prompt + 2 few-shot examples
    prompt = (
        "You are rewriting drug-interaction information into ONE short, patient-friendly suggestion.\n"
        "Use ONLY the data provided (types/reason/context). Do not invent facts.\n"
        "Return EXACTLY ONE sentence, under 24 words, in plain English.\n"
        "Prefer: 'Avoid', 'Use only with', 'Monitor', 'Separate doses', 'Ask your clinician'.\n"
        "Do not mention videos or guidelines.\n"
        "\n"
        "Example:\n"
        "Types: moderate; Reason: n/a; Context: Warnings: GI bleeding, ulcer. -> Avoid taking together unless advised; risk of stomach bleeding—ask your clinician.\n"
        "Example:\n"
        "Types: pharmacokinetic; Reason: n/a; Context: Precautions: liver disease. -> Use only with advice; one drug can change levels—monitor and consult your clinician.\n"
        "\n"
        f"Types: {type_line}\n"
        f"Reason: {reason_line}\n"
        f"Context: {ctx if ctx else 'n/a'}\n"
        "Suggestion:"
    )

    eos_id = getattr(tokenizer, "eos_token_id", None)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=42,
            do_sample=True,
            temperature=0.4,
            top_p=0.9,
            repetition_penalty=1.3,
            no_repeat_ngram_size=4,
            pad_token_id=eos_id,
            eos_token_id=eos_id,
        )
    txt = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return _clean_llm(txt)

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
        drugs, ignored_terms = extract_drugs(user_text)
        # keep ignored terms only in logs (no on-screen warning)
        if ignored_terms:
            st.session_state.chat_log.append({"role":"system","text":"ignored_terms: "+", ".join(ignored_terms), "time": when})

        # Drugs detected
        if drugs:
            st.markdown('<div class="section-h">Drugs detected</div>', unsafe_allow_html=True)
            chips = " ".join([f'<span class="pill">{_escape_html(d)}</span>' for d in drugs])
            st.markdown(chips, unsafe_allow_html=True)

        if len(drugs) < 2:
            st.info("Please include at least two valid drug names.")
        else:
            st.markdown('<div class="section-h">Results</div>', unsafe_allow_html=True)

            bot_lines = []
            # iterate all unique pairs
            for i in range(len(drugs)):
                for j in range(i+1, len(drugs)):
                    d1, d2 = drugs[i], drugs[j]

                    d1_info = get_drug_details(d1)
                    d2_info = get_drug_details(d2)
                    recs    = get_interactions(d1, d2)

                    pair_title = f"{d1_info.get('Drug', d1)} × {d2_info.get('Drug', d2)}"
                    st.markdown(f'<div class="pair-title">{pair_title}</div>', unsafe_allow_html=True)

                    if recs:
                        # collect types and combine reason internally (not shown)
                        types = sorted({(r.get("type") or "Unknown") for r in recs})
                        combined_reason = " ".join([r.get("reason","").strip() for r in recs if r.get("reason")]).strip()

                        st.markdown('<span class="badge badge-hit">Interaction in database</span>', unsafe_allow_html=True)

                        # type chips + one-line definition
                        if types:
                            st.markdown(
                                '<div class="types-line">Interaction type(s): ' +
                                " ".join([f'<span class="type-chip">{_escape_html(t)}</span>' for t in types]) +
                                '</div>',
                                unsafe_allow_html=True,
                            )
                            meaning = type_definitions_line(types)
                            if meaning:
                                st.markdown(f'<div class="type-def">What this means: { _escape_html(meaning) }</div>', unsafe_allow_html=True)

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

                        # Bot response — LLM only (fallback only if model totally unavailable)
                        guidance = guidance_with_llm(combined_reason, types, d1_info, d2_info)
                        if not guidance:
                            guidance = "Unable to generate advice right now. Please consult your clinician."

                        st.markdown(
                            f'<div class="para-card"><div class="para-head">Bot response</div>{_escape_html(guidance)}</div>',
                            unsafe_allow_html=True,
                        )
                        st.caption("Generated by LLM from graph data.")
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

                        st.markdown(
                            f'<div class="para-card"><div class="para-head">Bot response</div>No interaction found for this pair.</div>',
                            unsafe_allow_html=True,
                        )
                        bot_lines.append(f"{pair_title} — no interaction found")

            if bot_lines:
                st.session_state.chat_log.append({"role":"bot","text":"\n".join(bot_lines),"time":when})
