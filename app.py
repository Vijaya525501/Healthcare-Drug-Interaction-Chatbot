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

import streamlit as st
from neo4j import GraphDatabase

# Transformers import (version-proof)
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    from transformers.models.auto import AutoTokenizer
    from transformers import AutoModelForCausalLM

import torch

st.set_page_config(page_title="Healthcare – Drug Interaction Checker", page_icon="💊", layout="centered")

# ── Styling
st.markdown(
    """
    <style>
      :root{
        --border:#e5e7eb;
        --muted:#6b7280;
        --soft:#f8fafc;
        --chip:#f9fafb;
      }
      .title-wrap{display:flex;align-items:center;gap:.6rem;margin:.25rem 0 .25rem}
      .title-icon{font-size:1.4rem;line-height:1}
      .app-title{
        font-size:1.35rem;font-weight:800;letter-spacing:.2px;
        background:linear-gradient(90deg,#111827 0%,#0ea5e9 60%,#14b8a6 100%);
        -webkit-background-clip:text;background-clip:text;color:transparent;
      }
      .advice{color:var(--muted);font-size:0.92rem;margin:.15rem 0 .9rem}
      .stTextInput>div>div>input{font-size:0.95rem}
      .stButton>button{width:100%}
      .section-h{font-size:1.05rem;font-weight:700;margin:.9rem 0 .35rem}
      .pair-title{font-weight:700;margin:.6rem 0 .25rem}
      .badge{display:inline-block;padding:2px 8px;border-radius:9999px;
             font-size:.80rem;font-weight:700;border:1px solid var(--border)}
      .badge-hit{background:#fef2f2;color:#7f1d1d;border-color:#fecaca}
      .badge-ok{background:#ecfdf5;color:#065f46;border-color:#a7f3d0}
      .pill{display:inline-block;padding:4px 10px;margin:4px 6px 0 0;border-radius:9999px;
            font-size:0.85rem;border:1px solid var(--border);background:var(--chip)}
      .para-card{border:1px solid var(--border);background:#f6f7f9;border-radius:10px;padding:10px}
      .para-head{font-weight:700;margin-bottom:4px;color:#111827}
      .conv-row{display:flex;gap:.6rem;align-items:flex-start;margin:1rem 0 .2rem}
      .conv-ico{font-size:1.1rem}
      .conv-card{flex:1;border:1px solid var(--border);background:var(--soft);
                 border-radius:10px;padding:10px}
      .conv-meta{color:var(--muted);font-size:.8rem;margin-bottom:4px}
    </style>
    """,
    unsafe_allow_html=True,
)

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

# ── Cache: driver + GPT-2 (slow tokenizer avoids Rust wheels)
@st.cache_resource(show_spinner=False)
def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

@st.cache_resource(show_spinner=False)
def get_llm():
    model_id = "gpt2"
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    mdl = AutoModelForCausalLM.from_pretrained(model_id)
    return tok, mdl

driver = get_driver()
tokenizer, model = get_llm()

# ── Stopwords, aliases, extraction helpers
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

# Exit intent detection
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

def paraphrase_reason_one_sentence(text: str) -> str:
    """One short, patient-friendly sentence. No extra info; just rephrase."""
    t = (text or "").strip()
    if not t: return ""
    prompt = (
        "Rewrite the following in one short, patient-friendly sentence. "
        "Do not add information or name interaction types. "
        f'Text: "{t}"'
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    eos_id = getattr(tokenizer, "eos_token_id", None)  # GPT-2 pad fix
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model.generate(
            **inputs, max_new_tokens=50, do_sample=False,
            pad_token_id=eos_id, eos_token_id=eos_id
        )
    out = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    out = out.split(".")[0].strip()
    return (out + ".") if out else t

def all_pairs(drugs):
    n = len(drugs)
    for i in range(n):
        for j in range(i + 1, n):
            yield drugs[i], drugs[j]

def _escape_html(s: str) -> str:
    return s.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

# ── Session state for logs (NOT displayed)
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

# ── INPUT (form clears on submit)
with st.form("qform", clear_on_submit=True):
    query = st.text_input(
        "Question",
        placeholder="Enter drug names (e.g., Ibuprofen with Dexamethasone?)",
        key="query",
        label_visibility="collapsed",
    )
    go = st.form_submit_button("Check interactions", type="primary")

latest_user = None
latest_bot  = None

# ── PROCESS
if go:
    user_text = (query or "").strip()
    when = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if user_text:
        st.session_state.chat_log.append({"role":"user","text":user_text,"time":when})
        latest_user = {"text": user_text, "time": when}

    # Exit intent: say goodbye and skip processing
    if user_text and EXIT_PATTERN.search(user_text):
        goodbye = "Goodbye! Have a nice day."
        st.session_state.chat_log.append({"role":"bot","text":goodbye,"time":when})
        latest_bot = {"text": goodbye, "time": when}
    else:
        drugs, ignored_terms = extract_drugs(user_text)

        # keep ignored terms only in logs (no on-screen warning)
        if ignored_terms:
            st.session_state.chat_log.append(
                {"role":"system","text":"ignored_terms: "+", ".join(ignored_terms), "time": when}
            )

        if len(drugs) < 2:
            st.info("Please include at least two valid drug names.")
        else:
            # context chips
            st.markdown('<div class="section-h">Drugs detected</div>', unsafe_allow_html=True)
            st.markdown("".join([f'<span class="pill">{d}</span>' for d in drugs]), unsafe_allow_html=True)

            bot_lines = []
            st.markdown('<div class="section-h">Results</div>', unsafe_allow_html=True)

            for d1, d2 in all_pairs(drugs):
                d1_info = get_drug_details(d1)
                d2_info = get_drug_details(d2)
                recs    = get_interactions(d1, d2)

                pair_title = f"{d1_info.get('Drug', d1)} × {d2_info.get('Drug', d2)}"
                st.markdown(f'<div class="pair-title">{pair_title}</div>', unsafe_allow_html=True)

                if recs:
                    # dedupe reasons
                    raw_reasons = [r.get("reason","").strip() for r in recs if r.get("reason")]
                    reasons = []
                    seen = set()
                    for rr in raw_reasons:
                        if rr and rr not in seen:
                            reasons.append(rr); seen.add(rr)
                    combined_reason = " ".join(reasons).strip()

                    st.markdown('<span class="badge badge-hit">Interaction in database</span>', unsafe_allow_html=True)

                    if combined_reason:
                        guidance = paraphrase_reason_one_sentence(combined_reason)
                        if guidance:
                            st.markdown(f'<div class="para-card"><div class="para-head">Guidance</div>{guidance}</div>', unsafe_allow_html=True)
                            bot_lines.append(f"{pair_title} — {guidance}")
                    with st.expander("Neo4j reasons", expanded=False):
                        if reasons:
                            for i, rr in enumerate(reasons, 1):
                                st.write(f"{i}. {rr}")
                        else:
                            st.caption("No reason text present for this relationship.")
                else:
                    st.markdown('<span class="badge badge-ok">No known interaction found</span>', unsafe_allow_html=True)
                    bot_lines.append(f"{pair_title} — no interaction found")

                # Optional compact details
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

            bot_text = "\n".join(bot_lines)
            st.session_state.chat_log.append({"role":"bot","text":bot_text,"time":when})
            latest_bot = {"text": bot_text, "time": when}

# ── Compact conversation header (shows only latest turn)
if latest_user:
    st.markdown(
        f"""
        <div class="conv-row">
          <div class="conv-ico">🧑‍💬</div>
          <div class="conv-card">
            <div class="conv-meta">{latest_user["time"]} — User</div>
            {_escape_html(latest_user["text"])}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
if latest_bot:
    st.markdown(
        f"""
        <div class="conv-row">
          <div class="conv-ico">🤖</div>
          <div class="conv-card">
            <div class="conv-meta">{latest_bot["time"]} — Bot</div>
            {_escape_html(latest_bot["text"]) if latest_bot["text"] else "—"}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

#st.caption("Note: Results are limited to interactions present in your Neo4j graph.")
