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

# Minimal, smaller heading + basic styling
st.markdown(
    """
    <style>
      .app-title{font-size:1.10rem;font-weight:700;margin:0.25rem 0 0.75rem}
      .stTextInput>div>div>input{font-size:0.95rem}
      .stButton>button{width:100%}
      .section-h{font-size:1.05rem;font-weight:600;margin:1rem 0 0.35rem}
      .para-box{background:#f6f7f9;border:1px solid #e5e7eb;border-radius:8px;padding:10px}
      .log-time{color:#6b7280;font-size:0.8rem}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<div class="app-title">Healthcare — Drug–Drug Interaction Checker</div>', unsafe_allow_html=True)

# ── Secrets (exact keys)
NEO4J_URI  = st.secrets["NEO4J_URI"]
NEO4J_USER = st.secrets["NEO4J_USERNAME"]
NEO4J_PASS = st.secrets["NEO4J_PASS"]
DB_NAME    = st.secrets["NEO4J_DATABASE"]

# ── Caching: driver and LLM (GPT-2 only)
@st.cache_resource(show_spinner=False)
def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

@st.cache_resource(show_spinner=False)
def get_llm():
    """
    Use base GPT-2 (small, CPU-friendly). Use slow tokenizer to avoid Rust wheels.
    """
    model_id = "gpt2"
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=False)
    mdl = AutoModelForCausalLM.from_pretrained(model_id)
    return tok, mdl

driver = get_driver()
tokenizer, model = get_llm()

# ── Stopwords + tokenization helpers
STOPWORDS = {
    "a","an","and","are","as","at","be","by","for","from","has","he","in","is","it","its",
    "of","on","that","the","to","was","were","will","with","you","your","yours","we","us",
    "can","i","what","how","why","reason","when","now","take","am","having"
}
WORD_RE = re.compile(r"\b[a-zA-Z][a-zA-Z]+\b")

# ── Cypher queries
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
        if not x:
            continue
        x = str(x).strip()
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

def extract_drugs(text: str):
    known = get_known_drug_names()
    words = WORD_RE.findall(text or "")
    found, unknown = [], []
    for w in words:
        lw = w.lower()
        if lw in STOPWORDS:
            continue
        if lw in known:
            found.append(lw)
        else:
            m = get_close_matches(lw, known, n=1, cutoff=0.88)
            if m:
                found.append(m[0])
            else:
                unknown.append(w)
    # unique while preserving order
    seen = set(); out = []
    for d in found:
        if d not in seen:
            out.append(d); seen.add(d)
    return out, unknown

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
    if not t:
        return ""
    prompt = (
        "Rewrite the following in one short, patient-friendly sentence. "
        "Do not add information or name interaction types. "
        f'Text: "{t}"'
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    # GPT-2 has no pad token: use eos as pad if needed
    eos_id = getattr(tokenizer, "eos_token_id", None)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=eos_id,
            eos_token_id=eos_id,
        )
    out = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    out = out.split(".")[0].strip()
    return (out + ".") if out else t

def all_pairs(drugs):
    n = len(drugs)
    for i in range(n):
        for j in range(i + 1, n):
            yield drugs[i], drugs[j]

# ── Session state for logs
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

# ── UI
query = st.text_input(
    "Question",
    placeholder="Enter 2+ drug names in natural language…",
    key="query",
    label_visibility="collapsed",
)
go = st.button("Check interactions", type="primary")

if go:
    user_text = (st.session_state.query or "").strip()
    if user_text:
        st.session_state.chat_log.append(
            {"role": "user", "text": user_text, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        )

    drugs, unknown = extract_drugs(user_text)
    if unknown:
        st.warning("Unrecognized names: " + ", ".join(unknown))
    if len(drugs) < 2:
        st.info("Please include at least two valid drug names.")
    else:
        bot_summary_lines = []
        for d1, d2 in all_pairs(drugs):
            d1_info = get_drug_details(d1)
            d2_info = get_drug_details(d2)
            recs = get_interactions(d1, d2)

            st.markdown(f'<div class="section-h">{d1_info.get("Drug", d1)} × {d2_info.get("Drug", d2)}</div>', unsafe_allow_html=True)
            if recs:
                reasons = [r.get("reason", "").strip() for r in recs if r.get("reason")]
                reasons = [r for r in reasons if r]
                types = sorted({(r.get("type") or "Unknown") for r in recs})
                combined_reason = " ".join(reasons).strip()

                with st.expander("Neo4j reasons", expanded=False):
                    for i, rr in enumerate(reasons, 1):
                        st.write(f"{i}. {rr}")

                paraphrase = paraphrase_reason_one_sentence(combined_reason) if combined_reason else ""
                if paraphrase:
                    st.markdown(f'<div class="para-box">{paraphrase}</div>', unsafe_allow_html=True)
                    bot_summary_lines.append(f"{d1_info.get('Drug', d1)} × {d2_info.get('Drug', d2)} — {paraphrase}")
                else:
                    st.write("Reason text is missing in the graph.")
                    bot_summary_lines.append(f"{d1_info.get('Drug', d1)} × {d2_info.get('Drug', d2)} — reason missing")

                st.write("Type(s): " + ", ".join(types))
            else:
                st.write("No known interaction found in the database.")
                bot_summary_lines.append(f"{d1_info.get('Drug', d1)} × {d2_info.get('Drug', d2)} — no interaction found")

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

        st.session_state.chat_log.append(
            {"role": "bot", "text": "\n".join(bot_summary_lines), "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        )

    # Clear the input after processing
    st.session_state.query = ""

# ── Simple timestamped log at the bottom
if st.session_state.chat_log:
    st.markdown('<div class="section-h">Log</div>', unsafe_allow_html=True)
    for item in st.session_state.chat_log[::-1]:  # newest first
        who = "User" if item["role"] == "user" else "Bot"
        st.markdown(f'<span class="log-time">{item["time"]} — {who}</span>', unsafe_allow_html=True)
        st.write(item["text"])
        st.markdown("---")

st.caption("Note: Results are limited to interactions present in your Neo4j graph.")
