# app.py
# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT SECRETS REQUIRED:
# NEO4J_URI = "neo4j+s://<your-db>.databases.neo4j.io"
# NEO4J_USER = "neo4j"
# NEO4J_PASS = "********"
# DB_NAME    = "neo4j"             # optional; defaults to "neo4j"
# ─────────────────────────────────────────────────────────────────────────────

import re
from difflib import get_close_matches
import streamlit as st
from neo4j import GraphDatabase
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

st.set_page_config(page_title="Healthcare – Drug Interaction Checker", page_icon="💊", layout="centered")
st.title("💊 Healthcare — Drug–Drug Interaction Checker")

# ── Config / Secrets
NEO4J_URI  = st.secrets["NEO4J_URI"]
NEO4J_USER = st.secrets["NEO4J_USERNAME"]       # updated to match your secret
NEO4J_PASS = st.secrets["NEO4J_PASS"]
DB_NAME    = st.secrets["NEO4J_DATABASE"]       # updated to match your secret
# ── Caching: driver, tokenizer/model, known drugs
@st.cache_resource(show_spinner=False)
def get_driver():
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

@st.cache_resource(show_spinner=False)
def get_llm():
    model_name = "microsoft/phi-2"  # always ON
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else None,
    )
    return tok, mdl

driver = get_driver()
tokenizer, model = get_llm()

# ── Lightweight, built‑in stopwords (all lowercase; avoid NLTK downloads)
STOPWORDS = {
    "a","an","and","are","as","at","be","by","for","from","has","he","in","is","it","its",
    "what","how","why","reason",  # <- fixed missing comma issue
    "of","on","that","the","to","was","were","will","with","you","your","yours","we","us",
    "can","i","when","what","when","how","can"
}

WORD_RE = re.compile(r"\b[a-zA-Z][a-zA-Z]+\b")

# ── Neo4j queries
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
            # soft spell-correct against known drugs
            m = get_close_matches(lw, known, n=1, cutoff=0.88)
            if m:
                found.append(m[0])
            else:
                unknown.append(w)
    # unique while preserving order
    seen=set(); found_u=[]
    for d in found:
        if d not in seen:
            found_u.append(d); seen.add(d)
    return found_u, unknown

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
    # Strict: paraphrase only; deterministic; no extra info
    text = (text or "").strip()
    if not text:
        return ""
    prompt = (
        "Rewrite the following in one short, patient-friendly sentence. "
        "Do not add information or name interaction types. "
        f'Text: "{text}"'
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    out = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    out = out.split(".")[0].strip()
    return (out + ".") if out else text

def all_pairs(drugs):
    n = len(drugs)
    for i in range(n):
        for j in range(i+1, n):
            yield drugs[i], drugs[j]

# ── UI
query = st.text_input(
    "Type your question (e.g., “Can I take Ibuprofen with Dexamethasone?”)",
    placeholder="Enter 2+ drug names in natural language…"
)
go = st.button("Check interactions", type="primary")

if go:
    drugs, unknown = extract_drugs(query)
    if unknown:
        st.warning("Unrecognized names (check spelling or try another): " + ", ".join(unknown))
    if len(drugs) < 2:
        st.info("Please include at least two valid drug names.")
    else:
        for d1, d2 in all_pairs(drugs):
            d1_info = get_drug_details(d1)
            d2_info = get_drug_details(d2)
            recs = get_interactions(d1, d2)

            st.subheader(f"🔎 {d1_info.get('Drug', d1)}  ×  {d2_info.get('Drug', d2)}")
            if recs:
                # combine multiple reasons/types
                reasons = [r.get("reason","").strip() for r in recs if r.get("reason")]
                reasons = [r for r in reasons if r]
                types   = sorted({(r.get("type") or "Unknown") for r in recs})
                combined_reason = " ".join(reasons).strip()

                # show verbatim (truth source)
                with st.expander("Raw reason(s) from Neo4j", expanded=False):
                    for i, rr in enumerate(reasons, 1):
                        st.write(f" Reason {i}:  {rr}")
                    st.caption("Source: r.reason on :INTERACTS_WITH")

                # LLM paraphrase (single sentence)
                paraphrase = paraphrase_reason_one_sentence(combined_reason) if combined_reason else ""
                if paraphrase:
                    st.success(paraphrase)
                else:
                    st.info("Interaction noted, but no reason text stored in the graph.")

                # meta
                st.write(f" Type(s): {', '.join(types)}")
            else:
                st.info("No known interaction found in the database.")

            # Optional compact details for each drug
            with st.expander(f"About {d1_info.get('Drug', d1)}"):
                if d1_info.get("Treats"): st.write(" Treats: " + ", ".join(d1_info["Treats"]))
                if d1_info.get("SideEffects"): st.write(" Side effects: " + ", ".join(d1_info["SideEffects"]))
                if d1_info.get("Warnings"): st.write(" Warnings: " + ", ".join(d1_info["Warnings"]))
                if d1_info.get("Precautions"): st.write(" Precautions: " + ", ".join(d1_info["Precautions"]))
            with st.expander(f"About {d2_info.get('Drug', d2)}"):
                if d2_info.get("Treats"): st.write(" Treats: " + ", ".join(d2_info["Treats"]))
                if d2_info.get("SideEffects"): st.write(" Side effects: " + ", ".join(d2_info["SideEffects"]))
                if d2_info.get("Warnings"): st.write(" Warnings: " + ", ".join(d2_info["Warnings"]))
                if d2_info.get("Precautions"): st.write(" Precautions: " + ", ".join(d2_info["Precautions"]))

        st.caption("Note: Results are limited to interactions present in your Neo4j graph. No writes are performed.")
