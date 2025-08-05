from neo4j import GraphDatabase
from itertools import combinations
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import re
from difflib import get_close_matches
from rake_nltk import Rake
import nltk
from nltk.corpus import stopwords

# Ensure NLTK resources
for resource in ['stopwords', 'punkt', 'punkt_tab']:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource)

# =========================================================
#  MediBot - Cleaned & Deduplicated Drug Interaction Checker
# =========================================================

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASS = "Vijayalaxmi@18"
DB_NAME = "chat"
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASS))

# Load Phi-2 LLM
model_name = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------

def flatten_and_unique(nested_list):
    """Flatten any nested list and remove duplicates while preserving order."""
    flat = []
    for item in nested_list:
        if isinstance(item, list):
            flat.extend(flatten_and_unique(item))
        elif item:
            flat.append(str(item).strip())
    seen, unique_flat = set(), []
    for x in flat:
        if x not in seen:
            unique_flat.append(x)
            seen.add(x)
    return unique_flat

def get_known_drug_names():
    query = "MATCH (d:Drug) RETURN toLower(d.drug_name) AS name"
    with driver.session(database=DB_NAME) as session:
        result = session.run(query)
        return set(record["name"] for record in result)

def correct_spelling(word, known_drugs):
    matches = get_close_matches(word.lower(), known_drugs, n=1, cutoff=0.8)
    return matches[0] if matches else None

stop_words = set(stopwords.words('english'))

def extract_drugs_from_text(user_input):
    known_drugs = get_known_drug_names()
    words = re.findall(r'\b[a-zA-Z][a-zA-Z]+\b', user_input)
    found, unknown = [], []
    for word in words:
        lw = word.lower()
        # Skip common English stopwords to avoid false unknowns
        if lw in stop_words:
            continue
        if lw in known_drugs:
            found.append(lw)
        else:
            corrected = correct_spelling(word, known_drugs)
            if corrected:
                found.append(corrected)
            else:
                unknown.append(word)
    return found, unknown

def auto_clean_text(text: str) -> str:
    """Remove junk chars and compress spaces."""
    if not text:
        return ""
    text = re.sub(r"[^a-zA-Z0-9 ,.;]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def summarize_warning(warnings):
    """Take only the first sentence for display purposes."""
    summarized = []
    for w in warnings:
        sentence = str(w).split(".")[0]
        summarized.append(sentence.strip())
    return flatten_and_unique(summarized)

def simplify_interaction_reason(drug1, drug2, reason, itype):
    """Generate one short, meaningful patient-friendly sentence without duplicates."""
    reason = auto_clean_text(reason)
    rake = Rake()
    rake.extract_keywords_from_text(reason)
    scored_keywords = rake.get_ranked_phrases_with_scores()

    meaningful_risks = []
    seen = set()
    for score, kw in scored_keywords:
        kw_clean = kw.strip().capitalize()
        kw_lower = kw_clean.lower()
        if any(skip in kw_lower for skip in [
            drug1.lower(), drug2.lower(), "drug", "treat", "symptom",
            "precaution", "warning", "cause", "therapy", "share risk"
        ]):
            continue
        if kw_clean not in seen and len(kw_clean.split()) >= 1:
            seen.add(kw_clean)
            meaningful_risks.append(kw_clean)

    main_risks = meaningful_risks[:2]
    risk_text = ", ".join(main_risks) if main_risks else "health issues"

    prompt = (
        f"Explain for a patient in one short sentence: "
        f"Taking {drug1} and {drug2} together may increase the risk of {risk_text}. "
        f"Do not mention interaction types or technical terms."
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)
    explanation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    final_sentence = explanation.split(".")[0].strip()
    return final_sentence + "."

def get_drug_details(drug):
    """Fetch all connected info for a drug, deduplicated for readability."""
    query = """
    MATCH (d:Drug)
    WHERE toLower(d.drug_name) = $drug
    OPTIONAL MATCH (d)-[:TREATS]->(c:Condition)
    OPTIONAL MATCH (d)-[:HAS_SIDE_EFFECT]->(s:SideEffect)
    OPTIONAL MATCH (d)-[:HAS_WARNING]->(w:Warning)
    OPTIONAL MATCH (d)-[:HAS_PRECAUTION]->(p:Precaution)
    OPTIONAL MATCH (d)-[:CAUSES]->(ca:Cause)
    RETURN d.drug_name AS Drug,
           collect(DISTINCT c.condition_name) AS Treats,
           collect(DISTINCT s.sideeffect_name) AS SideEffects,
           collect(DISTINCT w.warning_name) AS Warnings,
           collect(DISTINCT p.precaution_name) AS Precautions,
           collect(DISTINCT ca.cause_name) AS Causes
    """
    with driver.session(database=DB_NAME) as session:
        rec = session.run(query, drug=drug.lower()).single()
        if rec:
            return {
                "Drug": rec["Drug"],
                "Treats": flatten_and_unique(rec["Treats"]),
                "SideEffects": flatten_and_unique(rec["SideEffects"]),
                "Warnings": flatten_and_unique(rec["Warnings"]),
                "Precautions": flatten_and_unique(rec["Precautions"]),
                "Causes": flatten_and_unique(rec["Causes"]),
            }
        return {}

def get_interaction_from_neo4j(drug1, drug2):
    query = """
    MATCH (d1:Drug)-[r:INTERACTS_WITH]-(d2:Drug)
    WHERE (toLower(d1.drug_name) = $drug1 AND toLower(d2.drug_name) = $drug2)
       OR (toLower(d1.drug_name) = $drug2 AND toLower(d2.drug_name) = $drug1)
    RETURN d1.drug_name AS Drug1, d2.drug_name AS Drug2, 
           r.reason AS reason, r.type AS type
    """
    with driver.session(database=DB_NAME) as session:
        return session.run(query, drug1=drug1, drug2=drug2).data()

def get_all_drug_pairs(drugs):
    return list(combinations(drugs, 2))

def format_structured_output(drug1_info, drug2_info, interaction_info):
    def format_drug_block(drug_info, index):
        lines = [f"🔹 Drug {index}: {drug_info['Drug']}"]
        if drug_info.get('Treats'): lines.append(f"   • Treats: {', '.join(drug_info['Treats'])}")
        if drug_info.get('SideEffects'): lines.append(f"   • Side Effects: {', '.join(drug_info['SideEffects'])}")
        if drug_info.get('Warnings'): lines.append(f"   • Warnings: {', '.join(drug_info['Warnings'])}")
        if drug_info.get('Precautions'): lines.append(f"   • Precautions: {', '.join(drug_info['Precautions'])}")
        if drug_info.get('Causes'): lines.append(f"   • Causes: {', '.join(drug_info['Causes'])}")
        return "\n".join(lines)

    lines = [
        format_drug_block(drug1_info, 1),
        "",
        format_drug_block(drug2_info, 2),
        ""
    ]

    if interaction_info:
        lines.append("🔹 Interaction")
        lines.append(f"   • Type: {interaction_info['type']}")
        lines.append(f"   • Reason: {interaction_info['reason']}")
        lines.append("   • Recommendation: Consult your doctor if frequent use is needed.")
    else:
        lines.append("🔹 Interaction")
        lines.append("   • No known interaction found.")
        lines.append("   • Recommendation: Generally safe under medical guidance.")

    return "\n".join(lines)

# ---------------------------------------------------------
# Main Console Bot
# ---------------------------------------------------------
def start_medibot():
    print("\n💊 Healthcare Chatbot - Drug Interaction Checker")
    print("\nType 'exit' to quit.\n")
    while True:
        user_input = input("Enter drug interaction question: ").strip()
        if user_input.lower() == "exit":
            print(" Goodbye!")
            break

        drug_names, unknowns = extract_drugs_from_text(user_input)

        if len(drug_names) < 2:
            if unknowns:
                print(f" ❗ Unrecognized drug name(s): {', '.join(unknowns)}. Please check spelling.\n")
            else:
                print(" ❗ Please mention at least two valid drug names.\n")
            continue

        seen_pairs = set()

        for d1, d2 in get_all_drug_pairs(drug_names):
            pair = tuple(sorted([d1, d2]))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)

            d1_info = get_drug_details(d1)
            d2_info = get_drug_details(d2)
            interactions = get_interaction_from_neo4j(d1, d2)

            if interactions:
                reasons = [auto_clean_text(r.get("reason", "")) for r in interactions]
                types = {r.get("type", "Unknown") for r in interactions}

                interaction_info = {
                    "type": ", ".join(types),
                    "reason": ". ".join(reasons)
                }

                print(format_structured_output(d1_info, d2_info, interaction_info))
                print("\n Brief LLM Response:")
                print(simplify_interaction_reason(
                    d1_info["Drug"], d2_info["Drug"], interaction_info['reason'], interaction_info['type']
                ))

            else:
                print(format_structured_output(d1_info, d2_info, None))
                print("\n Brief LLM Response:")
                print(f"{d1_info['Drug']} and {d2_info['Drug']} have no known interaction. Generally safe under medical guidance.")

            print("\n" + "-"*80 + "\n")

if __name__ == "__main__":
    start_medibot()
