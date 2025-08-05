import streamlit as st
from datetime import datetime
from SimpleBot import (
    extract_drugs_from_text, get_all_drug_pairs, get_interaction_from_neo4j,
    get_drug_details, simplify_interaction_reason, format_structured_output
)

st.set_page_config(page_title="💊 Healthcare Drug Interaction Chatbot", layout="centered")

# Title
st.markdown("<h2 style='text-align:center;'>💊 Healthcare - Drug Interaction Chatbot</h2>", unsafe_allow_html=True)

# Visible instruction message below title
st.markdown("""
<p style="
    text-align:center;
    font-size:18px;
    color:#222;
    background-color: #f0f4f8;
    border-radius: 8px;
    padding: 8px 15px;
    margin-bottom: 20px;
    font-weight: 600;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
">
Enter 2 or more drugs and seek simple advice!
</p>
""", unsafe_allow_html=True)

# Custom CSS for chat bubbles with icons
st.markdown("""
<style>
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 10px;
}
.user-message, .bot-message {
    display: flex;
    align-items: flex-start;
    max-width: 80%;
    padding: 10px 14px;
    border-radius: 12px;
    margin: 5px 0;
    word-wrap: break-word;
    white-space: pre-wrap;
}
.user-message {
    background-color: #dcf8c6;
    color: black;
    align-self: flex-end;
}
.bot-message {
    background-color: #f1f0f0;
    color: black;
    align-self: flex-start;
}
.user-icon {
    font-size: 24px;
    margin-right: 8px;
    margin-top: 2px;
}
.bot-icon {
    font-size: 24px;
    margin-right: 8px;
    margin-top: 2px;
}
.timestamp {
    font-size: 10px;
    color: #888;
    margin-top: 2px;
    margin-left: 40px;
}
.details-block {
    margin-top: 5px;
    margin-bottom: 15px;
    border-left: 3px solid #ccc;
    padding-left: 10px;
    font-family: monospace;
    font-size: 14px;
    white-space: pre-wrap;
    max-width: 80%;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "details" not in st.session_state:
    st.session_state.details = []

user_input = st.chat_input("Ask about 2 or more drugs...")

if user_input:
    now = datetime.now().strftime("%H:%M")
    st.session_state.messages.append({"role": "user", "content": user_input, "time": now})

    with st.spinner("🔎 Analyzing your question... Please wait"):
        try:
            drug_names, unrecognized = extract_drugs_from_text(user_input)
            if len(drug_names) < 2:
                if unrecognized:
                    response = f"⚠️ Unrecognized drug name(s): {', '.join(unrecognized)}. Please check spelling."
                else:
                    response = "❗ Please mention at least two valid drug names."
                st.session_state.messages.append({"role": "bot", "content": response, "time": now})
                st.session_state.details = []
            else:
                responses = []
                details = []
                seen_pairs = set()

                for d1, d2 in get_all_drug_pairs(drug_names):
                    pair = tuple(sorted([d1, d2]))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)

                    results = get_interaction_from_neo4j(d1, d2)
                    d1_info = get_drug_details(d1)
                    d2_info = get_drug_details(d2)

                    if results:
                        reasons = {r.get("reason", "unknown") for r in results}
                        types = {r.get("type", "Unknown") for r in results}
                        reason_text = ". ".join(reasons)
                        type_text = ", ".join(types)
                        brief_resp = simplify_interaction_reason(d1_info["Drug"], d2_info["Drug"], reason_text, type_text)
                        responses.append(brief_resp)

                        detail_text = (
                            f"💊 Drug 1: {d1_info['Drug']}\n"
                            f"   • Treats: {', '.join(d1_info.get('Treats', []))}\n"
                            f"   • Side Effects: {', '.join(d1_info.get('SideEffects', []))}\n"
                            f"   • Warnings: {', '.join(d1_info.get('Warnings', []))}\n\n"
                            f"💊 Drug 2: {d2_info['Drug']}\n"
                            f"   • Treats: {', '.join(d2_info.get('Treats', []))}\n"
                            f"   • Side Effects: {', '.join(d2_info.get('SideEffects', []))}\n"
                            f"   • Warnings: {', '.join(d2_info.get('Warnings', []))}\n\n"
                            f"⚠️ Interaction:\n"
                            f"   • Type: {type_text}\n"
                            f"   • Reason: {reason_text}\n"
                            f"   • Recommendation: Consult your doctor if frequent use is needed."
                        )
                        details.append(detail_text)
                    else:
                        no_int_msg = f"✅ No known interaction between {d1_info['Drug']} and {d2_info['Drug']}."
                        responses.append(no_int_msg)
                        detail_text = (
                            f"💊 Drug 1: {d1_info['Drug']}\n"
                            f"   • Treats: {', '.join(d1_info.get('Treats', []))}\n"
                            f"   • Side Effects: {', '.join(d1_info.get('SideEffects', []))}\n"
                            f"   • Warnings: {', '.join(d1_info.get('Warnings', []))}\n\n"
                            f"💊 Drug 2: {d2_info['Drug']}\n"
                            f"   • Treats: {', '.join(d2_info.get('Treats', []))}\n"
                            f"   • Side Effects: {', '.join(d2_info.get('SideEffects', []))}\n"
                            f"   • Warnings: {', '.join(d2_info.get('Warnings', []))}\n\n"
                            f"✅ Interaction:\n"
                            f"   • No known interaction found.\n"
                            f"   • Recommendation: Generally safe under medical guidance."
                        )
                        details.append(detail_text)

                final_response = "\n\n".join(f"🔷 {r}" for r in responses)
                st.session_state.messages.append({"role": "bot", "content": final_response, "time": now})
                st.session_state.details = details
        except Exception as e:
            error_msg = f"❌ An error occurred: {e}"
            st.session_state.messages.append({"role": "bot", "content": error_msg, "time": now})
            st.session_state.details = []

# Show chat history
st.write("---")
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for msg in st.session_state.messages:
    role_class = "user-message" if msg['role'] == "user" else "bot-message"
    icon = "🧑‍⚕️" if msg['role'] == "user" else "🤖"
    st.markdown(f"<div class='{role_class}'><span class='user-icon'>{icon}</span>{msg['content']}<div class='timestamp'>{msg['time']}</div></div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# Show detailed node info on demand
if st.session_state.messages and st.button("📄 Show Last Interaction Detailed Info"):
    if st.session_state.details:
        for d in st.session_state.details:
            st.markdown(f"<div class='details-block'>{d}</div>", unsafe_allow_html=True)
