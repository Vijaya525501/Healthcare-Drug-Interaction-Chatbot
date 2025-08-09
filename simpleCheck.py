import streamlit as st
from neo4j import GraphDatabase

st.title("✅ Neo4j Aura Connection Test")

# Read credentials from Streamlit secrets
uri = st.secrets["NEO4J_URI"]
user = st.secrets["NEO4J_USER"]
password = st.secrets["NEO4J_PASS"]

driver = GraphDatabase.driver(uri, auth=(user, password))

try:
    with driver.session() as session:
        msg = session.run("RETURN 'Aura OK' AS msg").single()["msg"]
        count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]

    st.success(f"🟢 Connected to Neo4j Aura: {msg}")
    st.write(f"Total nodes in database: {count}")

except Exception as e:
    st.error(f"🔴 Connection failed: {e}")
