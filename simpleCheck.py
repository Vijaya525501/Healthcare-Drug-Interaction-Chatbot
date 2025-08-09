import streamlit as st
from neo4j import GraphDatabase

st.title("✅ Neo4j Aura Connection Test")

# Credentials from Streamlit secrets
uri = st.secrets["neo4j+s://5acfeed9.databases.neo4j.io"]
user = st.secrets["neo4j"]
password = st.secrets["KnNt42Z-f5uWvfRLRLByYKq758nubecZNkYafbaCf1I"]

driver = GraphDatabase.driver(uri, auth=(user, password))

try:
    with driver.session() as session:
        msg = session.run("RETURN 'Aura OK' AS msg").single()["msg"]
        count = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]

    st.success(f"🟢 Connected to Neo4j Aura: {msg}")
    st.write(f"Total nodes in database: {count}")

except Exception as e:
    st.error(f"🔴 Connection failed: {e}")
