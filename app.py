# frontend_app.py

import streamlit as st
import requests

st.set_page_config(page_title="Literature Review Assistant", layout="wide")

st.title("🧠 Systematic Literature Review Assistant")

title = st.text_input("Enter your research topic:")
inclusion_criteria = st.text_area("Inclusion Criteria (comma-separated)").split(",")
exclusion_criteria = st.text_area("Exclusion Criteria (comma-separated)").split(",")

if st.button("Generate Review"):
    with st.spinner("Fetching and analyzing papers..."):
        response = requests.post(
            "http://127.0.0.1:8000/generate-review",
            json={
                "title": title,
                "inclusion_criteria": inclusion_criteria,
                "exclusion_criteria": exclusion_criteria
            }
        )
        if response.status_code == 200:
            data = response.json()
            st.subheader("📚 Literature Review Summary")
            st.write(data["review_summary"])

            st.subheader("📝 Individual Papers")
            for paper in data["papers"]:
                st.markdown(f"**{paper['title']}**")
                st.write(paper['abstract'])
                if "url" in paper:
                    st.markdown(f"[Read More]({paper['url']})")
                st.markdown("---")
        else:
            st.error("Failed to generate review. Check the backend logs.")

