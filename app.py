import streamlit as st

st.title("Environment Test")

# Test 1: Check for the requirements file
import os
if os.path.exists("requirements.txt"):
    st.success("requirements.txt exists in the root folder!")
    with open("requirements.txt", "r") as f:
        st.code(f.read(), language="text")
else:
    st.error("requirements.txt is MISSING from the root folder!")

# Test 2: Try to import the libraries one by one to find the killer
st.write("### Library Checks:")
libs = ["pandas", "matplotlib", "scipy", "numpy"]

for lib in libs:
    try:
        __import__(lib)
        st.write(f"✅ {lib} is installed.")
    except ImportError:
        st.write(f"❌ {lib} is NOT installed.")

# Test 3: Minimalist Logic (No Matplotlib needed)
st.write("### Data Input Test")
text_input = st.text_area("Paste some text here to see if the app is alive:")
if text_input:
    words = text_input.lower().split()
    st.write(f"Word count: {len(words)}")
