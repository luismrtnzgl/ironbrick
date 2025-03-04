import streamlit as st
import sklearn
import numpy
import joblib

st.write(f"🔹 scikit-learn en Streamlit Cloud: {sklearn.__version__}")
st.write(f"🔹 numpy en Streamlit Cloud: {numpy.__version__}")
st.write(f"🔹 joblib en Streamlit Cloud: {joblib.__version__}")
