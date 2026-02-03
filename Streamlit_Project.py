import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv(r"C:\Users\malbu\test\drug\app\drug200.csv")

st.set_page_config(page_title="Drug Dataset Dashboard", layout="wide")

# -----------------------------
# Title & Description
# -----------------------------
st.title("Drugs A, B, C, X, Y — Dataset Overview")

st.markdown("""
This dashboard provides an overview of the drug dataset used for 
Decision Tree and Random Forest classification.
""")

# -----------------------------
# Basic Info (Metrics)
# -----------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Number of Patients", len(df))
col2.metric("Min Age", df['Age'].min())
col3.metric("Max Age", df['Age'].max())

# -----------------------------
# Categorical Distributions
# -----------------------------
st.subheader("Categorical Distributions")

col4, col5, col6 = st.columns(3)

with col4:
    st.markdown("### Sex Distribution")
    st.write(df['Sex'].value_counts(normalize=True) * 100)

with col5:
    st.markdown("### Blood Pressure")
    st.write(df['BP'].value_counts(normalize=True) * 100)

with col6:
    st.markdown("### Cholesterol")
    st.write(df['Cholesterol'].value_counts(normalize=True) * 100)

# -----------------------------
# Numerical Distributions
# -----------------------------
st.subheader("Numerical Distributions")

col7, col8 = st.columns(2)

with col7:
    st.markdown("### Age Distribution")
    fig, ax = plt.subplots()
    ax.hist(df['Age'], bins=10)
    ax.set_xlabel("Age")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

with col8:
    st.markdown("### Na_to_K Distribution")
    fig, ax = plt.subplots()
    ax.hist(df['Na_to_K'], bins=10)
    ax.set_xlabel("Na_to_K")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

# -----------------------------
# Drug Distribution
# -----------------------------
st.subheader("Drug Distribution")

drug_counts = df['Drug'].value_counts()

fig, ax = plt.subplots()
ax.bar(drug_counts.index, drug_counts.values)
ax.set_xlabel("Drug")
ax.set_ylabel("Count")
st.pyplot(fig)

# -----------------------------
# Raw Data (Optional)
# -----------------------------
with st.expander("Show Raw Data"):
    st.dataframe(df)
