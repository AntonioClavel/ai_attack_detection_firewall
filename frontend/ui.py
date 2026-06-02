import streamlit as st
import pandas as pd
import requests
import time
import plotly.express as px

st.set_page_config(page_title="IA Network Firewall", layout="wide")

API_URL = "http://api:8000/predict"

st.title("AI Firewall Testing Page")
st.write("Launch a number of desired attacks randomly selected from the CIC-UNSW-NB15 dataset to test the AI firewall")

st.sidebar.header("Testing Parameters")
num_test = st.sidebar.number_input("Number of connections to analyze", 5, 200, 20)

if st.sidebar.button("Launch test"):
    try:
        df_data = pd.read_csv("Dataset/Data.csv")
        df_label = pd.read_csv("Dataset/Label.csv")
        
        mapping_real = {"0": 'Normal/Benign', "1": 'Analysis', "2": 'Backdoor', "3": 'DoS', 
                        "4": 'Exploits', "5": 'Fuzzers', "6": 'Generic', "7": 'Reconnaissance', 
                        "8": 'Shellcode', "9": 'Worms'}

        indices = df_data.sample(num_test).index
        results = []
        
        prog = st.progress(0)
        
        for i, idx in enumerate(indices):
            row_data = df_data.loc[idx].to_dict()
            real_label = mapping_real.get(str(df_label.iloc[idx]['Label']))
            
            resp = requests.post(API_URL, json=row_data).json()
            
            results.append({
                "Row": idx,
                "Real Classification": real_label,
                "AI Prediction": resp['prediction']
            })
            
            prog.progress((i + 1) / num_test)
            
        df_res = pd.DataFrame(results)
        
        st.subheader("Analysis Results")
        
        st.markdown("**Real Attacks Distribution sent to the Firewall**")
        
        df_pie = df_res.copy()
        counts = df_pie['Real Classification'].value_counts()
        df_pie['Legend Label'] = df_pie['Real Classification'].apply(lambda x: f"{x} ({counts[x]})")
        
        fig1 = px.pie(df_pie, names='Legend Label', hole=.3, color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig1, use_container_width=True)
        
        st.subheader("Details of every line:")
        
        def color_rows(row):
            if row['AI Prediction'] == 'Attack Detected':
                return ['background-color: #fdb9b9'] * len(row)
            return [''] * len(row)

        st.dataframe(df_res.style.apply(color_rows, axis=1), use_container_width=True)
        
    except Exception as e:
        st.error(f"ERROR ({e})")