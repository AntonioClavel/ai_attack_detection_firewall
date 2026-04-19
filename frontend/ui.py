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
                "NN Prediction": resp['nn_prediction'],
                "XGB Prediction": resp['xgb_prediction']
            })
            
            prog.progress((i + 1) / num_test)
            
        df_res = pd.DataFrame(results)
        
        st.subheader("Analysis Results")
        col_a, col_b = st.columns(2)
        
        with col_a:
            st.markdown("**Real Attacks Distribution**")
            
            df_pie = df_res.copy()
            counts = df_pie['Real Classification'].value_counts()
            df_pie['Legend Label'] = df_pie['Real Classification'].apply(lambda x: f"{x} ({counts[x]})")
            
            fig1 = px.pie(df_pie, names='Legend Label', hole=.3, color_discrete_sequence=px.colors.sequential.RdBu)
            
            st.plotly_chart(fig1, use_container_width=True)
            
        with col_b:
            st.markdown("**XGBoost Model Attack Prediction**")
            
            attack_data = df_res[df_res['XGB Prediction'] != 'N/A (Normal Traffic)']
            
            if not attack_data.empty:
                attack_counts = attack_data['XGB Prediction'].value_counts().reset_index()
                attack_counts.columns = ['Attack Type', 'Count']
                
                fig2 = px.bar(attack_counts, x='Attack Type', y='Count', 
                              color='Attack Type', text='Count')
                
                fig2.update_layout(showlegend=False)
                fig2.update_yaxes(dtick=1)
                
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No attacks detected by XGBoost in this sample.")

        st.subheader("Details of every line:")
        def color_rows(row):
            if row['NN Prediction'] == 'Attack Detected':
                return ['background-color: #fdb9b9'] * len(row)
            return [''] * len(row)

        st.dataframe(df_res.style.apply(color_rows, axis=1), use_container_width=True)
        
    except Exception as e:
        st.error(f"ERROR ({e})")