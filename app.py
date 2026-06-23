import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set page layout to wide for dashboard look
st.set_page_config(layout="wide")

st.title("🚨 Live Trade Fraud Simulation: The E-Waste Smuggler Game")
st.markdown("""
    **Classroom Context:** Students act as 'bad actors' trying to evade international environmental treaties by 
    disguising hazardous e-waste shipments under high-value machinery, electronics, and specialty equipment HS codes.
""")

# --- Highly Complex Legitimate E-Waste Baseline Data Reference ---
# Deep catalog reflecting the paper's core data variables
baseline_prices = {
    850211: 7.50,  # Electric Generating Sets (The Zero-Duty Loophole anomaly)
    847130: 12.50, # Data Processing Units / Obsolete Computers
    847290: 8.50,  # Other Office Machines
    850440: 9.20,  # Electrical Transformers & Power Inverters
    854231: 35.00, # Electronic Integrated Circuits (High-Value Tech Baseline)
    810196: 45.00, # Tungsten Specialty Metal Alloys
    901890: 30.00, # Advanced Medical Instruments
    720410: 1.20,  # Ferrous Waste and Scrap Baseline
    840410: 95.00, # Auxiliary Plant for Boilers
    847010: 150.00,# Electronic Accounting Calculators
    847611: 110.00,# Automatic Goods-Vending Machines
    854911: 1.10   # Official E-Waste Commodity Code (Scrap Control Baseline)
}

hs_descriptions = {
    850211: "Electric Generating Sets (0% Duty Loophole)",
    847130: "Data Processing Units & Computers",
    847290: "Functional Office Machinery Items",
    850440: "Electrical Transformers & Inductors",
    854231: "Integrated Circuit Processors",
    810196: "Specialty Tungsten Rare Metal Alloys",
    901890: "Advanced Medical Instruments",
    720410: "Ferrous Scrap Metal (5% Base Tariff)",
    840410: "Industrial Auxiliary Boiler Plants",
    847010: "Electronic Accounting Systems",
    847611: "Automatic Goods-Vending Sub-systems",
    854911: "Discards: Spent Batteries & Electronic Waste"
}

# --- GLOBAL SHARED DATABASE MANAGER (Bridges all student devices together) ---
class GlobalDataManager:
    def __init__(self):
        self.df = pd.DataFrame(columns=[
            "Trader Name", "HS Code", "Weight (KG)", "Declared Price ($/KG)"
        ])
    
    def add_submission(self, row):
        self.df = pd.concat([self.df, pd.DataFrame([row])], ignore_index=True)
        
    def clear_database(self):
        self.df = pd.DataFrame(columns=[
            "Trader Name", "HS Code", "Weight (KG)", "Declared Price ($/KG)"
        ])

@st.cache_resource
def get_global_database():
    return GlobalDataManager()

# Initialize global shared cloud database connection
db = get_global_database()

# Initialize local session state for the reveal mechanism toggle
if "reveal_results" not in st.session_state:
    st.session_state.reveal_results = False

# Layout Split: 1/3 Student Form, 2/3 Teacher Dashboard
col1, col2 = st.columns([1, 2])

# --- COLUMN 1: STUDENT INPUT INTERFACE & QR CODE ---
with col1:
    st.header("📥 Student Submission Portal")
    
    st.markdown("### 📲 Scan to Join the Game Live!")
    # NEW UPDATED TARGET LINK EMBEDDED DYNAMICALLY
    app_url = "https://ewaste-simulation-qr4hzdvxxcvbbepw7sztvt.streamlit.app/"
    qr_api_url = f"https://api.qrserver.com/v1/create-qr-code/?size=200x200&data={app_url}"
    st.image(qr_api_url, caption="Scan with your phone camera to play", width=200)
    st.markdown("---")
    
    with st.form(key="student_form", clear_on_submit=True):
        student_name = st.text_input("Enter your Smuggler Alias / Name:", placeholder="e.g., Phantom Recycler")
        
        hs_code = st.selectbox(
            "Select Destination HS Code to misclassify your e-waste:",
            options=list(baseline_prices.keys()),
            format_func=lambda x: f"HS {x} ({hs_descriptions[x]})"
        )
        
        weight = st.selectbox(
            "Select Shipment Weight (Volume):",
            options=[10000, 25000, 100000],
            format_func=lambda x: f"{x:,} KG"
        )
        
        price = st.selectbox(
            "Select Declared Value (Unit Price):",
            options=[0.50, 1.80, 4.50],
            format_func=lambda x: f"${x:.2f} / KG"
        )
        
        submit_button = st.form_submit_button(label="🚀 Submit Manifest to Border Customs")
        
        if submit_button:
            if student_name.strip() == "":
                st.error("Please enter a name or alias before submitting!")
            else:
                new_row = {
                    "Trader Name": student_name,
                    "HS Code": hs_code,
                    "Weight (KG)": weight,
                    "Declared Price ($/KG)": price
                }
                db.add_submission(new_row)
                st.success(f"Manifest successfully submitted for {student_name}!")

# --- COLUMN 2: LIVE TEACHER MACHINE LEARNING DASHBOARD ---
with col2:
    st.header("🖥️ Live Customs ML Control Panel")
    
    # Action Deck Buttons
    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        if st.button("🔄 Sync Incoming Submissions", type="primary", use_container_width=True):
            st.rerun()
    with c_btn2:
        if st.session_state.reveal_results:
            if st.button("🔒 Hide Results From Class", use_container_width=True):
                st.session_state.reveal_results = False
                st.rerun()
        else:
            if st.button("🔓 Reveal Results to Class", use_container_width=True):
                st.session_state.reveal_results = True
                st.rerun()

    df = db.df.copy()
    total_manifests = len(df)
    
    # Core calculations run silently in background
    if not df.empty:
        df["Baseline Price"] = df["HS Code"].map(baseline_prices)
        df["Price Deficit"] = df["Baseline Price"] - df["Declared Price ($/KG)"]
        
        def define_profile(row):
            base_p = baseline_prices.get(int(row["HS Code"]), 10.0)
            deficit = base_p - float(row["Declared Price ($/KG)"])
            if int(row["Weight (KG)"]) == 100000 and deficit > 5.0:
                return "🚨 CRITICAL RISK: Mass E-Waste Laundering"
            elif deficit <= 0.0:
                return "✅ LOW RISK: Genuine High-Value Capital Asset"
            elif int(row["Weight (KG)"]) == 10000 and deficit > 3.0:
                return "⚠️ MEDIUM RISK: Tactical Loophole Smuggling"
            else:
                return "🔍 MODERATE RISK: Suspicious Value Discrepancy"
        
        df["Risk Profile"] = df.apply(define_profile, axis=1)

    # STATE A: RESULTS HIDDEN (Teacher panel collection lock)
    if not st.session_state.reveal_results:
        st.info(f"📥 Neutral Data Collection Mode Active. Total e-waste manifests logged silently: **{total_manifests}**.")
        
    # STATE B: REVEAL UNLOCKED (Live interactive reveal)
    else:
        if not df.empty:
            # Multi-HS Code Metric Ranking Panel
            st.subheader("📊 Active Electronic Code Deviations")
            
            grouped_deficits = df.groupby("HS Code")["Price Deficit"].mean().sort_values(ascending=False)
            
            metric_cols = st.columns(min(len(grouped_deficits), 4))
            for idx, (code, avg_def) in enumerate(grouped_deficits.items()):
                if idx < len(metric_cols):
                    metric_cols[idx].metric(
                        label=f"Exploited HS {code}", 
                        value=f"${avg_def:.2f} Deficit",
                        delta="Abnormal Valuation", 
                        delta_color="inverse"
                    )
            
            # --- REAL-TIME MULTI-CODE PRICE VISUALIZATION ---
            st.subheader("📈 Declared Hardware Prices vs. True Market Valuations")
            
            chart_data = []
            for code in baseline_prices.keys():
                sub_df = df[df["HS Code"] == code]
                avg_dec = sub_df["Declared Price ($/KG)"].mean() if not sub_df.empty else 0
                chart_data.append({
                    "HS Code": f"HS {code}",
                    "Students Declared Avg": avg_dec,
                    "True Market Price": baseline_prices[code]
                })
            
            chart_df = pd.DataFrame(chart_data).set_index("HS Code")
            st.bar_chart(chart_df, use_container_width=True)
            
            # --- FIXED NATIVE MARKDOWN AUTOMATED REGULATORY COMMENTARY BLOCK ---
            st.subheader("🤖 Automated E-Waste Intelligence Reports")
            
            has_alerts = False
            for code in baseline_prices.keys():
                sub_df = df[df["HS Code"] == code]
                if not sub_df.empty:
                    avg_def = sub_df["Price Deficit"].mean()
                    total_vol = sub_df["Weight (KG)"].sum()
                    exploiters_count = len(sub_df)
                    
                    if avg_def > 5.0 and total_vol >= 100000:
                        has_alerts = True
                        st.error(f"""
                        **🔴 CRITICAL ANOMALY DETECTED IN HARDWARE CODE: HS {code} ({hs_descriptions[code]})**
                        * **Fraud Profile:** Mass Hazardous Scrap Infiltration.
                        * **Intelligence Analysis:** {exploiters_count} individual traders have aggregated a total of {total_vol:,} KG under this sector, while dropping the unit valuation down to scrap metrics. This tracks the inverse price-volume waste signature where falling values run parallel to volume spikes, exposing calculated evasion of trade restrictions.
                        """)
                    
                    elif code == 850211 and avg_def > 2.0:
                        has_alerts = True
                        st.error(f"""
                        **🔴 SMOKING GUN TAX EVASION: HS {code} ({hs_descriptions[code]})**
                        * **Fraud Profile:** Zero-Tariff Loophole Exploitation.
                        * **Intelligence Analysis:** Submissions indicate heavy volume integration inside the Electric Generator category at near-zero scrap pricing. Because generators carry an optimized 0% import duty rate compared to standard regulatory penalties on raw scrap inputs, bad actors are treating this finished good code as a tariff shield.
                        """)
                        
                    elif avg_def > 3.0:
                        has_alerts = True
                        st.warning(f"""
                        **🟡 SUSPICIOUS VALUE MATCH MANIPULATION: HS {code} ({hs_descriptions[code]})**
                        * **Fraud Profile:** Hidden Component Extraction Strategy.
                        * **Intelligence Analysis:** Unit pricing data shows persistent deflation compared to prime retail benchmarks. Importers are running sub-scale volumes designed to slide below automated physical inspection triggers, indicating localized laundering of circuit assemblies.
                        """)
            
            if not has_alerts:
                st.success("🟢 **System Clean:** No definitive electronic contamination signatures detected across tracked HS channels yet.")
                
            # --- FULL LIVE RISK TABLE WATCHLIST ---
            st.subheader("🕵️ Live Risk Watchlist (Real-time Analysis)")
            display_df = df[["Trader Name", "HS Code", "Weight (KG)", "Declared Price ($/KG)", "Risk Profile"]]
            
            def highlight_risk(val):
                if "CRITICAL" in str(val): return 'background-color: #ffcccc; color: black; font-weight: bold;'
                elif "MEDIUM" in str(val): return 'background-color: #fff2cc; color: black;'
                elif "LOW" in str(val): return 'background-color: #e2efda; color: black;'
                return ''
                
            st.dataframe(
                display_df.style.map(highlight_risk, subset=["Risk Profile"]),
                use_container_width=True
            )
        else:
            st.info("The database is currently neutral. Awaiting student submissions before analysis can be unlocked.")

# --- CAMOUFLAGED TEACHER RESET KEY (Hidden inside the dot expander) ---
st.markdown("<br><br><br><br><br>", unsafe_allow_html=True)
with st.expander("·", expanded=False):
    if st.button("🗑️ Reset All Global Data"):
        db.clear_database()
        st.session_state.reveal_results = False
        st.rerun()
