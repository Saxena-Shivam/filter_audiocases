import os
import streamlit as st
import pandas as pd
import numpy as np
import faiss
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import re
import json

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Audio Use Case Extractor", 
    page_icon="🎧",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        .header-style {
            font-size: 20px;
            font-weight: bold;
            color: #1f77b4;
            margin-top: 20px;
        }
        .highlight-box {
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 20px;
        }
        .term-chip {
            display: inline-block;
            background-color: #e1f5fe;
            padding: 3px 8px;
            border-radius: 15px;
            margin: 3px;
            font-size: 0.8em;
        }
        .result-item {
            padding: 12px 15px;
            border-left: 4px solid #4a90e2;
            margin: 8px 0;
            background-color: #ffffff;
            border-radius: 0 8px 8px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .result-item h4 {
            margin: 0 0 5px 0;
            color: #2c3e50;
        }
        .difficulty-sanity {
            color: #27ae60;
            font-weight: bold;
            background-color: #e8f5e9;
            padding: 2px 8px;
            border-radius: 12px;
            display: inline-block;
        }
        .difficulty-l2 {
            color: #f39c12;
            font-weight: bold;
            background-color: #fff3e0;
            padding: 2px 8px;
            border-radius: 12px;
            display: inline-block;
        }
        .difficulty-l4 {
            color: #e74c3c;
            font-weight: bold;
            background-color: #ffebee;
            padding: 2px 8px;
            border-radius: 12px;
            display: inline-block;
        }
        .matched-term {
            background-color: #d4edff;
            padding: 2px 6px;
            border-radius: 4px;
            font-weight: bold;
        }
        .score-badge {
            background-color: #e0e0e0;
            padding: 2px 8px;
            border-radius: 12px;
            font-family: monospace;
        }
    </style>
""", unsafe_allow_html=True)
# Main title with better formatting
st.title("🎯 Audio Use Case Extractor")
st.markdown("**GenAI-powered filtering for audio testing use cases**")
st.markdown("---")

# Short form dictionary (same as original)
short_form_dict = {
    "AD": "audio decode (playback)",
    "AE": "audio encode (record)",
    "FBSP": "speaker protection",
    "DMEF": "default media engine (audio playback/record)",
    "SSR": "sub system restart (a command which restarts DSP-related systems)",
    "PDR": "process domain restart (restarts the current audio process)",
    "FNN": "advanced noise suppression (VoIP/voice/record)",
    "SMECNS": "default noise suppression algorithm (VoIP/voice/record)",
    "Fluenece": "advanced noise suppression (VoIP/voice/record)",
    "BT": "Bluetooth",
    "QACT": "software to check the path the audio is taking",
    "DP_out": "external display use cases",
    "EcRef": "echo reference",
    "B2B": "back to back",
    "HS": "headset",
    "HTTP": "network streaming",
    "LDAC": "Bluetooth codec",
    "MicOcc": "mic occlusion",
    "LPI": "low power mode (only in SVA cases)",
    "NLPI": "non low power mode (only in SVA cases)",
    "APTX": "default codec for Bluetooth",
    "EDID": "extended display identification data (monitor configuration exchange)",
    "HDMI": "high-definition multimedia interface (audio/video output path)",
    "HPD": "hot plug detect (used in HDMI/DP connections)",
    "Device Switch": "audio routing change due to physical/logical device change",
    "RX": "receive path (e.g., incoming audio)",
    "TX": "transmit path (e.g., microphone transmission)",
    "DUT": "device under test",
    "VoIP": "voice over internet protocol (call-based audio)",
    "Alarm": "system sound alert or notification",
    "Notification": "system prompt sound or message"
}

short_forms = list(short_form_dict.keys())

# Score dictionary for difficulty (same as original)
score_dict = {
    "AD": 0.5,
    "AE": 1.0,
    "FBSP": 2.5,
    "DMEF": 1.5,
    "SSR": 1.0,
    "PDR": 1.0,
    "FNN": 1.5,
    "SMECNS": 1.5,
    "Fluenece": 1.5,
    "BT": 1.5,
    "QACT": 0.5,
    "DP_out": 1.5,
    "EcRef": 2.0,
    "B2B": 1.5,
    "HS": 1.0,
    "HTTP": 1.0,
    "LDAC": 1.5,
    "MicOcc": 2.5,
    "LPI": 2.0,
    "NLPI": 1.5,
    "APTX": 1.0,
    "EDID": 1.5,
    "HDMI": 1.5,
    "HPD": 1.5,
    "Device Switch": 1.5,
    "RX": 1.0,
    "TX": 1.0,
    "DUT": 0.5,
    "VoIP": 1.5,
    "Alarm": 1.0,
    "Notification": 1.0
}

# API keys section in sidebar
with st.sidebar:
    st.header("🔑 API Configuration")
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Required for advanced features"
    )
    st.markdown("---")
    st.header("ℹ️ About")
    st.markdown("""
    This tool helps you filter and analyze audio testing use cases based on:
    - Technical terms
    - Difficulty scoring
    - Custom criteria
    """)
    st.markdown("---")
    st.header("📚 Short Forms Legend")
    with st.expander("View all short forms"):
        for key, value in short_form_dict.items():
            st.markdown(f"**{key}**: {value}")

# File upload section
st.markdown("## 📂 Step 1: Upload Your Data")
uploaded_file = st.file_uploader(
    "Upload your Excel file with use cases",
    type=["xlsx"],
    help="File should contain a 'Name' column with use cases"
)

if uploaded_file:
    try:
        df = pd.read_excel(uploaded_file)
        if "Name" not in df.columns:
            st.error("❌ The Excel file must contain a 'Name' column.")
            st.stop()
            
        use_cases = df["Name"].dropna().astype(str).tolist()
        st.success(f"✅ Successfully loaded {len(use_cases)} use cases")
        
        # Show preview
        with st.expander("Preview first 10 use cases"):
            st.table(pd.DataFrame(use_cases[:10], columns=["Use Case"]))
            
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        st.stop()

# Filtering section
if uploaded_file and groq_api_key:
    st.markdown("## 🔍 Step 2: Filter Use Cases")
    
    with st.container():
        st.markdown("### Inclusion Criteria")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Must contain at least one of these:**")
            prefer_have = st.multiselect(
                "Preferred terms",
                short_forms,
                help="Use cases must contain AT LEAST ONE of these terms",
                key="prefer_have"
            )
            
        with col2:
            must_not_have_option=[term for term in short_forms if term not in prefer_have]
            st.markdown("#### 2️⃣ Exclude Terms")
            must_not_have = st.multiselect(
                "Select terms to exclude",
                must_not_have_option,
                help="Use cases containing ANY of these terms will be excluded",
                key="must_not_have"
            )
    
    # Advanced filtering
    with st.expander("⚙️ Advanced Filtering Options"):
        use_weights = st.checkbox("Enable weighted scoring", value=False)
        
        if use_weights:
            key_term_options = [term for term in short_forms if term in prefer_have and term not in must_not_have]
            key_terms = st.multiselect(
                "Key terms for weighted scoring",
                key_term_options,
                help="These terms will contribute to the difficulty score"
            )
            
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.number_input(
                "Number of results to show",
                min_value=1,
                max_value=500,
                value=50,
                step=1
            )
        with col2:
            difficulty_select = st.selectbox(
                "Filter by difficulty",
                ["Any", "sanity", "L2", "L4"]
            )
    
    # Search button with better styling
    if st.button("🔎 Search Use Cases", type="primary", use_container_width=True):
        if not prefer_have:
            st.warning("Please select at least one preferred term")
            st.stop()
            
        with st.spinner("Analyzing use cases..."):
            # Original filtering logic
            def normalize(text):
                return re.sub(r"[-_]", "", text.lower())

            def contains_any(term_list, text):
                norm_text = normalize(text)
                for term in term_list:
                    norm_term = normalize(term)
                    if re.search(re.escape(norm_term), norm_text, re.IGNORECASE):
                        return True
                return False

            def get_score(uc):
                score = 0
                for key, val in score_dict.items():
                    if key.lower() in uc.lower():
                        score += val
                return score

            def classify_difficulty(score):
                if score < 1:
                    return "sanity"
                elif score < 4:
                    return "L2"
                else:
                    return "L4"

            # Apply filters
            filtered = [uc for uc in use_cases if contains_any(prefer_have, uc)]
            
            if must_not_have:
                filtered = [uc for uc in filtered if not contains_any(must_not_have, uc)]
                
            if use_weights and key_terms:
                filtered = [uc for uc in filtered if contains_any(key_terms, uc)]
                
            if not filtered:
                st.warning("No use cases match your criteria")
                st.stop()
                
            # Score and sort
            results = []
            for uc in filtered:
                freq = sum(1 for term in prefer_have if term.lower() in uc.lower())
                score = get_score(uc)
                diff = classify_difficulty(score)
                results.append((uc, freq, score, diff))
                
            results.sort(key=lambda x: (-x[1], -x[2]))
            # ------------- Diverse Picking: Maximize Preferred Term Coverage -------------
            picked = []
            picked_set = set()
            term_idx = 0
            results_copy = results.copy()

            while len(picked) < top_k and results_copy:
                term = prefer_have[term_idx % len(prefer_have)]
                # Find the first result not yet picked that contains this term
                for i, (uc, freq, score, diff) in enumerate(results_copy):
                    if uc not in picked_set and term.lower() in uc.lower():
                        picked.append((uc, freq, score, diff))
                        picked_set.add(uc)
                        results_copy.pop(i)
                        break
                else:
                    # If no more use cases for this term, move to next term
                    pass
                term_idx += 1

            # If not enough picked, fill up with remaining highest scoring
            if len(picked) < top_k:
                for uc, freq, score, diff in results_copy:
                    if uc not in picked_set:
                        picked.append((uc, freq, score, diff))
                    if len(picked) >= top_k:
                        break
            # Apply difficulty filter
            if difficulty_select != "Any":
                results = [r for r in results if r[3] == difficulty_select]
            
            # Display results
            st.markdown(f"## 📊 Results ({len(results)} matched)")
            
            if not results:
                st.info("No use cases match all your criteria")
            else:
                # Summary stats
                col1, col2 = st.columns(2)
                col1.metric("Total Matched", len(results))
                col2.metric("Most Common Difficulty", 
                        max(set([r[3] for r in results]), key=[r[3] for r in results].count))

                for idx, (uc, freq, score, diff) in enumerate(picked):
                    highlighted_uc = uc
                    for term in prefer_have:
                        if term.lower() in uc.lower():
                            highlighted_uc = highlighted_uc.replace(term, f'<span class="matched-term">{term}</span>')
                    with st.container():
                        st.markdown(f"""
                        <div class="result-item">
                            <h4>{highlighted_uc}</h4>
                            <div style="display: flex; gap: 15px; align-items: center;">
                                <span class="difficulty-{diff.lower()}">{diff}</span>
                                <span>Matched terms: <strong>{freq}</strong></span>
                                <span class="score-badge">Score: {score:.1f}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                # Download option (use picked instead of results[:top_k])
                results_df = pd.DataFrame(picked, columns=["Use Case", "Matched Terms", "Score", "Difficulty"])
                st.download_button(
                    label="📥 Download Results",
                    data=results_df.to_csv(index=False),
                    file_name="filtered_use_cases.csv",
                    mime="text/csv"
                )
else:
    if not uploaded_file:
        st.info("Please upload an Excel file to begin")
    if not groq_api_key:
        st.info("Please enter your Groq API key in the sidebar")