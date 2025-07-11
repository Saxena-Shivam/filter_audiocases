import os
import streamlit as st
import pandas as pd
import numpy as np
import random
from dotenv import load_dotenv
import re
import json
EXCEL_PATH = "d:/Project-ARC/Question_Paper_Generator/audio/usecases.xlsx"  # <-- set your file path here

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
if uploaded_file:
    st.markdown("## 🔍 Step 2: Filter Use Cases")
    must_not_have = []
    with st.container():
        st.markdown("### Inclusion Criteria")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 1️⃣ Must contain at least one of these:")
            prefer_have = st.multiselect(
                "Preferred terms",
                short_forms,
                help="Use cases must contain AT LEAST ONE of these terms",
                key="prefer_have"
            )
        with col2:
            st.markdown("#### 2️⃣ Exclude Terms")
            must_not_have = st.multiselect(
                "Select terms to exclude",
                short_forms,
                help="Use cases containing ANY of these terms will be excluded",
                key="must_not_have"
            )
        conflict_prefer_exclude = set(prefer_have) & set(must_not_have)
        if conflict_prefer_exclude:
            st.warning(f"⚠️ The following terms are selected in BOTH Preferred and Exclude: {', '.join(conflict_prefer_exclude)}. Please resolve the conflict.")
            st.stop()

    # Advanced filtering
    with st.expander("⚙️ Advanced Filtering Options"):
        use_weights = st.checkbox("Enable weighted scoring", value=False)
        key_terms = []
        if use_weights:
            key_term_options = [term for term in prefer_have if term not in must_not_have]
            key_terms = st.multiselect(
                "Key terms for weighted scoring",
                key_term_options,
                help="These terms will contribute to the difficulty score"
            )
    col1, col2, col3 = st.columns(3)
    with col1:
        num_sanity = st.number_input("Number of 'sanity' use cases", min_value=0, max_value=500, value=0, step=1)
    with col2:
        num_l2 = st.number_input("Number of 'L2' use cases", min_value=0, max_value=500, value=0, step=1)
    with col3:
        num_l4 = st.number_input("Number of 'L4' use cases", min_value=0, max_value=500, value=0, step=1)

    if st.button("🔎 Search Use Cases", type="primary", use_container_width=True):
        if not prefer_have:
            st.warning("Please select at least one preferred term")
            st.stop()

        with st.spinner("Analyzing use cases..."):
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
                if score < 2:
                    return "sanity"
                elif score < 5:
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

            # --- Pick for each difficulty ---
            def pick_for_difficulty(results, prefer_have, count, difficulty):
                picked = []
                picked_set = set()
                term_idx = 0
                # Filter by difficulty
                results_copy = [r for r in results if r[3] == difficulty]
                # Shuffle for randomness
                random.shuffle(results_copy)
                while len(picked) < count and results_copy:
                    term = prefer_have[term_idx % len(prefer_have)]
                    for i, (uc, freq, score, diff) in enumerate(results_copy):
                        if uc not in picked_set and term.lower() in uc.lower():
                            picked.append((uc, freq, score, diff))
                            picked_set.add(uc)
                            results_copy.pop(i)
                            break
                    else:
                        pass
                    term_idx += 1
                # If not enough, fill up with remaining (still random order)
                if len(picked) < count:
                    for uc, freq, score, diff in results_copy:
                        if uc not in picked_set:
                            picked.append((uc, freq, score, diff))
                        if len(picked) >= count:
                            break
                return picked

            picked_sanity = pick_for_difficulty(results, prefer_have, num_sanity, "sanity")
            picked_l2 = pick_for_difficulty(results, prefer_have, num_l2, "L2")
            picked_l4 = pick_for_difficulty(results, prefer_have, num_l4, "L4")
            final_picked = picked_sanity + picked_l2 + picked_l4

            # Display results
            st.markdown(f"## 📊 Results {len(filtered)} use cases matched your required condition(s)")
            if not final_picked:
                st.info("No use cases match all your criteria")
            else:
                col1, col2 = st.columns(2)
                col1.metric("Total Output", len(final_picked))
                col2.metric("Sanity/L2/L4", f"{len(picked_sanity)}/{len(picked_l2)}/{len(picked_l4)}")
                st.markdown(f"### Showing {len(final_picked)} use cases below:")
                for idx, (uc, freq, score, diff) in enumerate(final_picked):
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

                # Download option
                results_df = pd.DataFrame(final_picked, columns=["Use Case", "Matched Terms", "Score", "Difficulty"])
                st.download_button(
                    label="📥 Download Results",
                    data=results_df.to_csv(index=False),
                    file_name="filtered_use_cases.csv",
                    mime="text/csv"
                )
                 # Download option for ALL filtered results (before difficulty picking)
                all_filtered_results = []
                for uc in filtered:
                    freq = sum(1 for term in prefer_have if term.lower() in uc.lower())
                    score = get_score(uc)
                    diff = classify_difficulty(score)
                    all_filtered_results.append((uc, freq, score, diff))
                all_filtered_df = pd.DataFrame(all_filtered_results, columns=["Use Case", "Matched Terms", "Score", "Difficulty"])
                st.download_button(
                    label="📥 Download ALL Filtered Use Cases",
                    data=all_filtered_df.to_csv(index=False),
                    file_name="all_filtered_use_cases.csv",
                    mime="text/csv"
                )
else:
    if not uploaded_file:
        st.info("Please upload an Excel file to begin")