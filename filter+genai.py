import os
import streamlit as st
import pandas as pd
import re
import json
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Set page config with improved styling
st.set_page_config(
    page_title="Audio Use Case Extractor",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for improved UI
st.markdown("""
    <style>
        .stApp {
            max-width: 1200px;
            padding: 2rem;
        }
        .stMarkdown h1 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 0.3rem;
        }
        .stMarkdown h2 {
            color: #2980b9;
            margin-top: 1.5rem;
        }
        .stMarkdown h3 {
            color: #16a085;
        }
        .stButton>button {
            background-color: #3498db;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            font-weight: bold;
            border: none;
        }
        .stButton>button:hover {
            background-color: #2980b9;
            color: white;
        }
        .stFileUploader {
            border: 2px dashed #3498db;
            border-radius: 5px;
            padding: 1rem;
        }
        .stTextInput input {
            border: 1px solid #3498db;
        }
        .stMultiSelect [role="button"] {
            border: 1px solid #3498db;
        }
        .stNumberInput input {
            border: 1px solid #3498db;
        }
        .stCheckbox [role="checkbox"] {
            border: 1px solid #3498db;
        }
        .info-box {
            background-color: #e8f4fc;
            border-left: 4px solid #3498db;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 0 5px 5px 0;
            color: #2c3e50;
        }
        /* Improved contrast for results */
        .result-item {
            padding: 0.8rem;
            margin: 0.5rem 0;
            border-radius: 5px;
            background-color: #f0f8ff;
            border-left: 4px solid #3498db;
            color: #2c3e50 !important;
        }
        .result-item b {
            color: #2c3e50;
        }
    </style>
""", unsafe_allow_html=True)


# Main title with better formatting
st.title("🎯 Audio Use Case Extractor")
st.markdown("""
    <div style="color: #7f8c8d; margin-bottom: 1.5rem;">
        Combine GenAI intelligence with powerful filtering to extract the most relevant audio use cases
    </div>
""", unsafe_allow_html=True)

# API keys section with improved layout
with st.expander("🔑 API Configuration", expanded=True):
    groq_api_key = st.text_input(
        "Enter your Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        help="Get your API key from https://console.groq.com/keys"
    )

# File upload with better visual cues
uploaded_file = st.file_uploader(
    "📤 Upload your use cases Excel file",
    type=["xlsx"],
    help="The file should contain a 'Name' column with your use cases"
)

# Short form dictionary (unchanged)
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

# Score dictionary for difficulty (unchanged)
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

# Helper functions (unchanged)
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
        return "Easy"
    elif score < 4:
        return "Medium"
    else:
        return "Hard"

# Main processing logic
if uploaded_file and groq_api_key:
    try:
        df = pd.read_excel(uploaded_file)
        if "Name" not in df.columns:
            st.error("❌ The Excel file must contain a 'Name' column.")
            st.stop()

        use_cases = df["Name"].dropna().astype(str).tolist()
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔍 Filter Criteria")
            with st.container(border=True):
                st.markdown("#### 1️⃣ Must Have Terms")
                prefer_have = st.multiselect(
                    "Select terms that must be present",
                    short_forms,
                    help="Use cases must contain AT LEAST ONE of these terms",
                    key="prefer_have"
                )
                
                st.markdown("#### 2️⃣ Exclude Terms")
                must_not_have_option=[term for term in short_forms if term not in prefer_have]
                must_not_have = st.multiselect(
                    "Select terms to exclude",
                    must_not_have_option,
                    help="Use cases containing ANY of these terms will be excluded",
                    key="must_not_have"
                )
        
        with col2:
            st.markdown("### ⚖️ Advanced Options")
            with st.container(border=True):
                use_weights = st.checkbox(
                    "Enable weighted scoring",
                    help="Assign weights to prioritize specific terms"
                )
                
                key_term_options = [term for term in short_forms if term in prefer_have and term not in must_not_have]
                if use_weights:
                    st.markdown("#### 3️⃣ Focus Terms")
                    key_terms = st.multiselect(
                        "Select focus terms for weighted scoring",
                        key_term_options,
                        help="Use cases must contain AT LEAST ONE of these focus terms",
                        key="key_terms"
                    )
        
        # Action section
        st.markdown("### 🚀 Generate Results")
        with st.container(border=True):
            col3, col4 = st.columns(2)
            with col3:
                top_k = st.number_input(
                    "Number of use cases to show",
                    min_value=1,
                    max_value=30,
                    value=10,
                    step=1
                )
            with col4:
                st.write("")
                st.write("")
                search_button = st.button(
                    "Search Use Cases",
                    type="primary",
                    use_container_width=True
                )
        
        if search_button and prefer_have:
            with st.spinner("Processing use cases..."):
                # Step 1: Keep use cases where ANY preferred term is present
                filtered = [uc for uc in use_cases if contains_any(prefer_have, uc)]
                # Step 2: Drop use cases where ANY forbidden term is present
                if must_not_have:
                    filtered = [uc for uc in filtered if not contains_any(must_not_have, uc)]
                # Step 3: Keep use cases where ANY key term is present
                if use_weights and key_terms:
                    filtered = [uc for uc in filtered if contains_any(key_terms, uc)]
                
                if not filtered:
                    st.warning("No use cases match your filter criteria.")
                    st.stop()

                # For each filtered use case, count frequency and score
                results = []
                for uc in filtered:
                    freq = sum(1 for term in prefer_have if term.lower() in uc.lower())
                    score = get_score(uc)
                    diff = classify_difficulty(score)
                    results.append((uc, freq, score, diff))

                # Sort by frequency (desc), then by score (desc)
                results.sort(key=lambda x: (-x[1], -x[2]))

                # Take top 80 or all if less
                top_n_for_genai = min(80, len(results))
                candidate_results = results[:top_n_for_genai]

                # Prepare prompt for Groq LLM
                llm_prompt = f"""
You are an expert at curating technical use cases.
Given the following list of use cases (with their matched terms and scores), select the top {top_k} that are the most diverse and representative.
Avoid repeating use cases that are too similar or that share the same main key term. Try to maximize the coverage of different features and terms.

Use cases:
{json.dumps([
    {
        'use_case': uc,
        'matched_terms': [term for term in prefer_have if term.lower() in uc.lower()],
        'score': score,
        'difficulty': diff
    }
    for uc, freq, score, diff in candidate_results
], indent=2)}

Return ONLY a JSON list of the selected use case strings.
"""

                # Call Groq LLM
                with st.spinner("Using GenAI to select the most diverse and relevant use cases..."):
                    llm = ChatGroq(
                        groq_api_key=groq_api_key,
                        model_name="llama3-70b-8192"
                    )
                    response = llm.invoke(llm_prompt)
                    try:
                        # Try direct JSON parse first
                        selected_use_cases = json.loads(response.content)
                    except Exception:
                        # Try to extract JSON list from the response using regex
                        match = re.search(r'(\[.*\])', response.content, re.DOTALL)
                        if match:
                            selected_use_cases = json.loads(match.group(1))
                        else:
                            st.error("GenAI response could not be parsed. Raw response:")
                            st.code(response.content)
                            selected_use_cases = []

                    if selected_use_cases:
                        st.markdown(f"### 🎯 GenAI-curated Top {top_k} Diverse Use Cases")
                        st.markdown("""
                            <div class="info-box">
                                These use cases were selected by AI to maximize diversity and relevance based on your criteria.
                            </div>
                        """, unsafe_allow_html=True)
                        
                        for i, uc in enumerate(selected_use_cases, 1):
                            st.markdown(f"""
                                <div class="result-item">
                                    <b>{i}.</b> {uc}
                                </div>
                            """, unsafe_allow_html=True)
        
        elif search_button and not prefer_have:
            st.info("ℹ️ Please select at least one 'Must Have' term to start filtering.")
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
else:
    if not uploaded_file:
        st.info("ℹ️ Please upload an Excel file to begin.")
    if not groq_api_key:
        st.info("ℹ️ Please enter your Groq API key to enable GenAI features.")