import streamlit as st
import openai
import numpy as np
import pandas as pd
from openai import OpenAI
import os
# ------------------------------------------
# 설정
# ------------------------------------------
API_KEY = st.secrets["OPENAI_API_KEY"]
FT_MODEL = st.secrets["FT_MODEL"]

client = OpenAI(api_key=API_KEY)
# ------------------------------------------
# App Layout
# ------------------------------------------
# 이 파일(App.py)의 디렉터리 경로
BASE_DIR = os.path.dirname(__file__)

app_logo      = os.path.join(BASE_DIR, "logo.png")

col1, col2 = st.columns([1, 7])

# 2) 왼쪽에 로고
col1.image(app_logo, width=2054)

# 3) 오른쪽에 DepressLLM 타이틀
col2.markdown("# DepressLLM")
st.markdown(
    """
    DepressLLM was developed through domain‑adaptive fine‑tuning of the GPT‑4.1 model.<br>
    We are currently preparing a paper entitled ‘A Domain‑Adapted Large Language Model Leveraging Real‑World Narrative Recordings for Interpretable Depression Detection.’
    """,
    unsafe_allow_html=True
)
# ------------------------------------------
# User Inputs
# ------------------------------------------
st.header("Enter Participant Transcripts")
happy    = st.text_area("Experience of Happiness:", height=150)
distress = st.text_area("Experience of Distress:", height=150)

# ------------------------------------------
# Prediction Trigger
# ------------------------------------------
if st.button("Predict Depression"):
    if not happy or not distress:
        st.error("Both transcripts are required to make a prediction.")
        st.stop()

    # Build messages for ChatCompletion
    system_msg = (
        "You will be given a transcript of a participant talking about "
        "the topics of happiness and distress.\n"
        "1. Classify the transcript into one of the PHQ-9 scores (0–27).\n"
        "2. Write a brief explanation for your prediction by referring to evidence from the transcript.\n"
        "3. Highlight all significant words or phrases that influenced your decision, separated by commas."
    )
    user_msg = (
        f"Experience of Happiness: {happy}\n\n"
        f"Experience of Distress: {distress}\n\n"
        "PHQ-9 score:"
    )

    ft_model = client.fine_tuning.jobs.retrieve(FT_MODEL).fine_tuned_model
    response = client.chat.completions.create(
        model=ft_model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg}
        ],
        temperature=0,
        top_p=1,
        logprobs=True,top_logprobs=20,
        max_tokens=2000,
    )

    # --------------------------------------
    # Parse Model Output
    # --------------------------------------
    content = response.choices[0].message.content
    lines   = content.splitlines()
    try:
        score = int(lines[0].strip())
    except ValueError:
        score = None

    explanation = ""
    significant = ""
    for line in lines:
        if line.startswith("Explanation:"):
            explanation = line.split("Explanation:",1)[1].strip()
        if line.startswith("Significant words/phrases:"):
            significant = line.split("Significant words/phrases:",1)[1].strip()

    # --------------------------------------
    # Token-level probabilities
    # --------------------------------------
    top_logprobs = response.choices[0].logprobs.content[0].top_logprobs
    token_probs  = [[entry.token, float(np.exp(entry.logprob)*100)] for entry in top_logprobs]
    df_probs     = pd.DataFrame(token_probs, columns=["Token","Prob (%)"])

    # --------------------------------------
    # Grouped probability & confidence
    # --------------------------------------
    def safe_int(x):
        try: return int(x)
        except: return None

    grp0_4 = sum(p for t,p in token_probs if isinstance(safe_int(t),int) and 0 <= safe_int(t) <= 4)
    grp5_27= sum(p for t,p in token_probs if isinstance(safe_int(t),int) and 5 <= safe_int(t) <= 27)
    tot    = grp0_4 + grp5_27
    pct0_4 = (grp0_4/tot*100) if tot else 0
    pct5_27= (grp5_27/tot*100) if tot else 0
    confidence = abs(pct5_27-50)/50 if tot else 0
    depression = pct5_27 >= 50

    # --------------------------------------
    # Display Results
    # --------------------------------------
    st.subheader(f"Predicted PHQ-9 Score: {score}")
    st.markdown(f"**Explanation:** {explanation}")
    st.markdown(f"**Significant words/phrases:** {significant}")
    st.markdown("---")
    st.write("**Top token probabilities:**")
    st.dataframe(df_probs)
    st.write(f"**Grouped probability** (0–4 vs 5–27): {pct0_4:.2f}% vs {pct5_27:.2f}%")

    
    st.write(f"**Confidence:** {confidence:.2f}")
    st.write(f"**Depression predicted?** {'Yes' if depression else 'No'}") 

# ------------------------------------------
# Footer: 기관 로고 & 문구
# ------------------------------------------
st.markdown("---")
st.markdown(
    """
    <p style="font-size:13px; text-align:center;">
      Developed by 
      <span style="color:#005CA9; font-weight:bold;">
        Electronics and Telecommunications Research Institute (ETRI)
      </span>
      &amp;
      <span style="color:#007847; font-weight:bold;">
        Chonnam National University
      </span>
    </p>
    """,
    unsafe_allow_html=True
)
f0, f1, f4,f2, f3  = st.columns([1.7,1,0.5,1.1,1.7])
f1.image( os.path.join(BASE_DIR, "etri2.png"),    width=2056)
f2.image(os.path.join(BASE_DIR, "chonnam.png"), width=2056)

