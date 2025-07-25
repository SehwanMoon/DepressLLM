import streamlit as st
import openai
import numpy as np
import pandas as pd
from openai import OpenAI
import os
import random
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

API_KEY = st.secrets["OPENAI_API_KEY"]
FT_MODEL = st.secrets["FT_MODEL"]

client = OpenAI(api_key=API_KEY)


import streamlit as st
import os, base64

BASE_DIR = os.path.dirname(__file__)
logo_path = os.path.join(BASE_DIR, "logo.png")

# 이미지 파일을 읽어서 base64로 인코딩
with open(logo_path, "rb") as f:
    data = f.read()
b64 = base64.b64encode(data).decode()

st.markdown(
    f"""
    <div style="display:flex; align-items:center;">
      <div style="flex:3; min-width:80px;">
        <img src="data:image/png;base64,{b64}" style="width:110%; height:auto;" />
      </div>
      <div style="flex:7; padding-left:1rem;">
        <h1 style="margin:0;">DepressLLM</h1>
        <p style="margin:0.5rem 0 0;">
          DepressLLM was developed through domain‑adaptive fine‑tuning of the GPT‑4.1 model. 
          It predicts depression based on Experience of Happiness and Experience of Distress data. 
          We are currently preparing a paper entitled <strong>‘A Domain‑Adapted Large Language Model Leveraging Real‑World Narrative Recordings for Interpretable Depression Detection.’
        </p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
# ------------------------------------------
# User Inputs
# ------------------------------------------
st.subheader("Enter Your Happiness & Distress Narratives")
happy    = st.text_area("Experience of Happiness:", height=150)
distress = st.text_area("Experience of Distress:", height=150)

# ------------------------------------------
# Prediction Trigger
# ------------------------------------------
if st.button("Predict Depression"):
    if not happy or not distress:
        st.error("Both texts are required to make a prediction.")
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
    import re
    m = re.match(r"PHQ-9 score:\s*(\d+)", lines[0])
    score = int(m.group(1))
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
    def safe_int(x):
        try:
            return int(x)
        except:
            return None
    
    # 1) 숫자 변환 컬럼 추가
    df_probs["num"] = df_probs["Token"].apply(safe_int)
    
    # 2) 0~27 범위 필터링
    df_filtered = df_probs[df_probs["num"].between(0, 27)]
    
    # 3) 불필요한 임시 컬럼 제거
    df_filtered = df_filtered.drop(columns=["num"])
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
    st.markdown(f"Predicted PHQ-9 Score: {score}")
    st.markdown(f"**Explanation:** {explanation}")
    st.markdown(f"**Significant words/phrases:** {significant}")
    st.markdown("---")
    st.write("**Top PHQ Score Token probabilities:**")
    st.dataframe(df_filtered)
    df_filtered["num"] = df_filtered["Token"].astype(int)
    probs = (
        df_filtered
        .set_index("num")["Prob (%)"]
        .reindex(range(28), fill_value=0)
        .values
    )
    
    # 3) 색상 지정 (0–9 초록, 10–27 빨강)
    nums   = np.arange(28)
    colors = ["green" if n < 5 else "red" for n in nums]
    
    # 4) 차트 그리기
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10,4))

    # 2) axes(플롯 영역) 배경색 설정
    ax.set_facecolor('#f2f2f2')
    ax.bar(nums, probs, color=colors, width=0.6)
    ax.set_xticks(nums)
    ax.set_xlabel("PHQ‑9 Score Token")
    ax.set_ylabel("Probability (%)")
    ax.set_title("SToPS (Score‑guided Token Probability Summation)")
    ax.grid(axis="y", alpha=0.0)
    
    # 5) 구간 합계 텍스트
    sum_norm = probs[:5].sum()
    sum_dep  = probs[5:].sum()

    st.pyplot(fig)
    st.write(f"**Grouped probability** (0–4 vs 5–27): {pct0_4:.2f}% vs {pct5_27:.2f}%")

    
    st.write(f"**Confidence:** {confidence:.2f}")
    st.write(f"**Depressive symptoms predicted?** {'Yes' if depression else 'No'}")

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
      <span style="color:#004098; font-weight:bold;">
        Chonnam National University Hospital
      </span>
    </p>
    """,
    unsafe_allow_html=True
)
etri_path    = os.path.join(BASE_DIR, "etri2.png")
chonnam_path = os.path.join(BASE_DIR, "chonnam.png")

def img_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

etri_b64    = img_b64(etri_path)
chonnam_b64 = img_b64(chonnam_path)

st.markdown(
    f"""
    <div style="
        display: flex;
        flex-direction: row;
        flex-wrap: nowrap;       /* 한 줄로 고정 */
        align-items: center;
        justify-content: center;
        gap: 3rem;               /* 로고 사이 간격 */
    ">
      <img src="data:image/png;base64,{etri_b64}"
           style="width:37%; max-width:100px; height:auto;" />
      <img src="data:image/png;base64,{chonnam_b64}"
           style="width:66%; max-width:170px; height:auto;" />
    </div>
    """,
    unsafe_allow_html=True
)
