import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ページ設定
st.set_page_config(page_title="🚤 競艇AI予想ツール", layout="centered")

st.title("🚤 競艇AI リアルタイム予想ツール")
st.caption("AIが各艇の勝率から3連単の推奨買い目を自動算出します")

# AIモデルの読み込み
@st.cache_resource
def load_model():
    return joblib.load("kyotei_ai_model.pkl")

model = load_model()

# サイドバー：レース選択
st.sidebar.header("レース設定")
jcd = st.sidebar.selectbox("開催場", ["大村 (24)", "平和島 (04)", "住之江 (12)", "桐生 (01)"])
rno = st.sidebar.slider("レース番号", 1, 12, 1)

st.subheader(f"📍 {jcd} - 第{rno}レース")

# 入力フォーム（各艇の勝率）
st.write("▼ 各艇の全国勝率を入力してください（初期値はサンプル数値です）")

col1, col2 = st.columns(2)
with col1:
    r1 = st.number_input("1号艇 勝率", value=7.25, step=0.1)
    r2 = st.number_input("2号艇 勝率", value=5.40, step=0.1)
    r3 = st.number_input("3号艇 勝率", value=4.80, step=0.1)
with col2:
    r4 = st.number_input("4号艇 勝率", value=5.10, step=0.1)
    r5 = st.number_input("5号艇 勝率", value=3.90, step=0.1)
    r6 = st.number_input("6号艇 勝率", value=3.20, step=0.1)

if st.button("🔥 AI予想を実行する", type="primary"):
    input_data = pd.DataFrame([{
        "1号艇_勝率": r1, "2号艇_勝率": r2, "3号艇_勝率": r3,
        "4号艇_勝率": r4, "5号艇_勝率": r5, "6号艇_勝率": r6
    }])

    # 各艇の1着確率を予測
    probs = model.predict_proba(input_data)[0]

    # 3連単全120通りの計算
    sanrentan_list = []
    for i in range(1, 7):
        for j in range(1, 7):
            for k in range(1, 7):
                if i != j and j != k and i != k:
                    p1 = probs[i - 1]
                    p2 = probs[j - 1] / (1 - p1 + 1e-6)
                    p3 = probs[k - 1] / (1 - p1 - p2 + 1e-6)
                    combo_prob = p1 * p2 * p3
                    sanrentan_list.append({
                        "買い目": f"{i} - {j} - {k}",
                        "AI予測確率 (%)": round(combo_prob * 100, 2)
                    })

    df_res = pd.DataFrame(sanrentan_list).sort_values(by="AI予測確率 (%)", ascending=False).reset_index(drop=True)

    st.markdown("---")
    st.success("🎯 **AI推奨 3連単買い目 TOP5**")
    st.dataframe(df_res.head(5), use_container_width=True)