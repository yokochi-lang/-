import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from datetime import datetime

# ページ設定
st.set_page_config(page_title="🚤 競艇AI リアルタイム予想ツール", layout="centered")

# 本日の日付をフォーマット取得（例: 2026年10月08日 (木)）
weekdays = ["月", "火", "水", "木", "金", "土", "日"]
now = datetime.now()
date_str = f"📅 {now.strftime('%Y年%m月%d日')} ({weekdays[now.weekday()]})"

# メインタイトルと日付の表示
st.caption(date_str)
st.title("🚤 競艇AI リアルタイム予想ツール")
st.caption("本日の開催場を自動判定し、3連単の推奨買い目を算出します")

# AIモデルの読み込み
@st.cache_resource
def load_model():
    return joblib.load("kyotei_ai_model.pkl")

model = load_model()

# 全国24場の一覧辞書 (場コード: 場名)
ALL_JCD = {
    "01": "桐生", "02": "戸田", "03": "江戸川", "04": "平和島", "05": "多摩川", "06": "浜名湖",
    "07": "蒲郡", "08": "常滑", "09": "津", "10": "三国", "11": "びわこ", "12": "住之江",
    "13": "尼崎", "14": "鳴門", "15": "丸亀", "16": "児島", "17": "宮島", "18": "徳山",
    "19": "下関", "20": "若松", "21": "芦屋", "22": "福岡", "23": "唐津", "24": "大村"
}

# 本日の開催場を判定して取得する関数
@st.cache_data(ttl=3600)  # 1時間キャッシュ
def get_active_venues():
    today_str = datetime.now().strftime("%Y%m%d")
    active_venues = {}
    headers = {"User-Agent": "Mozilla/5.0"}

    for jcd, name in ALL_JCD.items():
        url = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno=1&jcd={jcd}&hd={today_str}"
        try:
            res = requests.get(url, headers=headers, timeout=2)
            if res.status_code == 200 and "is-fs12" in res.text:
                active_venues[jcd] = f"{name} ({jcd})"
        except Exception:
            continue

    if not active_venues:
        return {jcd: f"{name} ({jcd})" for jcd, name in ALL_JCD.items()}

    return active_venues

# サイドバー設定
st.sidebar.header("レース設定")

with st.sidebar:
    with st.spinner("本日の開催場を確認中..."):
        active_venues = get_active_venues()

selected_jcd = st.sidebar.selectbox(
    "本日開催中の競艇場",
    options=list(active_venues.keys()),
    format_func=lambda x: active_venues[x]
)

rno = st.sidebar.slider("レース番号", 1, 12, 1)

st.subheader(f"📍 {active_venues[selected_jcd]} - 第{rno}レース")

# 出走表データ入力
st.write("▼ 各艇の全国勝率を入力してください")
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
