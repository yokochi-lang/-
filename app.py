import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import re

# ページ設定
st.set_page_config(page_title="🚤 競艇AI リアルタイム予想ツール", layout="centered")

# 本日の日付
weekdays = ["月", "火", "水", "木", "金", "土", "日"]
now = datetime.now()
today_str = now.strftime("%Y%m%d")
date_display = f"📅 {now.strftime('%Y年%m月%d日')} ({weekdays[now.weekday()]})"

st.caption(date_display)
st.title("🚤 競艇AI リアルタイム予想ツール")
st.caption("AIが3連単買い目をリアルタイム予想します")

# AIモデルの読み込み
@st.cache_resource
def load_model():
    return joblib.load("kyotei_ai_model.pkl")

model = load_model()

ALL_JCD = {
    "01": "桐生", "02": "戸田", "03": "江戸川", "04": "平和島", "05": "多摩川", "06": "浜名湖",
    "07": "蒲郡", "08": "常滑", "09": "津", "10": "三国", "11": "びわこ", "12": "住之江",
    "13": "尼崎", "14": "鳴門", "15": "丸亀", "16": "児島", "17": "宮島", "18": "徳山",
    "19": "下関", "20": "若松", "21": "芦屋", "22": "福岡", "23": "唐津", "24": "大村"
}

# 取得処理（タイムアウト2秒で高速判断）
@st.cache_data(ttl=300, show_spinner=False)
def get_race_table_safe(jcd, rno, date_key):
    url = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={rno}&jcd={jcd}&hd={date_key}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        res = requests.get(url, headers=headers, timeout=2)
        if res.status_code != 200:
            return None
        
        soup = BeautifulSoup(res.text, "html.parser")
        tbodies = soup.select("tbody.is-fs12") or soup.select("table.is-w740 tbody")
            
        if not tbodies or len(tbodies) < 6:
            return None
        
        racers = []
        for i in range(6):
            tbody = tbodies[i]
            name_tag = tbody.select_one(".is-name a") or tbody.select_one(".is-name")
            name = name_tag.text.strip().replace(" ", "").replace(" ", "").split("\n")[0] if name_tag else f"{i+1}号艇"
            
            class_tag = tbody.select_one(".is-name span")
            racer_class = class_tag.text.strip() if class_tag else "B1"
            
            text_all = tbody.text
            rates = re.findall(r"\b[1-9]\.\d{2}\b", text_all)
            win_rate = float(rates[0]) if rates else 5.00
            
            racers.append({
                "艇番": i + 1,
                "選手名": name,
                "級別": racer_class,
                "全国勝率": win_rate
            })
            
        return pd.DataFrame(racers)
    except Exception:
        return None

# サイドバー設定
st.sidebar.header("⚙️ レース選択")

selected_jcd = st.sidebar.selectbox(
    "競艇場を選択",
    options=list(ALL_JCD.keys()),
    format_func=lambda x: f"{ALL_JCD[x]} ({x})"
)

rno = st.sidebar.slider("レース番号", 1, 12, 1)

st.subheader(f"📍 {ALL_JCD[selected_jcd]} - 第{rno}レース")

# 自動取得を試行
df_racers = get_race_table_safe(selected_jcd, rno, today_str)

if df_racers is not None:
    st.success("✅ 公式出走表の自動取得に成功しました")
    st.dataframe(df_racers[["艇番", "選手名", "級別", "全国勝率"]], use_container_width=True, hide_index=True)
    win_rates = df_racers["全国勝率"].tolist()
    racer_names = df_racers["選手名"].tolist()
else:
    st.info("ℹ️ 公式サイトからの自動取得（海外サーバー接続制限）のため、勝率入力モードで実行できます。")
    st.write("▼ 各艇の全国勝率を入力・調整してください")
    col1, col2 = st.columns(2)
    with col1:
        r1 = st.number_input("1号艇 勝率", value=7.25, step=0.1)
        r2 = st.number_input("2号艇 勝率", value=5.40, step=0.1)
        r3 = st.number_input("3号艇 勝率", value=4.80, step=0.1)
    with col2:
        r4 = st.number_input("4号艇 勝率", value=5.10, step=0.1)
        r5 = st.number_input("5号艇 勝率", value=3.90, step=0.1)
        r6 = st.number_input("6号艇 勝率", value=3.20, step=0.1)
    win_rates = [r1, r2, r3, r4, r5, r6]
    racer_names = [f"1号艇", f"2号艇", f"3号艇", f"4号艇", f"5号艇", f"6号艇"]

if st.button("🔥 このレースのAI予想を計算する", type="primary"):
    input_data = pd.DataFrame([{
        "1号艇_勝率": win_rates[0], "2号艇_勝率": win_rates[1], "3号艇_勝率": win_rates[2],
        "4号艇_勝率": win_rates[3], "5号艇_勝率": win_rates[4], "6号艇_勝率": win_rates[5]
    }])

    probs = model.predict_proba(input_data)[0]

    sanrentan_list = []
    for i in range(1, 7):
        for j in range(1, 7):
            for k in range(1, 7):
                if i != j and j != k and i != k:
                    p1 = probs[i - 1]
                    p2 = probs[j - 1] / (1 - p1 + 1e-6)
                    p3 = probs[k - 1] / (1 - p1 - p2 + 1e-6)
                    combo_prob = p1 * p2 * p3
                    
                    r1_name = racer_names[i-1]
                    r2_name = racer_names[j-1]
                    r3_name = racer_names[k-1]
                    
                    sanrentan_list.append({
                        "買い目": f"{i} - {j} - {k}",
                        "組み合わせ": f"{i}号艇({r1_name}) ➔ {j}号艇({r2_name}) ➔ {k}号艇({r3_name})",
                        "AI信頼度(%)": round(combo_prob * 100, 2)
                    })

    df_res = pd.DataFrame(sanrentan_list).sort_values(by="AI信頼度(%)", ascending=False).reset_index(drop=True)

    st.markdown("---")
    st.success("🎯 **AI推奨 3連単本命買い目 TOP5**")
    st.dataframe(
        df_res[["買い目", "組み合わせ", "AI信頼度(%)"]].head(5),
        use_container_width=True,
        hide_index=True
    )
