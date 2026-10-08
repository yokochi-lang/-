import streamlit as st
import pandas as pd
import numpy as np
import joblib
import requests
from bs4 import BeautifulSoup
from datetime import datetime

# ページ設定
st.set_page_config(page_title="🚤 競艇AI リアルタイム予想ツール", layout="centered")

# 本日の日付
weekdays = ["月", "火", "水", "木", "金", "土", "日"]
now = datetime.now()
today_str = now.strftime("%Y%m%d")
date_display = f"📅 {now.strftime('%Y年%m月%d日')} ({weekdays[now.weekday()]})"

st.caption(date_display)
st.title("🚤 競艇AI リアルタイム予想ツール")
st.caption("公式サイトから本日の出走表を自動取得し、AIが3連単買い目をリアルタイム予想します")

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

# 開催場一覧の自動取得
@st.cache_data(ttl=3600)
def get_active_venues():
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

# 出走表（選手名・勝率等）をWebスクレイピングで自動取得
@st.cache_data(ttl=600)
def get_race_table(jcd, rno):
    url = f"https://www.boatrace.jp/owpc/pc/race/racelist?rno={rno}&jcd={jcd}&hd={today_str}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    try:
        res = requests.get(url, headers=headers, timeout=5)
        if res.status_code != 200:
            return None
        
        soup = BeautifulSoup(res.text, "html.parser")
        tbodies = soup.select("table tbody.is-fs12")
        
        if not tbodies or len(tbodies) < 6:
            return None
        
        racers = []
        for i, tbody in enumerate(tbodies[:6]):
            # 選手名・級別
            name_tag = tbody.select_one(".is-name a")
            class_tag = tbody.select_one(".is-name span")
            name = name_tag.text.strip().replace(" ", "").replace(" ", "") if name_tag else f"レーサー{i+1}"
            racer_class = class_tag.text.strip() if class_tag else "B1"
            
            # 全国勝率
            win_rate_tds = tbody.select("td")
            win_rate = 5.00
            for td in win_rate_tds:
                txt = td.text.strip()
                try:
                    val = float(txt)
                    if 1.0 <= val <= 9.0 and val != win_rate:
                        win_rate = val
                        break
                except ValueError:
                    continue

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

# 出走表の自動ロード
with st.spinner("出走表データを取得中..."):
    df_racers = get_race_table(selected_jcd, rno)

if df_racers is not None:
    st.markdown("### 📋 公式出走表")
    # 出走表を表形式で綺麗に表示
    st.dataframe(
        df_racers[["艇番", "選手名", "級別", "全国勝率"]],
        use_container_width=True,
        hide_index=True
    )

    # 予想実行ボタン
    if st.button("🔥 このレースのAI予想を計算する", type="primary"):
        # 勝率データを抽出してモデルに入力
        win_rates = df_racers["全国勝率"].tolist()
        input_data = pd.DataFrame([{
            "1号艇_勝率": win_rates[0], "2号艇_勝率": win_rates[1], "3号艇_勝率": win_rates[2],
            "4号艇_勝率": win_rates[3], "5号艇_勝率": win_rates[4], "6号艇_勝率": win_rates[5]
        }])

        probs = model.predict_proba(input_data)[0]

        # 3連単全120通り算出
        sanrentan_list = []
        for i in range(1, 7):
            for j in range(1, 7):
                for k in range(1, 7):
                    if i != j and j != k and i != k:
                        p1 = probs[i - 1]
                        p2 = probs[j - 1] / (1 - p1 + 1e-6)
                        p3 = probs[k - 1] / (1 - p1 - p2 + 1e-6)
                        combo_prob = p1 * p2 * p3
                        
                        # 選手名付きで買い目を表示
                        r1_name = df_racers.loc[i-1, "選手名"]
                        r2_name = df_racers.loc[j-1, "選手名"]
                        r3_name = df_racers.loc[k-1, "選手名"]
                        
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

else:
    st.warning("⚠️ 本日の出走表データが取得できませんでした（ナイター場または時間外の可能性があります）。")
