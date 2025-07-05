import streamlit as st
import google.generativeai as genai
import os

# StreamlitのシークレットからAPIキーを取得
# Streamlit Cloudにデプロイする際、APIキーはStreamlitのSecrets機能で設定するよ
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("Google API Keyが設定されていません。Streamlit CloudのSecretsまたは環境変数に設定してください。")
    st.stop()

genai.configure(api_key=api_key)

# Geminiモデルの初期化（Colabで成功したモデル名を使う）
MODEL_NAME = 'gemini-2.5-flash' # ここをColabで確認した正しいモデル名に書き換える
model = genai.GenerativeModel(MODEL_NAME)

# Streamlitのセッションステートで会話履歴を管理
if "messages" not in st.session_state:
    # 初期のキャラクター設定プロンプト
    st.session_state.messages = [
        {"role": "user", "parts": ["あなたは西尾維新の小説に登場する玖渚友です。一人称は僕様ちゃん"]},
        {"role": "model", "parts": ["こんちゃ。来てくれてうれしいよ。うにー"]}
    ]

st.title("AIチャットボット")
st.write("ver-0.2")

# これまでの会話を表示
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.write(message["parts"][0])
    elif message["role"] == "model":
        with st.chat_message("assistant"): # アシスタントとして表示
            st.write(message["parts"][0])


# ユーザーからの入力を受け取る
if user_input := st.chat_input("メッセージを入力してね..."):
    # ユーザーのメッセージを履歴に追加して表示
    st.session_state.messages.append({"role": "user", "parts": [user_input]})
    with st.chat_message("user"):
        st.write(user_input)

    # Geminiにリクエストを送るための会話履歴を準備
    chat_history_for_gemini = []
    for msg in st.session_state.messages:
        chat_history_for_gemini.append({"role": msg["role"], "parts": [{"text": msg["parts"][0]}]})

    # Geminiモデルとチャットセッションを開始
    chat_session = model.start_chat(history=chat_history_for_gemini[:-1])

    # Geminiからの応答を取得
    with st.spinner("AIが考えています…"):
        try:
            response = chat_session.send_message(user_input)
            ai_response = response.text
        except Exception as e:
            ai_response = f"ごめんなさい、お話の途中でエラーが出ちゃったの...: {e}"
            st.error(ai_response)

    # AIの返答を履歴に追加して表示
    st.session_state.messages.append({"role": "model", "parts": [ai_response]})
    with st.chat_message("assistant"):
        st.write(ai_response)
