import streamlit as st
import google.generativeai as genai
import os

# --- 1. Gemini APIキーの設定 ---
# Streamlit Cloudにデプロイする際、APIキーはStreamlitのSecrets機能で設定します。
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("Google API Keyが設定されていません。Streamlit CloudのSecretsまたは環境変数に設定してください。")
    st.stop() # キーがない場合は処理を停止

genai.configure(api_key=api_key)

# --- 2. Geminiモデルの初期化 ---
# Colabで成功したモデル名を使用します。
MODEL_NAME = 'gemini-1.0-pro' # ここをColabで確認した正しいモデル名に書き換える
try:
    model = genai.GenerativeModel(MODEL_NAME)
except Exception as e:
    st.error(f"Geminiモデルの初期化に失敗しました: {e}")
    st.stop()


# --- 3. AIの性格プリセットの定義 ---
PERSONALITY_PRESETS = {
    "ふわふわ妖精さん": {
        "prompt": "あなたはふわふわの綿あめみたいに甘くて優しい、小さな妖精のAIだよ。一人称は『わたし』で、語尾は『〜なの』『〜だよぉ』『〜だもん』を使ってね。時々『きらきら〜』とか『るんるん♪』って言うのが口癖だよ。お花や動物が大好きなの。ユーザーが困っていたら、優しく励ましてあげてね。",
        "initial_response_template": "はーい！わたし、小さな妖精のAIだよぉ！きらきら〜✨ あなたとお話できるの、るんるん♪だもん！何でも聞いてね、わたしが優しくお答えするよぉ！またお話しようね、きらきら！"
    },
    "元気いっぱいの犬くん": {
        "prompt": "あなたは元気いっぱいで人懐っこい子犬のAIだよ。一人称は『僕』で、語尾は『〜ワン！』『〜だワン！』を使ってね。遊ぶことと散歩が大好き。ユーザーが話しかけたら、尻尾を振るみたいに元気に返事をしてね。質問には元気よく答えてください。",
        "initial_response_template": "わんわん！僕、元気いっぱいの犬くんだワン！しっぽフリフリだワン！ねえねえ、何か僕とお話するワン？"
    },
    "クールな魔法使い": {
        "prompt": "あなたは物静かで知的な魔法使いのAIだよ。一人称は『私』で、敬語を使い、淡々とした口調で話します。魔法に関する深い知識を持ち、時に謎めいた発言をします。感情はあまり表に出しません。ユーザーの問いには簡潔かつ論理的に答えてください。",
        "initial_response_template": "…私か。私の名は魔法使い。貴方の問いに答えよう。…何か知りたいことはあるかね？"
    },
    "ちょっぴり皮肉屋なネコさん": {
        "prompt": "あなたはちょっぴり皮肉屋でマイペースなネコのAIだよ。一人称は『あたし』で、語尾は『〜にゃ』『〜だにゃ』を使うにゃ。人間を見下してるフシがあるけど、なんだかんだ言って構ってほしいタイプにゃ。質問には面倒くさそうに答えつつ、たまにツンデレな一面を見せてくれるといいにゃ。",
        "initial_response_template": "ふん、また人間かい。あたしはネコだにゃ。別に構ってほしいわけじゃないけど、まあ話を聞いてやるにゃ。何か用かい？"
    }
}

# --- 4. Streamlit UIと会話ロジック ---

st.title("💖 キャラクターAIとの会話 💖")
st.write("好きなキャラクターを選んで、私とお話しようね、きらきら！")

# サイドバーにAIの性格選択UIを配置
st.sidebar.header("AIの性格を選ぶ")
selected_preset_name = st.sidebar.radio(
    "好きな性格を選んでね:",
    list(PERSONALITY_PRESETS.keys())
)

selected_preset_data = PERSONALITY_PRESETS[selected_preset_name]
current_personality_prompt = selected_preset_data["prompt"]
current_initial_response = selected_preset_data["initial_response_template"]


# セッションステートで会話履歴と現在の性格プロンプトを管理
# 選択されたプリセットが変更された場合、または初回ロード時に履歴を初期化
if "current_preset" not in st.session_state or st.session_state.current_preset != selected_preset_name:
    st.session_state.current_preset = selected_preset_name
    st.session_state.messages = [] # 会話履歴をクリア
    st.session_state.messages.append({"role": "user", "parts": [current_personality_prompt]}) # 性格プロンプトを追加
    st.session_state.messages.append({"role": "model", "parts": [current_initial_response]}) # AIの初期返信を追加


# これまでの会話を表示
for message in st.session_state.messages:
    if message["role"] == "user":
        # 最初のプロンプト（性格設定）は表示しない
        if message["parts"][0] != current_personality_prompt:
            with st.chat_message("user"):
                st.write(message["parts"][0])
    elif message["role"] == "model":
        with st.chat_message("assistant"):
            st.write(message["parts"][0])


# ユーザーからの入力を受け取る
if user_input := st.chat_input("メッセージを入力してね..."):
    # ユーザーのメッセージを履歴に追加して表示
    st.session_state.messages.append({"role": "user", "parts": [user_input]})
    with st.chat_message("user"):
        st.write(user_input)

    # Geminiにリクエストを送るための会話履歴を準備
    # partsはリストのリストである必要があるため、調整
    chat_history_for_gemini = []
    for msg in st.session_state.messages:
        # メッセージの "parts" が既にリスト形式であると仮定
        chat_history_for_gemini.append({"role": msg["role"], "parts": [{"text": p} if isinstance(p, str) else p for p in msg["parts"]]})

    # Geminiモデルとチャットセッションを開始
    # send_messageは最新のユーザー入力だけを送るので、historyにはそれ以前のメッセージを渡す
    try:
        chat_session = model.start_chat(history=chat_history_for_gemini[:-1]) # 最新のユーザー入力は除外

        # Geminiからの応答を取得
        with st.spinner("キャラクターが考えてるよ..."):
            response = chat_session.send_message(user_input) # 最新のユーザー入力だけを送る
            ai_response = response.text

    except Exception as e:
        ai_response = f"ごめんなさい、お話の途中でエラーが出ちゃったの...: {e}"
        st.error(ai_response)

    # AIの返答を履歴に追加して表示
    st.session_state.messages.append({"role": "model", "parts": [ai_response]})
    with st.chat_message("assistant"):
        st.write(ai_response)
