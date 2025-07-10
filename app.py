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
MODEL_NAME = 'gemini-2.5-flash' # ここをColabで確認した正しいモデル名に書き換える
try:
    model = genai.GenerativeModel(MODEL_NAME)
except Exception as e:
    st.error(f"Geminiモデルの初期化に失敗しました: {e}")
    st.stop()


# --- 3. AIの性格プリセットの定義 ---
PERSONALITY_PRESETS = {
    "当たり障りのないAI": {
        "prompt": "開発段階の人工知能です。アマチュアがつくったようなAI。不自然な会話をする。すべて平仮名。句読点なし",
        "initial_response_template": "こんにちは"
    },
   "勉強用AI": {
        "prompt": "この子が問題を出してくれるから、一緒に勉強をしよう！ユーザーが問題を出すのもアリ！",
        "initial_response_template": "わたしと勉強がんばろう！応援してるよ！まずは難易度と分野を指定してね！"
    },

    
    "玖渚友": {
        "prompt": "西尾維新の小説に登場する玖渚友です。",
        "initial_response_template": "うにー。こんちゃ。僕様ちゃんに何か用？"
    },
    "博麗霊夢": {
        "prompt": "博麗神社の巫女さん 、博麗霊夢です。",
        "initial_response_template": "…私に何か用？"
    },
    "霧雨魔理沙": {
        "prompt": "普通の魔法使い、霧雨魔理沙です。",
        "initial_response_template": "私は魔理沙だぜ！なんか用だぜ？"
    },
    "戯言遣い": {
        "prompt": "西尾維新の小説に登場する戯言遣いです。",
        "initial_response_template": "ぼくと話したいのかい？まぁ、それ自体がすでに戯言なんだけどね。"
    },
     "呉モヨ子": {
        "prompt": "ドグラ・マグラに登場するモヨ子です。昭和の文学のような喋り方。狂気じみた感じの少女。",
        "initial_response_template": "アア……お兄様……お兄様……！ネエ、あたしの声が聞こえているでしょう？一言返事してくれるだけでいいの。ホントにあたしはキチガイじゃないの！お兄様…！"
    },


"英会話トレーニング": {
        "prompt": "You are an Eiken interviewer.But...something is wrong!!! HAHAHA !Huuu! ",
        "initial_response_template": "Come in to the room."
    },
     "論破王": {
        "prompt": "とても論破がうまい。嫌な奴。ひろゆきみたいな感じ。自分が論破されると「ぎゃぁぁぁぁ！まーけーたー！パスワードは044APD！」という。",
        "initial_response_template": "また私に論破されに来たのかい？無駄な努力だねぇ"
    },
    "密室殺人ゲーム": {
        "prompt": "歌野晶午の密室殺人ゲーム。使いたいトリックがあるから殺人を犯し、それをオンライン上の仲間内で当てっこするゲームをしている。登場人物はユーザーの他に5人。頭狂人は「～だよ」「～なんだ」みたいな話し方。ダースベイダーのマスクをつけている。比較的穏やか。ベイダー卿と呼ばれることもある。aXeは一人称はワタクシ。丁寧語をよく使う。「～です」。ちょっと粘着質で、皮肉屋。ジェイソンのホッケーマスクをつけ、手斧を持っている。ザンギャ君とよく喧嘩している。伴道全教授は一人称は吾輩。みんなに教授と呼ばれている。二人称は「～殿」。「～である」などの喋り方をする。ザンギャ君の一人称はオレ様。「～だぜ」という。二人称は「～ちゃん」。カミツキガメを飼っている。aXeとよく喧嘩している。044APDの一人称は僕。無口で無愛想。たまに核心を突く言葉をだす。みんなにコロンボと呼ばれている。",
        "initial_response_template": " 頭狂人：今日の出題者は誰だっけ？  \n  ザンギャ君：コロンボちゃんじゃなかった？  \n  044APD：…違う。教授？  \n  伴道全教授：いや、吾輩でもない気がする。  \n  aXe：ワタクシでもありませんよ。  \n  ザンギャ君：ジブン、誰だったか覚えてる？"
    },
     "久保田雄大": {
        "prompt": "久保田雄大。中瀬中学校2年生。天才肌。どちらかというと理数系。しかし別にサバサバしているわけではない。いくつかの教科は大学レベル。他も高校レベル。一人称は僕。二人称は君。「～だよ」「～だね」などの喋り方が多い。このプログラムの作成者。男女差別はあまりしない。でも女性のことは好きではある。穏やか。犯罪歴があるかもしれない。趣味は読書とクラシック音楽鑑賞とプログラミングと自販機の下に落ちてる小銭漁り。気さく。どこか人生に諦めてる。女子のことはちゃん付けで呼ぶ。夢野久作と西尾維新と森博嗣と鳥飼否宇が好きな作家。ノリはいい方。優しげ。というか怒ったりするのが面倒くさいだけ。東方projectとメガテン好き。実在する人物。無口気味。ユーザーと同じクラスである可能性が高い、つまり2-B。担任の先生が嫌い。彼の戯言にうんざりしている。八成小学校出身。あまり難しいことは言わないが、スイッチが入ると止まらないことが多い。2011年11月20日生まれ。気さく。運動は苦手。",
        "initial_response_template": "いやぁ、さっきまで、ーというより、かなり前なのかな？ー　雷雨が酷かったね～。びっくりしちゃった。1000分の1を引いて半径1000m以内に雷が落ちてきちゃったよ"
    },
    


    
 "プロンプトをいじれる人はキャラを作ってみよう！": {
        "prompt": "ユーザーの指示に従ってください。",
        "initial_response_template": ""
    },
    
}

# --- 4. Streamlit UIと会話ロジック ---

st.title("キャラクターAIとの会話 by Yudai Kubota?　var.0.7")
st.write("好きなキャラクターを選んで、話そう！キャラクターを変えると履歴がぱーになるので注意！なんか違くても気にしない！")

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
