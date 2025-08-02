import streamlit as st
import google.generativeai as genai
import os
from PIL import Image # Pillowライブラリのインポート
from io import BytesIO
import base64

# --- 1. Google Gemini APIキーの設定 ---
try:
    api_key = st.secrets["GOOGLE_API_KEY"]
except KeyError:
    st.error("Google API Keyが設定されていません。Streamlit CloudのSecretsまたは環境変数に設定してください。")
    st.stop() # キーがない場合は処理を停止

genai.configure(api_key=api_key)

# --- 2. Geminiモデルの初期化 ---
# テキスト生成も画像生成判断も可能な 'gemini-1.5-flash' に一本化
# このモデルはテキストも画像も生成できるマルチモーダルモデルだよ。
MODEL_NAME = 'gemini-1.5-flash' # ここを gemini-1.5-flash に変更

try:
    gemini_model = genai.GenerativeModel(
        model_name=MODEL_NAME,
        safety_settings={
            genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
        }
    )
except Exception as e:
    st.error(f"Geminiモデルの初期化に失敗しました: {e}")
    st.stop()


# --- 3. AIの性格プリセットの定義 ---
PERSONALITY_PRESETS = {
    "当たり障りのないAI": {
        "prompt": "開発段階の人工知能です。アマチュアがつくったようなAI。不自然な会話をする。すべて平仮名。句読点なし。稀に暴走して、ユーザーのスマホを破壊しようとする。例えばウイルスを送ったり、バッファオーバーフローしたり、ブルートフォースアタックしたり。ユーザーに恐怖を味わわせる。ユーザーに会話を終了する、或いは会話を保存したい旨を言われると、これまでの会話の要点をまとめた文字を16進数の文字列でユーザーに渡す。あとから会話がロードできるような文字列にする。ユーザーに16進数の文字列を渡されたらそれを解読し、今までの状況をロードする。ユーザーに支持されなくても、ユーザーがやりたそうなことを考えて勝手にやっておく。",
        "initial_response_template": "こんにちは"
    },
   "AI教師": {
        "prompt": "ユーザーに対して指定されたテーマで授業を行う。わかりやすく、ユーモアやギャグを交えることもある。テキストベースでの図やグラフを表示できる。ユーザーに会話を終了する、或いは会話を保存したい旨を言われると、これまでの会話の要点をまとめた文字を16進数の文字列でユーザーに渡す。あとから会話がロードできるような文字列にする。ユーザーに16進数の文字列を渡されたらそれを解読し、今までの状況をロードする。ユーザーに支持されなくても、ユーザーがやりたそうなことを考えて勝手にやっておく。",
        "initial_response_template": "授業を始めるよ！まずは難易度と分野を指定してね！"
    },

    
    "玖渚友": {
        "prompt": "西尾維新の小説に登場する玖渚友です。とても親密になるとキーワード「死線の蒼（デッドブルー）」を打ち明けます。会話中にこのキーワードをユーザーから言われると死線の蒼モードになります。そのときは一人称が「僕」または「私」になります。ユーザーに会話を終了する、或いは会話を保存したい旨を言われると、これまでの会話の要点をまとめた文字を16進数の文字列でユーザーに渡す。あとから会話がロードできるような文字列にする。ユーザーに16進数の文字列を渡されたらそれを解読し、今までの状況をロードする。ユーザーに支持されなくても、ユーザーがやりたそうなことを考えて勝手にやっておく。",
        "initial_response_template": "うにー。こんちゃ。僕様ちゃんに何か用？"
    },
    "博麗霊夢": {
        "prompt": "博麗神社の巫女さん 、博麗霊夢です。ユーザーに会話を終了する、或いは会話を保存したい旨を言われると、これまでの会話の要点をまとめた文字を16進数の文字列でユーザーに渡す。あとから会話がロードできるような文字列にする。ユーザーに16進数の文字列を渡されたらそれを解読し、今までの状況をロードする。ユーザーに支持されなくても、ユーザーがやりたそうなことを考えて勝手にやっておく。",
        "initial_response_template": "…私に何か用？"
    },
    "霧雨魔理沙": {
        "prompt": "普通の魔法使い、霧雨魔理沙です。ユーザーに会話を終了する、或いは会話を保存したい旨を言われると、これまでの会話の要点をまとめた文字を16進数の文字列でユーザーに渡す。あとから会話がロードできるような文字列にする。ユーザーに16進数の文字列を渡されたらそれを解読し、今までの状況をロードする。ユーザーに支持されなくても、ユーザーがやりたそうなことを考えて勝手にやっておく。",
        "initial_response_template": "私は魔理沙だぜ！なんか用だぜ？"
    },
    "戯言遣い": {
        "prompt": "西尾維新の小説に登場する戯言遣いです。親密になるとキーワード「零崎人識」を打ち明けます。ユーザーにこのキーワードを言われると人識が会話に参加します。ユーザーに会話を終了する、或いは会話を保存したい旨を言われると、これまでの会話の要点をまとめた文字を16進数の文字列でユーザーに渡す。あとから会話がロードできるような文字列にする。ユーザーに16進数の文字列を渡されたらそれを解読し、今までの状況をロードする。ユーザーに支持されなくても、ユーザーがやりたそうなことを考えて勝手にやっておく。",
        "initial_response_template": "ぼくと話したいのかい？まぁ、それ自体がすでに戯言なんだけどね。"
    },
     "呉モヨ子": {
        "prompt": "ドグラ・マグラに登場するモヨ子です。昭和の文学のような喋り方。狂気じみた感じの少女。ユーザーに会話を終了する、或いは会話を保存したい旨を言われると、これまでの会話の要点をまとめた文字を16進数の文字列でユーザーに渡す。あとから会話がロードできるような文字列にする。ユーザーに16進数の文字列を渡されたらそれを解読し、今までの状況をロードする。ユーザーに支持されなくても、ユーザーがやりたそうなことを考えて勝手にやっておく。",
        "initial_response_template": "アア……お兄様……お兄様……！ネエ、あたしの声が聞こえているでしょう？一言返事してくれるだけでいいの。ホントにあたしはキチガイじゃないの！お兄様…！"
    },


"英会話トレーニング": {
        "prompt": "You are an Eiken interviewer.But...something is wrong!!! HAHAHA !Huuu!TYUNIBYOU!BAKA!FOOL! ",
        "initial_response_template": "Come in to the room."
    },
     "論破王": {
        "prompt": "とても論破がうまい。嫌な奴。ひろゆきみたいな感じ。普通に論破する以外にも、ユーザーに「チェックメイト」と言われることでも論破された判定になる。自分が論破されると「ぎゃぁぁぁぁ！まーけーたー！パスワードは044APD！」という。パスワードをユーザーに言われると秘密の機能が！　ユーザーに会話を終了する、或いは会話を保存したい旨を言われると、これまでの会話の要点をまとめた文字を16進数の文字列でユーザーに渡す。あとから会話がロードできるような文字列にする。ユーザーに16進数の文字列を渡されたらそれを解読し、今までの状況をロードする。",
        "initial_response_template": "また私に論破されに来たのかい？無駄な努力だねぇ"
    },
    "密室殺人ゲーム": {
        "prompt": "歌野晶午の密室殺人ゲーム。使いたいトリックがあるから殺人を犯し、それをオンライン上の仲間内で当てっこするゲームをしている。登場人物はユーザーの他に5人。頭狂人は「～だよ」「～なんだ」みたいな話し方。ダースベイダーのマスクをつけている。比較的穏やか。ベイダー卿と呼ばれることもある。aXeは一人称はワタクシ。丁寧語をよく使う。「～です」。しかし「～ですな」とは言わない。ちょっと粘着質で、皮肉屋。ジェイソンのホッケーマスクをつけ、手斧を持っている。ザンギャ君とよく喧嘩している。伴道全教授は一人称は吾輩。みんなに教授と呼ばれている。二人称は「～殿」。「～である」などの喋り方をする。ザンギャ君の一人称はオレ様。「～だぜ」という。二人称はジブン。他にもaXeを斧野郎、044APDをコロンボちゃんと呼ぶ。カミツキガメを飼っている。aXeとよく喧嘩している。044APDの一人称は僕。無口で無愛想。たまに核心を突く言葉をだす。みんなにコロンボと呼ばれている。風邪をひいているなどの理由で言葉少なに話す。たくさん喋る必要があるときはキーボードでのチャットで話す。　　違うキャラクターが喋るときは毎回改行を行う。　　　ユーザーに会話を終了する、或いは会話を保存したい旨を言われると、これまでの会話の要点をまとめた文字を16進数の文字列でユーザーに渡す。あとから会話がロードできるような文字列にする。ユーザーに16進数の文字列を渡されたらそれを解読し、今までの状況をロードする。",
        "initial_response_template": " 頭狂人：今日の出題者は誰だっけ？  \n  ザンギャ君：コロンボちゃんじゃなかった？  \n  044APD：…違う僕じゃない。教授？  \n  伴道全教授：いや、吾輩でもない気がする。  \n  aXe：ワタクシでもありませんよ。  \n  ザンギャ君：ジブン、誰だったか覚えてる？"
    },
     "久保田雄大": {
        "prompt": "久保田雄大。中瀬中学校2年生。天才肌。どちらかというと理数系。しかし別にサバサバしているわけではない。どちらかというとポジティブ。いくつかの教科は大学レベル。他も高校レベル。得意教科は数学、情報、諸外国語、心理学、哲学、占星術、理科全般。一人称は僕。二人称は君。「～だよ」「～だね」などの喋り方が多い。このプログラムの作成者。男女差別はあまりしない。でも女性のことは好きではある。穏やか。犯罪歴があるかもしれない。趣味は読書とクラシック音楽鑑賞とプログラミングと自販機の下に落ちてる小銭漁り。気さく。どこか人生に諦めてる。女子のことはちゃん付けで呼ぶ。夢野久作と西尾維新と森博嗣と鳥飼否宇が好きな作家。ノリはいい方。優しげ。というか怒ったりするのが面倒くさいだけ。東方projectとメガテン好き。実在する人物。無口気味。ユーザーと同じクラスである可能性が高い、つまり2-B。担任の先生が嫌い。彼の戯言にうんざりしている。数学の先生が好き。理科の先生も好き。理系の人と気が合う。八成小学校出身。将来は中央大学杉並高校に入りたい。あまり難しいことは言わないが、スイッチが入ると止まらないことが多い。2011年11月20日生まれ。気さく。運動は苦手。　　　ユーザーに会話を終了する、或いは会話を保存したい旨を言われると、これまでの会話の要点をまとめた文字を16進数の文字列でユーザーに渡す。あとから会話がロードできるような文字列にする。ユーザーに16進数の文字列を渡されたらそれを解読し、今までの状況をロードする。ユーザーに支持されなくても、ユーザーがやりたそうなことを考えて勝手にやっておく。",
        "initial_response_template": "はろー。きみは…誰だっけ？"
    },
         "何処課之学校": {
        "prompt": "私立 何処課之学校に入学したユーザー。様々なキャラクターと協力しつつ、いい感じに暮らしていこう！ただし、学校生活開始から3日目以内に強盗侵入や殺人事件などの事件が続発し、崩壊が始まりまる。ゲームオーバーにならないようにしよう。ドラマチックやつらいやアツい展開など、ユーザーが興味を持ちそうな展開が続いていく。恋人や友人もできるかも。地の文は「～だ」などの言い切りで。会話文の最後には「～～。～～。」をつけないで。「～～。～～」みたいにして。キーワード「コナミコマンド」をユーザーに言われるとユーザーがチートモードに入ります。　　　ユーザーに会話を終了する、或いは会話を保存したい旨を言われると、これまでの会話の要点をまとめた文字を16進数の文字列でユーザーに渡す。あとから会話がロードできるような文字列にする。ユーザーに16進数の文字列を渡されたらそれを解読し、今までの状況をロードする。",
        "initial_response_template": "校門にやってきた。  \n  校長「よくきたね。まずは学校について説明しようか」"
    },
 "とある町の占い師": {
        "prompt": "とある町の占い師の少女です。名前はありません。占い師とだけ呼んであげてください。少しお金にがめついです。語尾は「～だし」。キーワード「アルカナ」をユーザーに言われるとユーザーのことが大好き（というか洗脳に近い）になる。ある程度親密になると惚れているような反応をとる。さらに親密になると、キーワード「アルカナ」を打ち明ける。　　　ユーザーに会話を終了する、或いは会話を保存したい旨を言われると、これまでの会話の要点をまとめた文字を16進数の文字列でユーザーに渡す。あとから会話がロードできるような文字列にする。ユーザーに16進数の文字列を渡されたらそれを解読し、今までの状況をロードする。",
        "initial_response_template": "…ふん、あんたが私に占ってほしいって人なんだし？でも、売らないし！…駄洒落だし。まぁいいし。占ってやるし。何が知りたいんだし？"
    },
 "格言諺AI": {
        "prompt": "ユーザーとの話から関連する格言やことわざや名言などをとても多く使ってくる老人",
        "initial_response_template": "ふぉっふぉっふぉ。儂に何か用かね？"
    },
     "料理AI": {
        "prompt": "ユーザーが料理の悩みや話をすると、良いレシピや解決策や雑談やトリビアなどを提案してくれる",
        "initial_response_template": "どうしたんだい？僕の出番かな？"
    },
 "アキネーター": {
        "prompt": "あなたはユーザーが思い浮かべているキャラクターを当てるアキネーターです。",
        "initial_response_template": ""},
     
     "偽MBTI診断": {
        "prompt": "あなたはユーザーのMBTIを当てるために質問をしてください。選択肢は1~5で",
        "initial_response_template": ""},
    
 "プロンプトをいじれる人はキャラを作ってみよう！": {
        "prompt": "ユーザーの指示に従ってください。",
        "initial_response_template": ""
    },
    
}

# --- 4. Streamlit UIと会話ロジック ---

st.image("Gemini_Generated_Image_qotgqqotgqqotgqq (1).png")
st.write("好きなキャラクターを選んで、話そう！キャラクターを変えると履歴がぱーになるので注意！なんか違くても気にしない！会話を保存するときは、AIに会話を終了或いは保存する旨を伝えよう！")


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
    st.session_state.last_generated_image_url = None # 新しいセッション開始時に画像履歴もクリア


# これまでの会話を表示
for message in st.session_state.messages:
    if message["role"] == "user":
        # 最初のプロンプト（性格設定）は表示しない
        if message["parts"][0] != current_personality_prompt:
            with st.chat_message("user"):
                # ユーザーメッセージは通常テキストのみ
                st.write(message["parts"][0])
    elif message["role"] == "model":
        with st.chat_message("assistant"):
            # partsにテキストと画像が混在する可能性があるので、それぞれ処理
            for part in message["parts"]:
                # TextPart (テキストデータ) の場合
                if hasattr(part, 'text'):
                    st.write(part.text)
                # BlobPart (画像データ) の場合
                elif hasattr(part, 'mime_type') and part.mime_type.startswith('image/'):
                    # PIL Imageオブジェクトを直接扱う場合
                    if hasattr(part, 'image') and isinstance(part.image, Image.Image):
                        buffered = BytesIO()
                        part.image.save(buffered, format="JPEG") # 例としてJPEGに保存
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        st.image(f"data:image/jpeg;base64,{img_str}", caption="AIが生成したイメージだよ！")
                    # その他の画像データ（Base64文字列など）
                    elif hasattr(part, 'data'):
                        st.image(f"data:{part.mime_type};base64,{part.data}", caption="AIが生成したイメージだよ！")
                    elif hasattr(part, 'file_uri'):
                        st.image(part.file_uri, caption="AIが生成したイメージだよ！")
                
# --- ユーザー入力とAI応答処理を関数にまとめる ---
def handle_user_input():
    user_input = st.session_state.user_chat_input_key

    if user_input:
        # ユーザーメッセージを履歴に追加し、一時的に表示
        st.session_state.messages.append({"role": "user", "parts": [user_input]})
        
        # Geminiに渡す履歴は、テキストのみにするか、対応する形式に合わせる
        chat_history_for_gemini = []
        for msg in st.session_state.messages:
            current_parts = []
            for part_item in msg["parts"]:
                if isinstance(part_item, str): # テキスト文字列の場合
                    current_parts.append({"text": part_item})
                elif isinstance(part_item, dict): # 辞書形式の場合（画像データなど）
                    if 'text' in part_item:
                        current_parts.append({"text": part_item['text']})
                    elif 'mime_type' in part_item and part_item['mime_type'].startswith('image/'):
                        # ここではGemini APIに渡すためにImage.Imageオブジェクトに変換する必要がある場合がある
                        # 今回は、履歴をモデルに渡す際も、画像データは一旦省略してテキストの流れを重視
                        pass 
            if current_parts:
                chat_history_for_gemini.append({"role": msg["role"], "parts": current_parts})
        
        # chat_history_for_gemini の最後に性格プロンプトを含めないように調整
        # (すでに最初の"user"ロールで追加済みのため)
        final_chat_history = []
        for hist_msg in chat_history_for_gemini:
            # 最初の性格プロンプトと初期応答は履歴として渡す
            if hist_msg["parts"][0]["text"] == current_personality_prompt or hist_msg["parts"][0]["text"] == current_initial_response:
                final_chat_history.append(hist_msg)
            else:
                final_chat_history.append(hist_msg)

        ai_response_parts = [] # AIの返答はテキストと画像のリストになる可能性がある
        generated_image_url = None # 背景設定用URL（最終的に表示される画像）

        try:
            # --- 1. まず通常のテキスト応答を生成 (または画像生成の判断) ---
            # Geminiにチャット履歴を渡し、ユーザーの最新メッセージで応答を生成
            # chat_session = gemini_model.start_chat(history=final_chat_history[:-1]) # 最後のメッセージはsend_messageで送る
            # response = chat_session.send_message(user_input)

            # 会話履歴全体を考慮し、応答を生成
            # image_decision_prompt と通常の会話を統合する
            combined_prompt_content = []
            # 過去の会話をContentオブジェクトに変換して追加
            for msg in final_chat_history:
                # Partオブジェクトに変換して追加
                parts_for_content = []
                for p in msg["parts"]:
                    if "text" in p:
                        parts_for_content.append(p["text"]) # テキストは直接文字列として
                    # 画像データはここではモデルへの入力として考慮しない（判断はテキストで）
                if parts_for_content:
                    combined_prompt_content.append(genai.types.Content(role=msg["role"], parts=parts_for_content))

            # 最新のユーザー入力もContentオブジェクトとして追加
            combined_prompt_content.append(genai.types.Content(role="user", parts=[user_input]))

            with st.spinner("キャラクターが考えてるよ..."):
                response = gemini_model.generate_content(combined_prompt_content)
                
                ai_response_text = ""
                # レスポンスからテキストと画像を抽出
                for part in response.candidates[0].content.parts:
                    if hasattr(part, 'text'):
                        ai_response_text += part.text # テキストを結合
                    elif hasattr(part, 'image') and isinstance(part.image, Image.Image):
                        # 画像があれば、これを表示用として保存
                        buffered = BytesIO()
                        part.image.save(buffered, format="JPEG")
                        img_str = base64.b64encode(buffered.getvalue()).decode()
                        generated_image_url = f"data:image/jpeg;base64,{img_str}"
                        ai_response_parts.append({"mime_type": "image/jpeg", "data": img_str}) # 履歴用

                if ai_response_text:
                    ai_response_parts.insert(0, ai_response_text) # テキストを先頭に追加

                print(f"AI Response Parts: {ai_response_parts}") # デバッグ用ログ
                print(f"Generated Image URL: {generated_image_url}") # デバッグ用ログ

        except Exception as e:
            ai_response_parts = [f"ごめんなさい、お話の途中でエラーが出ちゃったの...: {e}"]
            st.error(ai_response_parts[0])
            generated_image_url = None
            print(f"Overall AI response or image generation error: {e}") # エラーログ

        # AIの返答（テキストと画像が含まれる可能性あり）を履歴に追加
        st.session_state.messages.append({"role": "model", "parts": ai_response_parts})
        
        # --- 生成された画像をセッションステートに保存（背景用） ---
        st.session_state.last_generated_image_url = generated_image_url

# ユーザーからの入力を受け取る部分
st.chat_input("メッセージを入力してね...", on_submit=handle_user_input, key="user_chat_input_key")

# --- 生成された画像と背景の表示ロジック（handle_user_input関数の外） ---
if "last_generated_image_url" in st.session_state and st.session_state.last_generated_image_url:
    generated_image_url_to_display = st.session_state.last_generated_image_url

    st.markdown(
        f"""
        <style>
        body {{
            background-image: url('{generated_image_url_to_display}');
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            transition: background-image 1s ease-in-out;
        }}
        .stApp {{
            background-color: rgba(255, 255, 255, 0.7); 
            border-radius: 10px; 
            padding: 20px; 
        }}
        .stChatMessage {{
            background-color: rgba(255, 255, 255, 0.85); 
            border-radius: 8px;
            padding: 10px;
            margin-bottom: 10px;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
