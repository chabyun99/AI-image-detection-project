import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image

# 1. 載入你們的 .keras 模型檔
# 請確保 mobilenetv3_ai_real_best.keras 檔案與此指令碼放在同一個資料夾，或填寫絕對路徑
MODEL_PATH = "mobilenetv3_ai_real_best.keras"
model = tf.keras.models.load_model(MODEL_PATH)


# 2. 定義預測函式
def predict_image(img):
    if img is None:
        return "請上傳圖片"

    # 調整圖片大小至 MobileNet 標準輸入（通常為 224x224）
    img = img.resize((224, 224))

    # 將 PIL 圖片轉換為 Numpy 陣列
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    # 圖片數值歸一化（Preprocessing）
    # 作法 A：如果是直接除以 255 (將像素限制在 0 ~ 1 之間)
    img_array = img_array / 255.0

    # 增加批次維度 (Batch dimension)，從 (224, 224, 3) 變成 (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)

    # 執行模型預測
    # model.predict(img_array) 會回傳如 [[0.7993]] 的二維陣列，後方加 [0] 會變成一維陣列 [0.7993]
    prediction = model.predict(img_array)[0]

    # 印出預測結果方便在終端機（Terminal）檢視除錯
    print("實際的 prediction 內容是:", prediction)

    # 3. 根據模型的輸出層設計調整回傳格式：
    # 輸出層只有 1 個神經元（使用了 Sigmoid，數值靠近 1 是 AI，靠近 0 是真人）
    prob_ai = float(prediction[0])
    prob_real = 1.0 - prob_ai

    # 回傳給 Gradio 顯示的標籤與機率（Gradio 會自動幫你轉成精美的百分比進度條）
    return {"真人圖片 (Real)": prob_real, "AI 生成圖片 (AI Generated)": prob_ai}


# 4. 建立 Gradio 介面
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil", label="請上傳要辨識的圖片"),
    outputs=gr.Label(num_top_classes=2, label="預測分析結果"),
    title="MobileNet AI 圖片偵測系統",
    description="本系統使用 MobileNet 技術，分析輸入圖片究竟是『真人拍攝』還是『AI 繪圖生成』。",
)

# 5. 啟動本機伺服器
if __name__ == "__main__":
    # share=True 會在終端機產生一個臨時的 public URL (例如 xxxxx.gradio.live)`‵‵‵``
    # 任何拿到這個網址的人都可以直接連進來測試你們的模型
    interface.launch(share=True)``