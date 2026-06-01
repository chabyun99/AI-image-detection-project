import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns

# ==========================================
# 1. 參數與絕對路徑設定 (防呆)
# ==========================================

DATASET_PATH = r"C:\Users\E521-user1\Desktop\CNN\dataset" 
MODEL_PATH = r"C:\Users\E521-user1\Desktop\CNN\my_cnn_model.keras"  

IMG_SIZE = (224, 224)              
BATCH_SIZE = 32

print("\n" + "="*50)
print(f"正在檢查模型檔案: {MODEL_PATH}")
if os.path.exists(MODEL_PATH):
    # 印出檔案修改時間，確保這是我們剛剛訓練出來的那個！
    import time
    mtime = os.path.getmtime(MODEL_PATH)
    print(f"✅ 成功找到模型！最後修改時間: {time.ctime(mtime)}")
else:
    print(f"❌ 嚴重錯誤：找不到模型檔案！請確認 {MODEL_PATH} 是否存在。")
    exit() # 找不到就直接停止，不要浪費時間跑預測
print("="*50 + "\n")

model = load_model(MODEL_PATH)
print("模型載入成功！")

# ==========================================
# 2. 載入測試資料集（Categorical 模式）
# ==========================================
print("正在載入測試資料集...")
test_dataset = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    labels='inferred',
    label_mode='categorical', # 對齊訓練時的設定
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=False  
)

class_names = test_dataset.class_names
print(f"測試集偵測到的類別順序: {class_names}")

# 提取真實標籤 
y_true = []
for images, labels in test_dataset:
    y_true.extend(np.argmax(labels.numpy(), axis=1))
y_true = np.array(y_true)

# ==========================================
# 3. 進行預測與評估
# ==========================================
print("\n開始進行模型預測，這將會使用您剛才訓練出的最新權重...")
y_pred_raw = model.predict(test_dataset) 
y_pred_classes = np.argmax(y_pred_raw, axis=1)
y_pred_probs = y_pred_raw[:, 1]

# --- A. 分類報告 ---
print("\n" + "="*40)
print("【最新成果】分類報告:")
print("="*40)
print(classification_report(y_true, y_pred_classes, target_names=class_names))

# --- B. 混淆矩陣 ---
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix (Brand New Model)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# --- C. ROC 曲線 ---
fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# ==========================================
# 4. 單張圖片逐一判斷
# ==========================================
print("\n" + "="*40)
print("單張圖片個別預測結果分析:")
print("="*40)

for root, dirs, files in os.walk(DATASET_PATH):
    for file in files:
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
            file_path = os.path.join(root, file)
            
            img = tf.keras.utils.load_img(file_path, target_size=IMG_SIZE)
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0)
            
            preds = model.predict(img_array, verbose=0)[0] 
            pred_idx = np.argmax(preds) 
            pred_label = class_names[pred_idx]
            confidence = preds[pred_idx] * 100
                
            print(f"圖片: {file} -> 預測結果: {pred_label} | 信心值: {confidence:.2f}%")