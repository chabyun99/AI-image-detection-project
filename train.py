import tensorflow as tf
import matplotlib.pyplot as plt
import os

# ==========================================
# 0. 檢查 GPU 資源
# ==========================================
print("=" * 50)
print("系統硬體檢查中...")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"✅ 成功偵測到 GPU ({len(gpus)} 個):")
    for gpu in gpus:
        print(f"   - {gpu.name}")
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ 已啟用 GPU 記憶體動態分配 (Memory Growth)")
    except RuntimeError as e:
        print(e)
else:
    print("⚠️ 未偵測到 GPU！模型將使用 CPU 進行訓練，這可能會花費較長的時間。")
print("=" * 50 + "\n")

# ==========================================
# 1. 參數設定與資料路徑
# ==========================================
DATA_DIR = r"C:\Users\E521-user1\Desktop\CNN\train_dataset"
MODEL_SAVE_PATH = "my_cnn_model.keras"

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
# ⚠️ 為了讓 Early Stopping 發揮作用，我們把 Epochs 拉長，給模型更多時間尋找最佳解
EPOCHS = 40 

# ==========================================
# 2. 載入與切割資料集 (80%訓練, 20%驗證)
# ==========================================
print("正在載入訓練集...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2, 
    subset="training",    
    seed=123,             
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='categorical' 
)

print("正在載入驗證集...")
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",  
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

class_names = train_ds.class_names
num_classes = len(class_names)
print(f"偵測到的類別 ({num_classes}個): {class_names}\n")

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# ==========================================
# 3. 建立 CNN 模型 (抗過擬合增強版)
# ==========================================

# 🛡️ 新增防線一：資料增強 (Data Augmentation)
# 每次訓練時，隨機對圖片進行翻轉、旋轉、縮放，逼迫模型不能死背圖片
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    tf.keras.layers.RandomRotation(0.1), # 隨機旋轉 10%
    tf.keras.layers.RandomZoom(0.1),     # 隨機放大縮小 10%
    tf.keras.layers.RandomContrast(0.1)  # 隨機改變對比度，破除模型對亮度的依賴
], name="data_augmentation")

model = tf.keras.Sequential([
    # 第一步：先進行資料增強
    data_augmentation,
    
    # 第二步：原始縮放層
    tf.keras.layers.Rescaling(1./255),
    
    # 第一層卷積
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    
    # 第二層卷積
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    
    # 第三層卷積
    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.MaxPooling2D(),
    
    # 全連接層
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    # 加大 Dropout 比例到 0.6，進一步防止神經元死背答案
    tf.keras.layers.Dropout(0.6), 
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

opt = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ==========================================
# 4. 設定回調函數 (Callback) 
# ==========================================
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=MODEL_SAVE_PATH,
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# 🛡️ 新增防線二：提早停損 (Early Stopping)
# 如果連續 8 個 Epoch 驗證集的 Loss 都沒有下降，就強制停止訓練，並還原到最好的那一刻
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True,
    verbose=1
)

# ==========================================
# 5. 開始訓練模型
# ==========================================
print("\n開始訓練模型...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint, early_stopping] # 同時啟用儲存與停損機制
)
print("\n訓練結束！最佳模型已儲存為:", MODEL_SAVE_PATH)

# ==========================================
# 6. 繪製 Accuracy 與 Loss 曲線
# ==========================================
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(14, 5))

# Accuracy 曲線
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy', linewidth=2)
plt.plot(epochs_range, val_acc, label='Validation Accuracy', linewidth=2)
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.grid(alpha=0.3)

# Loss 曲線
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss', linewidth=2)
plt.plot(epochs_range, val_loss, label='Validation Loss', linewidth=2)
plt.legend(loc='upper right')
plt.title('Training and Validation Loss', fontsize=14)
plt.xlabel('Epochs', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()