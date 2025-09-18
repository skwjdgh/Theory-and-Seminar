
# =================================================================
# 라이브러리 import
# =================================================================
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras_tuner as kt

# =================================================================
# 1. 데이터 로드 및 3단계 분리
# =================================================================
fashion_mnist = keras.datasets.fashion_mnist
(train_val_images, train_val_labels), (test_images_orig, test_labels_orig) = (
    fashion_mnist.load_data()
)
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
all_images = np.concatenate([train_val_images, test_images_orig])
all_labels = np.concatenate([train_val_labels, test_labels_orig])
train_val_images, test_images, train_val_labels, test_labels = train_test_split(
    all_images, all_labels, test_size=0.4, random_state=42
)
train_images, val_images, train_labels, val_labels = train_test_split(
    train_val_images, train_val_labels, test_size=0.16666, random_state=42
)

# =================================================================
# 2. 데이터 전처리
# =================================================================
train_images = np.expand_dims(train_images / 255.0, -1).astype('float32')
val_images = np.expand_dims(val_images / 255.0, -1).astype('float32')
test_images = np.expand_dims(test_images / 255.0, -1).astype('float32')


# =================================================================
# 3. tf.data 파이프라인 설정
# =================================================================
BATCH_SIZE = 128
AUTOTUNE = tf.data.AUTOTUNE

data_augmentation = keras.Sequential(
    [
        keras.layers.RandomFlip("horizontal", input_shape=(28, 28, 1)),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
    ]
)

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.cache()
train_dataset = train_dataset.shuffle(buffer_size=1024)
train_dataset = train_dataset.batch(BATCH_SIZE)
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y), num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
val_dataset = val_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.cache()
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.cache()
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)


# =================================================================
# 4. 커스텀 콜백 정의
# =================================================================
class TestMetricsCallback(keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        self.test_accs = []
        self.test_losses = []

    def on_epoch_end(self, epoch, logs=None):
        loss, acc = self.model.evaluate(self.test_data, verbose=0)
        self.test_losses.append(loss)
        self.test_accs.append(acc)
        print(f" — test_loss: {loss:.4f} — test_acc: {acc:.4f}")

test_callback = TestMetricsCallback(test_dataset)


# =================================================================
# 5. [개선] L2 규제를 추가한 하이퍼모델 정의
# =================================================================
def build_model(hp):
    """L2 규제를 추가하고 탐색 공간을 조정한 하이퍼모델"""
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(28, 28, 1)))
    
    # L2 규제 강도를 하이퍼파라미터로 추가
    hp_l2 = hp.Choice('l2_reg', values=[1e-3, 1e-4])

    # === 컨볼루션 블록 ===
    model.add(keras.layers.Conv2D(
        filters=hp.Int('filters_1', min_value=32, max_value=64, step=16),
        kernel_size=3, padding='same',
        kernel_regularizer=keras.regularizers.l2(hp_l2) # L2 규제 적용
    ))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))

    # 모델 복잡도를 제한하여 탐색 시간 단축
    for i in range(hp.Int('num_conv_blocks', 1, 2)):
        model.add(keras.layers.Conv2D(
            filters=hp.Int(f'filters_{i+2}', min_value=64, max_value=128, step=32),
            kernel_size=3, padding='same',
            kernel_regularizer=keras.regularizers.l2(hp_l2) # L2 규제 적용
        ))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Activation('relu'))
    
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(hp.Float('dropout_1', 0.25, 0.4, step=0.05)))

    # === 분류기 블록 ===
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(
        units=hp.Int('dense_units', min_value=256, max_value=512, step=128),
        kernel_regularizer=keras.regularizers.l2(hp_l2) # L2 규제 적용
    ))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Activation('relu'))
    model.add(keras.layers.Dropout(hp.Float('dropout_2', 0.4, 0.5, step=0.1))) # 드롭아웃 소폭 강화
    model.add(keras.layers.Dense(10, activation='softmax'))

    # === 컴파일 ===
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-4])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


# =================================================================
# 6. [개선] 속도 최적화된 Hyperband 튜너 설정
# =================================================================
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10, # <-- 25에서 10으로 줄여 탐색 시간 대폭 단축
    factor=3,
    directory='hyperband_dir_fast_l2', # 새 프로젝트 디렉토리
    project_name='fashion_mnist_fast_tuning'
)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3) # patience 소폭 감소

print("--------- 하이퍼파라미터 탐색 시작 ---------")
tuner.search(
    train_dataset, 
    epochs=10, # max_epochs와 동일하게 설정
    validation_data=val_dataset,
    callbacks=[stop_early]
)
print("--------- 하이퍼파라미터 탐색 종료 ---------")

print("\n--- 최적 하이퍼파라미터 ---")
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
for param, value in best_hps.values.items():
    print(f"{param}: {value}")
print("---------------------------\n")


# =================================================================
# 7. 최적 모델로 최종 학습 진행
# =================================================================
model = tuner.hypermodel.build(best_hps)
model.summary()

early_stopping_cb = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=15, restore_best_weights=True, verbose=1
)
reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=1
)

print("--------- 최적 모델 학습 시작 ---------")
history = model.fit(
    train_dataset,
    epochs=150,
    validation_data=val_dataset,
    callbacks=[test_callback, early_stopping_cb, reduce_lr_cb]
)
print("--------- 최적 모델 학습 종료 ---------")


# =================================================================
# 8. 3개 데이터셋의 정확도 및 손실 그래프 시각화
# =================================================================
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
test_acc = test_callback.test_accs
loss = history.history["loss"]
val_loss = history.history["val_loss"]
test_loss = test_callback.test_losses

epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy", marker='.')
plt.plot(epochs_range, val_acc, label="Validation Accuracy", marker='.')
plt.plot(epochs_range, test_acc, label="Test Accuracy", marker='.')
plt.legend(loc="lower right")
plt.title("Training, Validation, and Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss", marker='.')
plt.plot(epochs_range, val_loss, label="Validation Loss", marker='.')
plt.plot(epochs_range, test_loss, label="Test Loss", marker='.')
plt.legend(loc="upper right")
plt.title("Training, Validation, and Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()


# =================================================================
# 9. 예측 결과 시각화
# =================================================================
predictions = model.predict(test_dataset)

test_images_vis = []
test_labels_vis = []
for image, label in test_dataset.unbatch():
    test_images_vis.append(image.numpy())
    test_labels_vis.append(label.numpy())
test_images_vis = np.array(test_images_vis)
test_labels_vis = np.array(test_labels_vis)

def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False); plt.xticks([]); plt.yticks([])
    plt.imshow(img.squeeze(), cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    color = "blue" if predicted_label == true_label else "red"
    plt.xlabel(
        f"Predicted: {class_names[predicted_label]} ({100*np.max(predictions_array):.2f}%)\nTrue: {class_names[true_label]}",
        color=color,
    )

def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False); plt.xticks(range(10), class_names, rotation=90); plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color("red")
    thisplot[true_label].set_color("blue")

num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2, 2 * i + 1)
    plot_image(i, predictions[i], test_labels_vis, test_images_vis)
    plt.subplot(num_rows, 2, 2 * i + 2)
    plot_value_array(i, predictions[i], test_labels_vis)
plt.tight_layout()
plt.show()