# =================================================================
# 라이브러리 import
# =================================================================
import test_tensorflow as tf
from test_tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# =================================================================
# 1. 데이터 로드 및 3단계 분리 (Train, Validation, Test)
# =================================================================
# Fashion MNIST 데이터셋 로드
fashion_mnist = keras.datasets.fashion_mnist
(train_val_images, train_val_labels), (test_images_orig, test_labels_orig) = (
    fashion_mnist.load_data()
)

# 클래스 이름 정의
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

# 로드된 전체 데이터를 합쳐서 다시 분배함 (총 70,000개)
all_images = np.concatenate([train_val_images, test_images_orig])
all_labels = np.concatenate([train_val_labels, test_labels_orig])

# 훈련셋(60%)과 테스트셋(40%)으로 먼저 나눔
train_val_images, test_images, train_val_labels, test_labels = train_test_split(
    all_images, all_labels, test_size=0.4, random_state=42
)

# 남은 훈련셋(60%)을 다시 훈련셋(50%)과 검증셋(10%)으로 나눔
train_images, val_images, train_labels, val_labels = train_test_split(
    train_val_images, train_val_labels, test_size=0.16666, random_state=42
)  # 0.16666 * 0.6 = 0.1

# =================================================================
# 2. 데이터 전처리
# =================================================================
# 픽셀 값을 0~1 사이로 정규화 및 채널 차원 추가
train_images = np.expand_dims(train_images / 255.0, -1)
val_images = np.expand_dims(val_images / 255.0, -1)
test_images = np.expand_dims(test_images / 255.0, -1)


# =================================================================
# 3. 테스트셋 평가를 위한 커스텀 콜백 정의
# =================================================================
class TestMetricsCallback(keras.callbacks.Callback):
    def __init__(self, test_data):
        self.test_data = test_data
        self.test_accs = []
        self.test_losses = []

    def on_epoch_end(self, epoch, logs=None):
        x_test, y_test = self.test_data
        loss, acc = self.model.evaluate(x_test, y_test, verbose=0)
        self.test_losses.append(loss)
        self.test_accs.append(acc)
        # 훈련 로그에 테스트 결과를 추가하여 출력
        print(f" — test_loss: {loss:.4f} — test_acc: {acc:.4f}")


# 콜백 인스턴스 생성
test_callback = TestMetricsCallback((test_images, test_labels))

# =================================================================
# 4. CNN 모델 구축 및 컴파일
# =================================================================
# model = keras.Sequential(
#     [
#         keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
#         keras.layers.MaxPooling2D((2, 2)),
#         keras.layers.Conv2D(64, (3, 3), activation="relu"),
#         keras.layers.MaxPooling2D((2, 2)),
#         keras.layers.Flatten(),
#         keras.layers.Dense(128, activation="relu"),
#         keras.layers.Dense(10, activation="softmax"),
#     ]
# )
# model.compile(
#     optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
# )
# 데이터 증강을 위한 레이어들
data_augmentation = keras.Sequential(
    [
        keras.layers.RandomFlip("horizontal", input_shape=(28, 28, 1)),
        keras.layers.RandomRotation(0.1),
        keras.layers.RandomZoom(0.1),
    ]
)

model = keras.Sequential(
    [
        data_augmentation, # <--- 모델의 맨 앞에 추가
        keras.layers.Conv2D(32, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation="relu"),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(10, activation="softmax"),
    ]
)
model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
# =================================================================
# 5. 모델 학습 (커스텀 콜백 사용)
# =================================================================
# print("--------- 모델 학습 시작 ---------")
# history = model.fit(
#     train_images,
#     train_labels,
#     epochs=10,
#     validation_data=(val_images, val_labels),
#     callbacks=[test_callback],
# )
# print("--------- 모델 학습 종료 ---------")
# EarlyStopping 콜백 정의
# monitor='val_loss': 검증 손실을 기준으로 멈출지 결정
# patience=3: 3 에포크 동안 성능 개선이 없으면 훈련 중단
# restore_best_weights=True: 훈련이 멈췄을 때 가장 성능이 좋았던 시점의 가중치로 복원
early_stopping_cb = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True
)

print("--------- 모델 학습 시작 ---------")
history = model.fit(
    train_images,
    train_labels,
    epochs=30,  # <-- epochs를 충분히 길게 설정해도 조기 종료가 제어해 줌
    validation_data=(val_images, val_labels),
    callbacks=[test_callback, early_stopping_cb], # <--- 여기에 추가
)
print("--------- 모델 학습 종료 ---------")

# =================================================================
# 6. 3개 데이터셋의 정확도 및 손실 그래프 시각화
# =================================================================
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
test_acc = test_callback.test_accs
loss = history.history["loss"]
val_loss = history.history["val_loss"]
test_loss = test_callback.test_losses

epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))

# 정확도 그래프
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.plot(epochs_range, test_acc, label="Test Accuracy")
plt.legend(loc="lower right")
plt.title("Training, Validation, and Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

# 손실 그래프
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.plot(epochs_range, test_loss, label="Test Loss")
plt.legend(loc="upper right")
plt.title("Training, Validation, and Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# =================================================================
# 7. 예측 결과 시각화
# =================================================================
# 테스트셋에 대한 예측 수행
predictions = model.predict(test_images)


# 이미지와 예측 결과를 함께 보여주는 함수
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img.squeeze(), cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    color = "blue" if predicted_label == true_label else "red"
    plt.xlabel(
        f"Predicted: {class_names[predicted_label]} ({100*np.max(predictions_array):.2f}%)\nTrue: {class_names[true_label]}",
        color=color,
    )


# 예측 확률을 막대 그래프로 보여주는 함수
def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10), class_names, rotation=90)
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    thisplot[predicted_label].set_color("red")
    thisplot[true_label].set_color("blue")


# 처음 15개의 테스트 이미지와 예측 결과를 출력
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2, 2 * i + 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2, 2 * i + 2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()
