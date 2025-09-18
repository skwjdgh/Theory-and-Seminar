# =================================================================
# 라이브러리 import
# =================================================================
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras_tuner as kt
import os

# Docker 환경에서 Matplotlib 백엔드 설정 (그래프 파일 저장을 위함)
if os.environ.get('DEBIAN_FRONTEND') == 'noninteractive':
    import matplotlib
    matplotlib.use('Agg')

print("TensorFlow Version:", tf.__version__)
print("KerasTuner Version:", kt.__version__)

# =================================================================
# 1. 데이터 로드 및 3단계 분리 (Train, Validation, Test)
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
    all_images, all_labels, test_size=0.2, random_state=42
)
train_images, val_images, train_labels, val_labels = train_test_split(
    train_val_images, train_val_labels, test_size=0.2, random_state=42
)

# =================================================================
# 2. 데이터 전처리
# =================================================================
def preprocess_data(images):
    images = np.expand_dims(images.astype("float32") / 255.0, -1)
    return images
train_images = preprocess_data(train_images)
val_images = preprocess_data(val_images)
test_images = preprocess_data(test_images)

datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
    horizontal_flip=True, zoom_range=0.1
)
datagen.fit(train_images)

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
        print(f"\r — test_loss: {loss:.4f} — test_acc: {acc:.4f}", end="")

# =================================================================
# 4. CNN 모델 구축 (튜닝용)
# =================================================================
def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(28, 28, 1)))
    hp_filters1 = hp.Int('filters_1', min_value=32, max_value=64, step=32)
    hp_filters2 = hp.Int('filters_2', min_value=64, max_value=128, step=64)
    hp_dropout1 = hp.Float('dropout_1', min_value=0.2, max_value=0.4, step=0.1)
    hp_dropout2 = hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.1)
    hp_units = hp.Int('units', min_value=128, max_value=256, step=128)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-4, 3e-4])
    # [개선] L2 규제 하이퍼파라미터 추가
    hp_l2_reg = hp.Choice('l2_regularization', values=[1e-4, 1e-5])

    model.add(keras.layers.Conv2D(hp_filters1, (3,3), padding="same", activation="relu",
                                  kernel_regularizer=keras.regularizers.l2(hp_l2_reg)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((2,2)))
    model.add(keras.layers.Dropout(hp_dropout1))

    model.add(keras.layers.Conv2D(hp_filters2, (3,3), padding="same", activation="relu",
                                  kernel_regularizer=keras.regularizers.l2(hp_l2_reg)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((2,2)))
    model.add(keras.layers.Dropout(hp_dropout2))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(hp_units, activation="relu",
                                 kernel_regularizer=keras.regularizers.l2(hp_l2_reg)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# =================================================================
# 5. 하이퍼파라미터 튜닝 실행
# =================================================================
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=15,
    executions_per_trial=1,
    directory='keras_tuner_dir',
    project_name='fashion_mnist_tuning_final'
)
early_stopping_tuner = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
print("\n--------- 1단계: 하이퍼파라미터 튜닝 시작 (고속 버전) ---------")
tuner.search(
    train_images, train_labels,
    epochs=10,
    validation_data=(val_images, val_labels),
    callbacks=[early_stopping_tuner]
)
print("\n--------- 1단계: 하이퍼파라미터 튜닝 종료 ---------")
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("\n--- 최적 하이퍼파라미터 ---")
print(f"Filters 1: {best_hps.get('filters_1')}")
print(f"Filters 2: {best_hps.get('filters_2')}")
print(f"Dense Units: {best_hps.get('units')}")
print(f"Learning Rate: {best_hps.get('learning_rate')}")
print("-------------------------\n")

# =================================================================
# 6. 최적 모델 학습
# =================================================================
final_model = tuner.hypermodel.build(best_hps)
EPOCHS = 30
BATCH_SIZE = 64
steps_per_epoch = len(train_images) // BATCH_SIZE
decay_steps = steps_per_epoch * EPOCHS
initial_learning_rate = best_hps.get('learning_rate')
cosine_decay_scheduler = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=initial_learning_rate, decay_steps=decay_steps, alpha=0.0
)
final_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=cosine_decay_scheduler),
    loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)
test_callback = TestMetricsCallback((test_images, test_labels))
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=7, restore_best_weights=True
)
print("\n--------- 2단계: 최적 모델 학습 시작 ---------")
history = final_model.fit(
    datagen.flow(train_images, train_labels, batch_size=BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=(val_images, val_labels),
    callbacks=[early_stopping, test_callback],
    verbose=1
)
print("\n--------- 2단계: 최적 모델 학습 종료 ---------")

# =================================================================
# 7. 최종 성능 평가
# =================================================================
print("\n--- 최종 성능 평가 (Test Set) ---")
test_loss, test_acc = final_model.evaluate(test_images, test_labels, verbose=2)
print(f"최종 테스트 손실: {test_loss:.4f}")
print(f"최종 테스트 정확도: {test_acc:.4f}")
print("-----------------------------------")

# =================================================================
# 8. 학습 과정 시각화
# =================================================================
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
test_acc_curve = test_callback.test_accs
loss = history.history["loss"]
val_loss = history.history["val_loss"]
test_loss_curve = test_callback.test_losses

epochs_range = range(min(len(acc), len(test_acc_curve)))
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc[:len(epochs_range)], label="Training Accuracy")
plt.plot(epochs_range, val_acc[:len(epochs_range)], label="Validation Accuracy")
plt.plot(epochs_range, test_acc_curve[:len(epochs_range)], label="Test Accuracy", linestyle='--')
plt.legend(loc="lower right")
plt.title("Training, Validation, and Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss[:len(epochs_range)], label="Training Loss")
plt.plot(epochs_range, val_loss[:len(epochs_range)], label="Validation Loss")
plt.plot(epochs_range, test_loss_curve[:len(epochs_range)], label="Test Loss", linestyle='--')
plt.legend(loc="upper right")
plt.title("Training, Validation, and Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")

# [개선] Docker 환경을 위해 그래프를 파일로 저장
plt.savefig('learning_curve.png')
print("\n학습 과정 그래프를 'learning_curve.png' 파일로 저장했습니다.")
plt.show()

# =================================================================
# 9. 최종 모델 예측 결과 시각화
# =================================================================
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False); plt.xticks([]); plt.yticks([])
    plt.imshow(img.squeeze(), cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions_array)
    color = "blue" if predicted_label == true_label else "red"
    plt.xlabel(
        f"Pred: {class_names[predicted_label]} ({100*np.max(predictions_array):.2f}%)\nTrue: {class_names[true_label]}",
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

print("\n최종 모델의 예측 결과를 시각화합니다...")
predictions = final_model.predict(test_images)
num_rows = 5
# [버그 수정] 한 줄에 2개의 subplot(이미지, 막대그래프)이 있으므로, num_cols는 2가 되어야 합니다.
num_cols = 2
num_images = num_rows * num_cols # 총 10개의 이미지 시각화

plt.figure(figsize=(2 * 2 * (num_cols + 1), 2 * num_rows)) # 가로 폭을 조금 더 여유있게 조정
plt.suptitle("최종 모델 예측 결과 (Final Model Predictions)", fontsize=16)
for i in range(num_images):
    plt.subplot(num_rows, 2, 2 * i + 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2, 2 * i + 2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout(rect=[0, 0.03, 1, 0.96])

# [개선] Docker 환경을 위해 예측 결과 그래프도 파일로 저장
plt.savefig('prediction_results.png')
print("예측 결과 그래프를 'prediction_results.png' 파일로 저장했습니다.")
plt.show()