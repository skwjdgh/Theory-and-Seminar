# =================================================================
# 라이브러리 import
# =================================================================
import test_tensorflow as tf
from test_tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras_tuner as kt # 1. 하이퍼파라미터 튜닝을 위한 라이브러리

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

# 데이터셋 통합 후 재분할
all_images = np.concatenate([train_val_images, test_images_orig])
all_labels = np.concatenate([train_val_labels, test_labels_orig])

train_val_images, test_images, train_val_labels, test_labels = train_test_split(
    all_images, all_labels, test_size=0.2, random_state=42 # 테스트셋 비율 20%로 조정
)
train_images, val_images, train_labels, val_labels = train_test_split(
    train_val_images, train_val_labels, test_size=0.2, random_state=42 # 훈련셋 중 20%를 검증셋으로 사용
)

# =================================================================
# 2. 데이터 전처리
# =================================================================
# (1) 정규화 + 채널 확장
def preprocess_data(images):
    images = np.expand_dims(images.astype("float32") / 255.0, -1)
    return images

train_images = preprocess_data(train_images)
val_images = preprocess_data(val_images)
test_images = preprocess_data(test_images)

# (2) 데이터 증강
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
datagen.fit(train_images)

# =================================================================
# 3. 개선점 1: KerasTuner를 사용한 하이퍼파라미터 탐색 모델 정의
# =================================================================
def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(28, 28, 1)))

    # 튜닝 대상 1: Conv 레이어의 필터 수
    hp_filters1 = hp.Int('filters_1', min_value=32, max_value=64, step=32)
    hp_filters2 = hp.Int('filters_2', min_value=64, max_value=128, step=64)

    # 튜닝 대상 2: 드롭아웃 비율
    hp_dropout1 = hp.Float('dropout_1', min_value=0.2, max_value=0.4, step=0.1)
    hp_dropout2 = hp.Float('dropout_2', min_value=0.2, max_value=0.5, step=0.1)

    model.add(keras.layers.Conv2D(hp_filters1, (3,3), padding="same", activation="relu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(hp_filters1, (3,3), padding="same", activation="relu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((2,2)))
    model.add(keras.layers.Dropout(hp_dropout1))

    model.add(keras.layers.Conv2D(hp_filters2, (3,3), padding="same", activation="relu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Conv2D(hp_filters2, (3,3), padding="same", activation="relu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.MaxPooling2D((2,2)))
    model.add(keras.layers.Dropout(hp_dropout2))

    model.add(keras.layers.Flatten())

    # 튜닝 대상 3: Dense 레이어의 유닛 수
    hp_units = hp.Int('units', min_value=128, max_value=256, step=128)
    model.add(keras.layers.Dense(hp_units, activation="relu"))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(10, activation="softmax"))

    # 튜닝 대상 4: 학습률 (Learning Rate)
    # CosineDecay는 별도 적용하므로 여기서는 초기 학습률만 튜닝
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-4, 3e-4])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# KerasTuner 설정 및 실행
tuner = kt.Hyperband(
    build_model,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='keras_tuner_dir',
    project_name='fashion_mnist_tuning'
)

# 튜닝 전 기존 로그 삭제
# !rm -rf ./keras_tuner_dir/

early_stopping_tuner = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

print("\n--------- 하이퍼파라미터 튜닝 시작 ---------")
tuner.search(
    train_images, train_labels,
    epochs=20, # 각 모델이 최대 20 epoch까지 학습
    validation_data=(val_images, val_labels),
    callbacks=[early_stopping_tuner]
)
print("\n--------- 하이퍼파라미터 튜닝 종료 ---------")


# 최적의 하이퍼파라미터 확인
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print("\n--- 최적 하이퍼파라미터 ---")
print(f"Filters 1: {best_hps.get('filters_1')}")
print(f"Filters 2: {best_hps.get('filters_2')}")
print(f"Dropout 1: {best_hps.get('dropout_1'):.2f}")
print(f"Dropout 2: {best_hps.get('dropout_2'):.2f}")
print(f"Dense Units: {best_hps.get('units')}")
print(f"Learning Rate: {best_hps.get('learning_rate')}")
print("-------------------------\n")


# =================================================================
# 4. 최적 모델 빌드 및 학습 전략 개선
# =================================================================
# 개선점 2: CosineDecay 학습 스케줄러 적용
# 개선점 3: Test set 평가 콜백 제거
# -----------------------------------------------------------------

# 최적 하이퍼파라미터로 새 모델 빌드
final_model = tuner.hypermodel.build(best_hps)

# CosineDecay 스케줄러 설정
EPOCHS = 50
BATCH_SIZE = 64
steps_per_epoch = len(train_images) // BATCH_SIZE
decay_steps = steps_per_epoch * EPOCHS

# best_hps에서 찾은 초기 학습률을 CosineDecay에 적용
initial_learning_rate = best_hps.get('learning_rate')
cosine_decay_scheduler = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=initial_learning_rate,
    decay_steps=decay_steps,
    alpha=0.0 # 학습 마지막에 도달할 최소 학습률
)

# 스케줄러를 적용하여 모델 재컴파일
final_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=cosine_decay_scheduler),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# 콜백 설정 (TestMetricsCallback 제거)
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10, # 튜닝된 모델이므로 조금 더 길게 관찰
    restore_best_weights=True
)

print("\n--------- 최적 모델 학습 시작 ---------")
history = final_model.fit(
    datagen.flow(train_images, train_labels, batch_size=BATCH_SIZE),
    epochs=EPOCHS,
    validation_data=(val_images, val_labels),
    callbacks=[early_stopping],
    verbose=1
)
print("--------- 최적 모델 학습 종료 ---------")


# =================================================================
# 5. 최종 평가 및 시각화 (개선점 3: Test Set 분리 평가)
# =================================================================
print("\n--- 단일 모델 최종 성능 평가 (Test Set) ---")
test_loss, test_acc = final_model.evaluate(test_images, test_labels, verbose=2)
print(f"최종 테스트 손실: {test_loss:.4f}")
print(f"최종 테스트 정확도: {test_acc:.4f}")
print("------------------------------------------")


# 학습 과정 시각화 (Train/Validation만)
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs_range = range(len(acc))

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title(f"Training and Validation Accuracy\nFinal Test Acc: {test_acc:.3f}")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title(f"Training and Validation Loss\nFinal Test Loss: {test_loss:.3f}")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


# =================================================================
# 6. 개선점 4: 모델 앙상블 (Model Ensemble)
# =================================================================
NUM_ENSEMBLE_MODELS = 5
predictions_list = []
ensemble_models = []

print(f"\n--------- {NUM_ENSEMBLE_MODELS}개 모델 앙상블 시작 ---------")
for i in range(NUM_ENSEMBLE_MODELS):
    print(f"\n--- 앙상블 모델 {i+1}/{NUM_ENSEMBLE_MODELS} 학습 ---")
    # 매번 새로운 모델을 최적의 하이퍼파라미터로 빌드 및 컴파일
    model = tuner.hypermodel.build(best_hps)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cosine_decay_scheduler),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    # 데이터 증강을 사용하므로 매번 다른 학습 결과를 보임
    model.fit(
        datagen.flow(train_images, train_labels, batch_size=BATCH_SIZE),
        epochs=EPOCHS, # 동일한 epoch 수로 학습
        validation_data=(val_images, val_labels),
        callbacks=[early_stopping],
        verbose=0 # 로그는 간결하게 출력하지 않음
    )
    ensemble_models.append(model)
    
    # 테스트셋에 대한 예측 수행
    preds = model.predict(test_images)
    predictions_list.append(preds)
    print(f"모델 {i+1} 학습 및 예측 완료")

# 앙상블 예측 (Soft Voting)
# 각 모델의 예측 확률을 평균
avg_predictions = np.mean(np.array(predictions_list), axis=0)

# 평균 확률에서 가장 높은 클래스 선택
ensemble_pred_labels = np.argmax(avg_predictions, axis=1)

# 앙상블 정확도 계산
ensemble_accuracy = np.mean(ensemble_pred_labels == test_labels)

print("\n--------- 앙상블 최종 결과 ---------")
print(f"단일 모델 테스트 정확도: {test_acc:.4f}")
print(f"앙상블 모델 테스트 정확도: {ensemble_accuracy:.4f}")
print("-----------------------------------")