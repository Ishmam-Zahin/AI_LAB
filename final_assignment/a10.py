





import os
import glob
import random
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam





SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

DATA_ROOT = "/home/zahin/Desktop/AI_LAB/datasets/shoe"            
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
TEST_DIR  = os.path.join(DATA_ROOT, "test")

IMG_SIZE = 224
EPOCHS = 5
BATCH_SIZE = 32

LABEL_MAP = {"left": 0, "right": 1}
CLASS_NAMES = {v: k for k, v in LABEL_MAP.items()}





def _collect_paths_for_split(split_dir):
    paths = []
    labels = []
    for cls_name, lbl in LABEL_MAP.items():
        cls_dir = os.path.join(split_dir, cls_name)
        if not os.path.isdir(cls_dir):
            print(f"Warning: missing folder {cls_dir} (skipping)")
            continue
        for ext in ("jpg","jpeg","png","bmp"):
            paths.extend(glob.glob(os.path.join(cls_dir, f"*.{ext}")))
        labels.extend([lbl] * len(glob.glob(os.path.join(cls_dir, "*.*"))))
    paths = sorted(paths)
    
    labels = []
    for p in paths:
        parent = os.path.basename(os.path.dirname(p)).lower()
        labels.append(LABEL_MAP.get(parent, -1))
    return paths, np.array(labels, dtype=np.int32)





def load_images_from_paths(paths, expected_size=IMG_SIZE):
    imgs = []
    valid_idx = []
    for i, p in enumerate(paths):
        try:
            im = Image.open(p).convert("RGB")
            if im.size != (expected_size, expected_size):
                print(f"Warning: {p} has size {im.size} - skipping (expected {expected_size}x{expected_size})")
                continue
            arr = np.array(im, dtype=np.uint8)
            imgs.append(arr)
            valid_idx.append(i)
        except Exception as e:
            print(f"Warning: failed to load {p}: {e}")
    if len(imgs) == 0:
        return np.empty((0, expected_size, expected_size, 3), dtype=np.uint8), np.array([], dtype=np.int32), []
    return np.stack(imgs, axis=0), valid_idx





def prepare_split(split_dir):
    paths, labels_all = _collect_paths_for_split(split_dir)
    x_np, valid_idx = load_images_from_paths(paths, expected_size=IMG_SIZE)
    if x_np.shape[0] == 0:
        raise ValueError(f"No valid images found in {split_dir}.")
    y_np = labels_all[valid_idx]
    
    x_display = x_np.astype(np.float32) / 255.0
    
    x_pre = preprocess_input(x_np.astype(np.float32))
    return x_pre, x_display, y_np





print("Loading training images...")
x_train_pre_raw, x_train_display_raw, y_train_raw = prepare_split(TRAIN_DIR)
print("Loading test images...")
x_test_pre, x_test_display, y_test = prepare_split(TEST_DIR)

print("Raw counts:")
print(" Train:", x_train_pre_raw.shape, y_train_raw.shape, " Test:", x_test_pre.shape, y_test.shape)






perm = np.random.permutation(len(x_train_pre_raw))
x_train_pre_raw = x_train_pre_raw[perm]
x_train_display_raw = x_train_display_raw[perm]
y_train_raw = y_train_raw[perm]

val_idx = int(0.8 * len(x_train_pre_raw))
x_train_pre = x_train_pre_raw[:val_idx]
y_train = y_train_raw[:val_idx]
x_val_pre = x_train_pre_raw[val_idx:]
y_val = y_train_raw[val_idx:]

print("Final dataset shapes:")
print(" Train:", x_train_pre.shape, y_train.shape)
print(" Val:  ", x_val_pre.shape,   y_val.shape)
print(" Test: ", x_test_pre.shape,  y_test.shape)





def build_vgg16_binary(img_size=IMG_SIZE, fine_tune_type="partial", lr=1e-4):
    base = VGG16(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
    if fine_tune_type == "partial":
        
        for layer in base.layers[:-4]:
            layer.trainable = False
        for layer in base.layers[-4:]:
            layer.trainable = True
    elif fine_tune_type == "whole":
        for layer in base.layers:
            layer.trainable = True
    else:
        raise ValueError("fine_tune_type must be 'partial' or 'whole'")

    x = Flatten()(base.output)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=base.input, outputs=out)
    model.compile(optimizer=Adam(learning_rate=lr),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model





print("\nBuilding PARTIAL fine-tune model (most base layers frozen)...")
model_partial = build_vgg16_binary(fine_tune_type="partial")
model_partial.summary(show_trainable=True)





history_partial = model_partial.fit(
    x_train_pre, y_train,
    validation_data=(x_val_pre, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)





print("\nBuilding WHOLE fine-tune model (all base layers trainable)...")
model_whole = build_vgg16_binary(fine_tune_type="whole")
model_whole.summary(show_trainable=True)





history_whole = model_whole.fit(
    x_train_pre, y_train,
    validation_data=(x_val_pre, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=1
)





def plot_history(hist, title_prefix="Model"):
    h = hist.history
    epochs = range(1, len(h['loss']) + 1)
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, h['loss'], label='train loss')
    plt.plot(epochs, h['val_loss'], label='val loss')
    plt.title(f"{title_prefix} loss")
    plt.xlabel('epoch'); plt.legend()
    plt.subplot(1,2,2)
    
    acc_key = 'accuracy' if 'accuracy' in h else 'acc'
    val_acc_key = 'val_' + acc_key
    plt.plot(epochs, h[acc_key], label='train acc')
    plt.plot(epochs, h[val_acc_key], label='val acc')
    plt.title(f"{title_prefix} accuracy")
    plt.xlabel('epoch'); plt.legend()
    plt.tight_layout()
    plt.show()

plot_history(history_partial, "Partial fine-tune")
plot_history(history_whole, "Whole fine-tune")





loss_p, acc_p = model_partial.evaluate(x_test_pre, y_test, verbose=0)
loss_w, acc_w = model_whole.evaluate(x_test_pre, y_test, verbose=0)
print("\nTest evaluation:")
print(f" Partial  -> loss: {loss_p:.4f}, acc: {acc_p:.4f}")
print(f" Whole    -> loss: {loss_w:.4f}, acc: {acc_w:.4f}")




n_display = 5
rng = np.random.default_rng(SEED)
n_test = len(x_test_pre)
if n_test == 0:
    print("No test images to display.")
else:
    n_display = min(n_display, n_test)
    chosen = rng.choice(n_test, size=n_display, replace=False)

    plt.figure(figsize=(18, 7))
    for i, idx in enumerate(chosen):
        img_disp = x_test_display[idx]        
        true_label = int(y_test[idx])

        p_partial = float(model_partial.predict(x_test_pre[idx:idx+1], verbose=0)[0][0])
        p_whole   = float(model_whole.predict(x_test_pre[idx:idx+1], verbose=0)[0][0])

        pred_partial = int(p_partial > 0.5)
        pred_whole   = int(p_whole > 0.5)

        ax1 = plt.subplot(2, n_display, 1 + i)
        ax1.imshow(img_disp)
        ax1.axis('off')
        ax1.set_title(f"True: {CLASS_NAMES[true_label]}\nPartial: {CLASS_NAMES[pred_partial]}\n(p={p_partial:.2f})")

        ax2 = plt.subplot(2, n_display, 1 + n_display + i)
        ax2.imshow(img_disp)
        ax2.axis('off')
        ax2.set_title(f"True: {CLASS_NAMES[true_label]}\nWhole: {CLASS_NAMES[pred_whole]}\n(p={p_whole:.2f})")

    plt.tight_layout()
    plt.show()

