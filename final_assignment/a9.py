





import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import VGG16, ResNet50, InceptionV3
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_pre
from tensorflow.keras.applications.resnet50 import preprocess_input as res_pre
from tensorflow.keras.applications.inception_v3 import preprocess_input as inc_pre
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import cv2





IMG_PATH = '/home/zahin/Desktop/AI_LAB/datasets/shoe/test/left/left_Iphone11Pro_299x299_0587.jpg'





def load_and_prepare(img_path, target_size, preprocess_fn):
    """Load image, resize, expand batch, preprocess"""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = cv2.resize(img, target_size)
    img = img.astype("float32")

    
    img_batch = np.expand_dims(img, axis=0)

    img_batch = preprocess_fn(img_batch)
    return img_batch





def get_conv_layers(model, max_layers=10):
    """Return up to max_layers Conv2D layers"""
    conv_layers = [l for l in model.layers if isinstance(l, tf.keras.layers.Conv2D)]
    return conv_layers[:max_layers]





def visualize_feature_maps(model_name, model, img_batch):
    """Extract and plot 100 feature maps (10 layers × 10 channels)"""

    conv_layers = get_conv_layers(model, 10)

    
    outputs = [layer.output for layer in conv_layers]
    feature_model = Model(inputs=model.input, outputs=outputs)

    feature_maps = feature_model.predict(img_batch)

    
    fig, axes = plt.subplots(10, 10, figsize=(12, 12))
    fig.suptitle(f"{model_name} Feature Maps", fontsize=16)

    plot_idx = 0

    for layer_map in feature_maps:
        if plot_idx >= 10:
            break

        fmap = layer_map[0]  

        channels = min(10, fmap.shape[-1])

        for ch in range(channels):
            ax = axes[plot_idx, ch]
            ax.imshow(fmap[:, :, ch], cmap='viridis')
            ax.axis("off")

        
        for ch in range(channels, 10):
            axes[plot_idx, ch].axis("off")

        plot_idx += 1

    
    for r in range(plot_idx, 10):
        for c in range(10):
            axes[r, c].axis("off")

    plt.tight_layout()
    plt.show()






vgg = VGG16(weights="imagenet", include_top=False)
vgg_img = load_and_prepare(IMG_PATH, (224, 224), vgg_pre)
visualize_feature_maps("VGG16", vgg, vgg_img)


resnet = ResNet50(weights="imagenet", include_top=False)
res_img = load_and_prepare(IMG_PATH, (224, 224), res_pre)
visualize_feature_maps("ResNet50", resnet, res_img)


inception = InceptionV3(weights="imagenet", include_top=False)
inc_img = load_and_prepare(IMG_PATH, (299, 299), inc_pre)
visualize_feature_maps("InceptionV3", inception, inc_img)







