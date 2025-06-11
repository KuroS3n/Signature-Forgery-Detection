from tensorflow.keras.applications import ResNet50
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import numpy as np

# ✅ Extract features using pretrained ResNet50 (for SVM training)
def extract_features_with_resnet(data_gen):
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    features = []

    print("🔍 Extracting features using ResNet50...")
    for i in range(len(data_gen)):
        batch = data_gen[i]

        if isinstance(batch, (tuple, list)) and len(batch) >= 1:
            batch_x = batch[0]  # Images

            if batch_x.shape[0] == 0:
                print(f"⚠️ Skipping empty batch at index {i}")
                continue

            # Convert grayscale (1-channel) to 3-channel for ResNet
            if batch_x.shape[-1] == 1:
                batch_x = np.repeat(batch_x, 3, axis=-1)

            preds = model.predict(batch_x, verbose=0)
            features.append(preds)
        else:
            print(f"❌ Unexpected batch format at index {i}: {batch}")

    if not features:
        raise ValueError("❌ No features extracted — input generator may be empty or misformatted.")

    return np.vstack(features)

# ✅ Train SVM on extracted features
def train_svm(features, labels):
    clf = make_pipeline(StandardScaler(), SVC(kernel='rbf', probability=True))
    clf.fit(features, labels)
    return clf