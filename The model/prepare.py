import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import glob

# Load all CSV files
print("Loading CSV files...")
csv_path = r"C:\Users\SaifEddine\OneDrive\Desktop\Projects\collect my own data\Data\*.csv"
all_files = glob.glob(csv_path)

X_list = []
y_list = []

for filename in all_files:
    df = pd.read_csv(filename)
    
    if len(df) != 300:
        print(f"Warning: {filename} has {len(df)} rows. Skipping.")
        continue
    
    features = df.iloc[:, 0:6].values
    label = df.iloc[0, 6]
    
    X_list.append(features)
    y_list.append(label)

X_original = np.array(X_list)
y_original = np.array(y_list)

print(f"Loaded {len(X_original)} original sessions")

# AGGRESSIVE DATA AUGMENTATION (create 20x more data)
def augment_session(session, num_augmentations=20):
    """Create multiple augmented versions of a single session"""
    augmented = [session]  # Include original
    
    for _ in range(num_augmentations - 1):
        aug = session.copy()
        
        # Add Gaussian noise 
        noise = np.random.normal(0, 0.03, aug.shape)
        aug = aug + noise
        
        # Random scaling 
        scale = np.random.uniform(0.9, 1.1)
        aug = aug * scale
        
        # Time warping 
        if np.random.rand() > 0.5:
            # Random time indices
            old_indices = np.linspace(0, 299, 300)
            warp_factor = np.random.uniform(0.95, 1.05)
            new_indices = np.linspace(0, 299, int(300 * warp_factor))
            
            # Interpolate to 300 samples
            aug_warped = np.zeros_like(aug)
            for feat in range(6):
                aug_warped[:, feat] = np.interp(
                    old_indices, 
                    np.linspace(0, 299, len(new_indices)), 
                    np.interp(new_indices, old_indices, aug[:, feat])
                )
            aug = aug_warped
        
        # 4. Random rotation 
        if np.random.rand() > 0.7:
            angle = np.random.uniform(-0.1, 0.1)
            # Rotate accelerometer readings slightly
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            aug_rot = aug.copy()
            aug_rot[:, 0] = aug[:, 0] * cos_a - aug[:, 1] * sin_a
            aug_rot[:, 1] = aug[:, 0] * sin_a + aug[:, 1] * cos_a
            aug = aug_rot
        
        augmented.append(aug)
    
    return augmented

print("\nApplying aggressive augmentation (20x per session)...")
X_augmented = []
y_augmented = []

for i, (session, label) in enumerate(zip(X_original, y_original)):
    augmented_sessions = augment_session(session, num_augmentations=20)
    X_augmented.extend(augmented_sessions)
    y_augmented.extend([label] * len(augmented_sessions))
    
    if (i + 1) % 20 == 0:
        print(f"  Processed {i + 1}/{len(X_original)} sessions...")

X_augmented = np.array(X_augmented)
y_augmented = np.array(y_augmented)

print(f"\nAugmented dataset: {len(X_augmented)} sessions (from {len(X_original)} original)")

# Filter to allowed labels
allowed_labels = [0, 8, 11, 14]
mask = np.isin(y_augmented, allowed_labels)
X_augmented = X_augmented[mask]
y_augmented = y_augmented[mask]

# Balance dataset
from collections import Counter
label_counts = Counter(y_augmented)
min_count = min(label_counts.values())

print(f"\nBalancing to {min_count} sessions per class...")

X_balanced = []
y_balanced = []

for label in allowed_labels:
    indices = np.where(y_augmented == label)[0]
    if len(indices) > min_count:
        np.random.seed(42)
        selected_indices = np.random.choice(indices, min_count, replace=False)
    else:
        selected_indices = indices
    
    X_balanced.append(X_augmented[selected_indices])
    y_balanced.append(y_augmented[selected_indices])

X_final = np.concatenate(X_balanced, axis=0)
y_final = np.concatenate(y_balanced, axis=0)

# Shuffle
np.random.seed(42)
shuffle_indices = np.random.permutation(len(X_final))
X_final = X_final[shuffle_indices]
y_final = y_final[shuffle_indices]

print(f"\nFinal balanced dataset: {len(X_final)} sessions")
print(f"Sessions per class: {len(X_final) // 4}")

# Encode labels
activity_names = {0: 'sit', 8: 'jum', 11: 'walk', 14: 'run'}
lb = LabelBinarizer()
y_encoded = lb.fit_transform(y_final)

print(f"\nLabel mapping:")
for i, label in enumerate(lb.classes_):
    print(f"  Index {i}: {activity_names[label]}")

# Split data 
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Save
np.savez_compressed(
    'my_data.npz', 
    X_train=X_train, 
    X_test=X_test, 
    y_train=y_train, 
    y_test=y_test
)

print(f"\n✓ Training samples: {len(X_train)}")
print(f"✓ Test samples: {len(X_test)}")
print(f"✓ Training set per class: {y_train.sum(axis=0)}")
print(f"✓ Test set per class: {y_test.sum(axis=0)}")
print("✓ Data saved to my_data.npz")
print("\nThis should give you ~1,920 training samples and ~480 test samples!")