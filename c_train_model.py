import pickle

data_dict = pickle.load(open('data\data.pickle', 'rb'))

import numpy as np

data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Check for class imbalance
unique, counts = np.unique(labels, return_counts=True)
print("Class distribution:")
for cls, count in zip(unique, counts):
    print(f"  Class {cls}: {count} samples ({count/len(labels)*100:.2f}%)")

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

print(f"\nTraining set size: {x_train.shape}")
print(f"Training set size: {y_train.shape}")
print(f"Test set size: {x_test.shape}")

model = RandomForestClassifier(
    n_estimators=100,        # Number of trees
    max_depth=10,            # Limit tree depth
    min_samples_split=5,     # Minimum samples to split
    min_samples_leaf=2,      # Minimum samples per leaf
    n_jobs=-1,               # Use all CPU cores
    random_state=42,
    # verbose=1                # This WILL show some progress!
)

history = model.fit(x_train, y_train)


with open('data/model.p', 'wb') as file:
    pickle.dump({'model': model}, file)



from sklearn.metrics import accuracy_score

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print("{}% of samples correctly classified correctly".format(score * 100))