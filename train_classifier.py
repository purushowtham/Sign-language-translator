import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the data
data_dict = pickle.load(open('./data.pickle', 'rb'))

# Check the lengths of all items
print("Inspecting data shapes...")
for i, item in enumerate(data_dict['data']):
    print(f"Item {i} shape: {np.shape(item)}")

# Ensure all data is padded to a consistent length (e.g., maximum length)
max_length = max(len(item) for item in data_dict['data'])
padded_data = [
    np.pad(item, (0, max_length - len(item)), 'constant') if len(item) < max_length else item
    for item in data_dict['data']
]
data = np.asarray(padded_data)

# Verify the padded data shape
print(f"Data shape after padding: {data.shape}")

# Convert labels to numpy array
labels = np.asarray(data_dict['labels'])

# Ensure labels and data sizes match
if len(data) != len(labels):
    raise ValueError(f"Mismatch: {len(data)} samples and {len(labels)} labels.")

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train the RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions
y_predict = model.predict(x_test)

# Evaluate accuracy
score = accuracy_score(y_predict, y_test)
print(f"{score * 100:.2f}% of samples were classified correctly!")

# Save the trained model
with open('model.p', 'wb') as f:
    pickle.dump({'model': model}, f)
