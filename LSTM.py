import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam

file_path = 'Student Depression Dataset.csv'
data = pd.read_csv(file_path)

# Select required columns, including the new one
selected_columns = ['Age', 'Gender', 'Academic Pressure', 'Have you ever had suicidal thoughts ?', 'Depression']
data = data[selected_columns]

# Encode Gender column
label_encoder = LabelEncoder()
data['Gender'] = label_encoder.fit_transform(data['Gender'].astype(str))

# Encode "Have you ever had suicidal thoughts ?" if necessary
if data['Have you ever had suicidal thoughts ?'].dtype == 'object':
    data['Have you ever had suicidal thoughts ?'] = label_encoder.fit_transform(data['Have you ever had suicidal thoughts ?'])

# Normalize numerical columns
scaler = MinMaxScaler()
data[['Age', 'Academic Pressure', 'Have you ever had suicidal thoughts ?']] = scaler.fit_transform(
    data[['Age', 'Academic Pressure', 'Have you ever had suicidal thoughts ?']]
)

# Split features and target
X = data[['Age', 'Gender', 'Academic Pressure', 'Have you ever had suicidal thoughts ?']].values
y = data['Depression'].values

# Reshape X for LSTM input
X = np.expand_dims(X, axis=2)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Enhanced LSTM model
model = Sequential([
    LSTM(64, activation='relu', input_shape=(X.shape[1], 1), return_sequences=True),
    Dropout(0.2),
    LSTM(32, activation='relu', return_sequences=False),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_split=0.2, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Make predictions
y_pred = model.predict(X_test)

# Threshold for binary classification
y_pred_class = (y_pred > 0.5).astype(int)

# Compute MSE, RMSE, and R²
mse = mean_squared_error(y_test, y_pred_class)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"(MSE): {mse}")
print(f"(RMSE): {rmse}")
print(f"(R²): {r2}")

# Plot ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred_class)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Depression', 'Depression'], yticklabels=['No Depression', 'Depression'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Plot Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_pred)

plt.figure(figsize=(10, 6))
plt.plot(recall, precision, color='green', lw=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.show()

# Plot MSE, RMSE, and R²
plt.figure(figsize=(10, 6))
metrics = ['MSE', 'RMSE', 'R²']
scores = [mse, rmse, r2]
plt.bar(metrics, scores, color=['blue', 'orange', 'green'])
plt.title('Error Metrics')
plt.ylabel('Scores')
plt.show()

# Plot training & validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training & validation loss
plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot distribution of predictions
plt.figure(figsize=(10, 6))
sns.histplot(y_pred, bins=20, kde=True, color='blue', label='Predicted Probabilities')
plt.axvline(0.5, color='red', linestyle='--', label='Threshold (0.5)')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.title('Distribution of Predicted Probabilities')
plt.legend()
plt.savefig('predicted_distribution.png')
plt.close()

# Plot comparison of actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', alpha=0.6, label='Actual')
plt.scatter(range(len(y_pred_class)), y_pred_class, color='red', alpha=0.4, label='Predicted')
plt.xlabel('Sample Index')
plt.ylabel('Depression (0: No, 1: Yes)')
plt.title('Comparison of Actual vs Predicted Values')
plt.legend()
plt.savefig('actual_vs_predicted.png')
plt.close()

