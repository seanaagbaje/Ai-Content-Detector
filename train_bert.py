from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import create_optimizer
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. Load the combined dataset (final_combined_dataset.csv)
print("Loading dataset...")
data = pd.read_csv('final_combined_dataset.csv')
print("Dataset loaded.")

# Remove rows where 'text' is not a string or contains invalid data (e.g., NaN)
print("Cleaning dataset...")
data = data.dropna(subset=['text'])  # Remove rows with NaN in 'text'
data = data[data['text'].apply(lambda x: isinstance(x, str))]  # Keep only rows where 'text' is a string
print("Dataset cleaned.")

# Separate the text and labels
texts = data['text'].tolist()
labels = data['label'].tolist()

# Verify initial dataset consistency
print(f"Number of text samples: {len(texts)}")
print(f"Number of labels: {len(labels)}")

# 2. Load the tokenizer using vocab.txt
print("Loading tokenizer...")
tokenizer = BertTokenizer(vocab_file='./models/bert/vocab.txt')
print("Tokenizer loaded.")

# Tokenize the text data
print("Tokenizing data...")
encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="tf")
print("Tokenization complete.")

# Convert TensorFlow tensors to NumPy arrays for train_test_split
input_ids = encodings['input_ids'].numpy()  # Convert input_ids to NumPy array

# Verify input_ids and labels length
print(f"Number of tokenized samples: {input_ids.shape[0]}")
print(f"Number of labels: {len(labels)}")

# Ensure that the lengths are the same
if input_ids.shape[0] != len(labels):
    print("Error: Mismatch between the number of input samples and labels!")
else:
    print("Data is consistent.")

# Ensure labels are integers
labels = [int(label) for label in labels]  # Ensure labels are integers

# Train-test split
print("Splitting data into training and validation sets...")
X_train, X_val, y_train, y_val = train_test_split(input_ids, labels, test_size=0.2, random_state=42)
print("Data split complete.")

# Convert to TensorFlow datasets
print("Preparing TensorFlow datasets...")
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(16)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(16)
print("TensorFlow datasets ready.")

# Debugging: Check a sample of the data
for data, label in train_dataset.take(1):
    print(f"Sample input data: {data}")
    print(f"Sample label: {label}")

# 3. Load the pre-trained BERT model (tf_model.h5)
print("Loading BERT model...")
model = TFBertForSequenceClassification.from_pretrained('./models/bert/', num_labels=2)
print("BERT model loaded.")

# 4. Set up optimizer and training
print("Setting up optimizer and training parameters...")
epochs = 2
batch_size = 32
steps_per_epoch = len(train_dataset) // batch_size
num_train_steps = steps_per_epoch * epochs

optimizer, lr_schedule = create_optimizer(
    init_lr=2e-5, num_train_steps=num_train_steps, num_warmup_steps=0)

# Compile the model with SparseCategoricalCrossentropy
print("Compiling the model...")
model.compile(optimizer=optimizer, loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
print("Model compiled.")

# 5. Train the model
print("Starting training...")
history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)
print("Training complete.")

# 6. Save the fine-tuned model
print("Saving the fine-tuned model...")
model.save_pretrained("./models/bert_finetuned/")
print("Model saved successfully.")
