import torch
from torch.utils.data import DataLoader, random_split

from dataset import HW5Dataset
from model import HW5Model
from sklearn.metrics import f1_score

from torch.nn.utils.rnn import pad_sequence
import subprocess

# Update the meta.csv if you generated some audio
# All the generated data should be label as 1 (generated from AI)
# command = 'find train_dataset -name "*flowavenet*wav" -exec echo "{}",1 >> train_dataset/meta.csv \\;'
# subprocess.run(command, shell=True)

def collate_batch(batch):
   mels, labels = zip(*batch)
   # Pad the mel pectrogram to make them have the same length
   mels = pad_sequence(mels, batch_first=True)
   return mels, torch.FloatTensor(labels)

RANDOM_STATE = 2024
dataset = HW5Dataset('train_dataset/meta.csv')
random_seed_generator = torch.Generator().manual_seed(RANDOM_STATE)

# TODO (5P): Create training and validation dataset based on `dataset`
# and `random_seed_generator`. You should split the dataset into 80%
# samples for the training, and 20% samples for the validation. You
# should use `random_seed_generator` to ensure consistency of the
# result.
# CHECK: https://pytorch.org/docs/stable/data.html#torch.utils.data.random_split

total_size = len(dataset)
print("Total size: " + str(total_size))

train_size = int(0.8 * total_size)  # 80% of the total size
val_size = total_size - train_size  # Remaining 20% for validation
print("Train size: " + str(train_size) + " Validation size: " + str(val_size))

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=random_seed_generator)

# TODO (5P): Create training and validation data loader based on
# `train_dataset` and `val_dataset`. For val_loader, please set
# `shuffle=False` for the consistency of the result.
# CHECK: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#preparing-your-data-for-training-with-dataloaders
# e.g. train_loader = DataLoader(..., collate_fn=collate_batch)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_batch)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_batch)


# Optional: change different hyperparameters, to get the best model.
hidden_size = 16
num_layers = 2
lr = 8e-3
patience = 5
max_n_epochs = 50

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = HW5Model(hidden_size=hidden_size, num_layers=num_layers, lr=lr, device=device)
model.train_epochs(train_loader, val_loader, patience=patience, max_n_epochs=max_n_epochs)

# Load your best model and plot the ROC curve of validation data
model.load_state_dict(torch.load("best_model.ckpt"))
y_pred_prob = model.predict_prob(val_loader)
y_pred = model.predict(val_loader)
y_true = torch.concat([labels for mel, labels in val_loader]).numpy().astype('float32')
print(f'Accuracy on validation set: {(y_pred == y_true).sum() / len(y_true):.2f}')
print(f'Area under precision recall curve on validation set: {model.evaluate(y_true, y_pred_prob):.2f}')
f1 = f1_score(y_true, y_pred, average='binary')
print(f'F1 Score on test set: {f1:.2f}')

## TA will evaluate you model as follows
'''
test_dataset = HW5Dataset('../test_dataset/meta.csv')
test_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_batch)

model = HW5Model(hidden_size=hidden_size, num_layers=num_layers)

# If you directly clone the github repo
model.load_state_dict(torch.load("best_model.ckpt"))
# # If you use colab and submit the .ipynb
# model.load_state_dict(torch.load("/content/best_model.ckpt"))

y_pred_prob = model.predict_prob(test_loader)
y_pred = model.predict(test_loader)
y_true = torch.concat([labels for mel, labels in val_loader]).numpy().astype('float32')
print(f'Accuracy on test set: {(y_pred == y_true).sum() / len(y_true):.2f}')
print(f'Area under precision recall curve on test set: {model.evaluate(y_true, y_pred_prob):.2f}')
f1 = f1_score(y_true, y_pred, average='binary')
print(f'F1 Score on test set: {f1:.2f}')
'''