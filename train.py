import torch
from models.bert_classification import BertForSequenceClassification
from models.bert_regression import BertForSequenceRegression
from utils.data_loader import data_sampler
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import numpy as np
import tqdm

def train(model, train_dataloader, val_dataloader, test_dataloader, epochs, learning_rate, device):
    # Optimizer and scheduler setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model.to(device)

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} is Training")
        
        for _, batch in enumerate(progress_bar):
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels, feature = batch

            model.zero_grad()  # Reset gradients
            loss, _ = model(input_ids, attention_mask, labels, feature)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            torch.cuda.empty_cache()  # Free memory

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Average training loss for epoch {epoch+1}: {avg_train_loss}")
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        predictions, true_labels = [], []
        with torch.no_grad():
            for batch in tqdm.tqdm(val_dataloader, desc="Evaluating"):
                batch = tuple(t.to(device) for t in batch)
                input_ids, attention_mask, labels, feature = batch
                loss, preds = model(input_ids, attention_mask, regression_labels=labels, Feature=feature)
                total_val_loss += loss.item()
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        avg_val_loss = total_val_loss / len(val_dataloader)
        print(f"Validation loss for epoch {epoch+1}: {avg_val_loss}")
        
    # Testing phase
    predictions, true_labels, rmse = test(model, test_dataloader, device)
    print("Regression training complete.")
    return predictions, true_labels

# Classify model training function
def cl_model_train(model, train_dataloader, val_dataloader, epochs, learning_rate, device):
    # Optimizer and scheduler setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    model.to(device)

    # Training loop for contrastive learning model
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        best_val = -np.inf
        progress_bar = tqdm.tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs} is Training")
        
        for _, batch in enumerate(progress_bar):
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels, feature = batch

            model.zero_grad()
            loss, _ = model(input_ids, attention_mask, labels)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()
            torch.cuda.empty_cache()

        avg_train_loss = total_loss / len(train_dataloader)

        # Evaluate model on validation set
        val_accuracy, _, _ = eval_model(model, val_dataloader, device)
        
        # Save best model based on validation accuracy
        if val_accuracy >= best_val:
            best_model = model
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.2f}, Validation Accuracy: {val_accuracy*100:.2f}%")

    print("Classify training complete.")
    return best_model

# Evaluation function for model on validation/test data
def eval_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels, feature = batch

            outputs, _ = model(input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1).flatten()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    return accuracy, all_labels, all_preds

# Testing function for model
def test(model, test_dataloader, device):
    model.eval()
    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in tqdm.tqdm(test_dataloader, desc="Testing"):
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels, feature = batch
            preds, _ = model(input_ids, attention_mask, regression_labels=None, Feature=feature)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    # rmse = mean_squared_error(true_labels, predictions, squared=False)
    rmse = np.sqrt(np.mean((np.array(true_labels) - np.array(predictions)) ** 2))

    return predictions, true_labels, rmse
