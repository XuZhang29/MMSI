# -*- coding: utf-8 -*-
import torch
from transformers import BertTokenizerFast
from models.bert_classification import BertForSequenceClassification
from models.bert_regression import BertForSequenceRegression
from utils.data_gener import generate_pairs_MASK
from utils.data_loader import data_sampler
from utils.eval import classify_eval, regression_eval
from train import cl_model_train, train, eval_model
from sklearn.model_selection import train_test_split
import pickle as pkl
import os

if __name__ == "__main__":
    
    if not os.path.exists('/data/IMLJP4train_encoded.pkl'):
        
        # Load Data: This is a sample dataset with only 50 cases provided to illustrate the data preprocessing method. 
        # To ensure the security of legal data, the full dataset is not included. 
        # Preprocessed data has already been provided; please ensure that the file 
        # `IMLJP4train_encoded.pkl` is correctly placed in the folder.
        
        data_path = "/data/IMLJP_50.pkl"     
        
        # Generate Data and Mask
        fds, cvs, labels, prisons = generate_pairs_MASK(data_path)
        
        # Shuffle and Split Data
        data = list(zip(fds, cvs, labels, prisons))
        train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

        # Unzip Data
        train_fds, train_cvs, train_labels, train_prisons = zip(*train_data)
        val_fds, val_cvs, val_labels, val_prisons = zip(*val_data)
        test_fds, test_cvs, test_labels, test_prisons = zip(*test_data)
        
        # Encodeing data and Data Samplers
        batch_size = 16
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        model_base = "bert-base-chinese"
        tokenizer = BertTokenizerFast.from_pretrained(model_base)
        train_sampler = data_sampler(train_fds, train_labels, train_labels, tokenizer, batch_size, max_len=512, shuffle=True)
        val_sampler = data_sampler(val_fds, val_labels, val_labels, tokenizer, batch_size, max_len=512, shuffle=True)
        test_sampler = data_sampler(test_fds, test_labels, test_labels, tokenizer, batch_size, max_len=512, shuffle=False)
        pkl.dump([train_sampler, val_sampler, test_sampler], open('/data/IMLJP4train_encoded.pkl', 'wb'))
        
    else:
        [train_sampler, val_sampler, test_sampler] = pkl.load(open('/data/IMLJP4train_encoded.pkl', 'rb'))
        batch_size = 16
        model_base = "bert-base-chinese"
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Parameters
    epochs = 10
    batch_size = batch_size
    learning_rate = 2e-5
    dropout_rate = 0.1
    
    # Initialize Models
    regression_model = BertForSequenceRegression(dropout_rate=dropout_rate, bert_model_path=model_base)
    classify_model = BertForSequenceClassification(dropout_rate=dropout_rate, bert_model_path=model_base, num_labels=2)

    
    # MMSI Training
    print('-' * 86)
    print(f'Classify Training...')

    # Train Classifier Model
    cl_model = cl_model_train(classify_model, train_sampler, val_sampler, epochs, learning_rate, device)
    
    # Generate Pseudo-labels from Classifier for Regression
    print(f'Generating pseudo-labels...')
    _, _, train_labels_p = eval_model(cl_model, train_sampler, device)
    _, _, val_labels_p = eval_model(cl_model, val_sampler, device)
    _, _, test_labels_p = eval_model(cl_model, test_sampler, device)
    
    Acc, P, R, F1 = classify_eval([test_labels, test_labels_p])
    print(f'Classify Model test results: Acc.: {Acc}, P.: {P}, R: {R}, F1: {F1}.')

    # Prepare Data Samplers for Regression
    print(f'Regression Training...')
    train_sampler = data_sampler(train_cvs, train_prisons, train_labels_p, tokenizer, batch_size, max_len=512, shuffle=True)
    val_sampler = data_sampler(val_cvs, val_prisons, val_labels_p, tokenizer, batch_size, max_len=512, shuffle=True)
    test_sampler = data_sampler(test_cvs, test_prisons, test_labels_p, tokenizer, batch_size, max_len=512, shuffle=False)

    # Train Regression Model
    predictions, groundtruth = train(regression_model, train_sampler, val_sampler, test_sampler, epochs, learning_rate, device)
    
    ImpScore, ImpAcc, ImpErr = regression_eval([predictions, groundtruth])
    print(f'Regression Model test results: ImpScore: {ImpScore}, ImpAcc: {ImpAcc}, ImpErr: {ImpErr}.')
    
    # Save Predictions and Groundtruth
    os.makedirs('/results', exist_ok=True)
    save_path = '/results/predicted_samples.pkl'
    pkl.dump([predictions, groundtruth], open(save_path, 'wb'))
    
    print(f"Training complete, results saved as {save_path}.")
    print('-' * 86)
    print('Note: This run result is a test of 50 samples. Due to the limitation of platform GPU resources, it is impossible to test the full sample data. The accuracy of the small sample test result is low and not referenceable. The test data has been provided, see README.md for details.')
    