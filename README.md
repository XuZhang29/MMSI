# Explainable Multidefendant Judgment Prediction Enhanced by Judicial Logic Based on Large Language Models

This repository contains the implementation code for the paper **"Explainable Multidefendant Judgment Prediction Enhanced by Judicial Logic Based on Large Language Models"**. The code utilizes PyTorch, transformers, and other essential libraries to build and train the models.

## Requirements

Please ensure your environment meets the following requirements. You can install the dependencies using `setup.bash` or manually via pip. Adjust the PyTorch version if your CUDA environment differs.

### Core Libraries
- **PyTorch and GPU support**  
  - `torch==2.0.1+cu118`
  - `torchvision==0.15.2+cu118`
  - `torchaudio==2.0.2+cu118`
- **Transformers and related tools**  
  - `transformers==4.31.0`
  - `tokenizers==0.13.3`  (Supports fast tokenization)
- **Data Processing**  
  - `numpy==1.23.5`
  - `pandas==1.5.3`
  - `scikit-learn==1.2.2`  (For data processing and evaluation)
- **Progress Bar Utility**  
  - `tqdm==4.66.1`

## Data Privacy and Preprocessing

Due to the sensitive nature of the legal data, we do not provide the original dataset. Instead, we offer:
1. A preprocessed dataset, encoded using the `bert-base-chinese` model, available as `IMLJP4train_encoded.pkl`. Ensure this file is correctly placed in the folder `/data/`.
2. A sample dataset, `IMLJP_50.pkl`, containing 50 cases as an example for understanding the data preprocessing pipeline. You can replace it with your own dataset for custom experiments.

The data used in the paper comes from repeated experiments with multiple random splits. While results from a single run may vary, the overall conclusions remain consistent.

### Requesting Data

If you require access to the dataset, please contact the corresponding author. Any data requests will undergo a strict review process, and the usage will be tightly controlled to ensure data security and ethical compliance.

## Setup

Run the following to install dependencies and set up your environment:
```bash
bash setup.bash
```

## How to Run

Ensure the encoded dataset `IMLJP4train_encoded.pkl` is correctly placed in the specified folder. If you are using your own dataset, preprocess it accordingly or refer to the sample dataset for guidance. Then, execute the main script:

```bash
python main.py
```

## Note

The results presented in the paper are aggregated from extensive experiments with multiple random splits. Single-run results may show minor differences, but the overall trends and conclusions remain unchanged.

For further details or questions, feel free to refer to the paper or contact the authors.

