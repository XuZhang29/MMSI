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

## Dataset and Preprocessing

An anonymized version of the dataset used in the paper is publicly available on Hugging Face:

- **Hugging Face dataset**: [SHerZH/IMLJP](https://huggingface.co/datasets/SHerZH/IMLJP)

Each example in the dataset (simplified) contains:

- `case_id`: internal case identifier
- `FD`: fact description (事实摘要)
- `CV`: court’s view / reasoning (裁判说理)
- `judgment`: a dictionary mapping each defendant to structured labels, e.g.:

  ```json
  "judgment": {
    "姜某某": {"guilt": "principal", "prison": 36, "probation": 36},
    "姜某":   {"guilt": "accomplice", "prison": 6,  "probation": 12}
  }

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

