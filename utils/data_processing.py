import torch

def preprocessing_for_bert(data, tokenizer, max_len):
    input_ids = []
    attention_masks = []
    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=sent,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True
        )
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks

def clean_cv(cv):
    sentences = cv.split('。')
    cleaned_cv = [sent for sent in sentences if not any(kw in sent for kw in ['主犯', '从犯', '主要', '次要'])]
    return '。'.join(cleaned_cv)
