import torch.nn as nn
from transformers import BertModel
from .attention import CustomAttentionLayer

class BertForSequenceClassification(nn.Module):
    def __init__(self, dropout_rate, bert_model_path, num_labels):
        super(BertForSequenceClassification, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_path, output_attentions=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.attention_layer = CustomAttentionLayer(self.bert.config.hidden_size, num_heads=self.bert.config.num_attention_heads)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, output_attentions=True)
        last_hidden_state = outputs.last_hidden_state
        last_attention = outputs.attentions[-1].clone()

        attn_output, _ = self.attention_layer(last_hidden_state, last_hidden_state, last_hidden_state, attn_mask=last_attention)
        cls_embedding = attn_output[:, 0, :] 
        cls_embedding = self.dropout(cls_embedding)
        logits = self.classifier(cls_embedding)

        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            return loss, logits

        return logits, last_attention
