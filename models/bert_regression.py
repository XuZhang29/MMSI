import torch
import torch.nn as nn
from transformers import BertModel
from .attention import CustomAttentionLayer

class BertForSequenceRegression(nn.Module):
    def __init__(self, dropout_rate, bert_model_path):
        super(BertForSequenceRegression, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_path, output_attentions=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.regressor = nn.Linear(self.bert.config.hidden_size, 1)
        self.attention_layer = CustomAttentionLayer(self.bert.config.hidden_size, num_heads=self.bert.config.num_attention_heads)
        self.feature_expander = nn.Linear(1, self.bert.config.hidden_size)

    def forward(self, input_ids, attention_mask, regression_labels=None, Feature=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True, output_attentions=True)
        last_hidden_state = outputs.last_hidden_state
        last_attention = outputs.attentions[-1].clone()

        attn_output, _ = self.attention_layer(last_hidden_state, last_hidden_state, last_hidden_state, attn_mask=last_attention)
        cls_embedding = attn_output[:, 0, :]
        
        if Feature is not None:
            expanded_feature = self.feature_expander(Feature.unsqueeze(1).float())
            cls_embedding = cls_embedding + expanded_feature

        cls_embedding = self.dropout(cls_embedding)
        regression_output = self.regressor(cls_embedding)

        if regression_labels is not None:
            regression_labels = regression_labels.to(torch.float32)
            regression_loss_fn = nn.MSELoss()
            loss = regression_loss_fn(regression_output.view(-1), regression_labels.view(-1))
            return loss, regression_output
        else:
            return regression_output, last_attention
