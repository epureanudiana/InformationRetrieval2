from torch import nn
from transformers import BertModel
from console_args import arg_parser


class BinaryClassifier(nn.Module):

    def __init__(self, n_classes):
        super(BinaryClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(arg_parser().pre_trained_model_name)
        self.drop = nn.Dropout(p=0.2)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        output = self.drop(pooled_output)
        return nn.functional.log_softmax(self.out(output), dim=1)