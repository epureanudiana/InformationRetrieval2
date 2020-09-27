import torch
from torch.utils.data import Dataset, DataLoader


# helper class for storing the data
class OpenDialKGDataset(Dataset):

    def __init__(self, messages, labels, tokenizer, max_len):
        self.messages = messages
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, item):
        message = str(self.messages[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            message,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            pad_to_max_length=True,
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt',
        )

        return {
            'message': message,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


# loader function for the above class

def create_data_loader(df, tokenizer, max_len, batch_size):
    dataset = OpenDialKGDataset(messages=df.Message.to_numpy(),
                                labels=df.Label.to_numpy(),
                                tokenizer=tokenizer,
                                max_len=max_len)

    return DataLoader(dataset, batch_size=batch_size)