import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
#from sklearn.metrics import confusion_matrix, classification_report




class IMDBDataset(Dataset):
    def __init__(self, text):
        self.text = text
    
    def __len__(self):
        return len(self.text['input_ids'])
    
    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.text.items()}
        return item




def detect(data, model, tokenizer):
    data = data.lower()

    text_tokens = tokenizer(data, padding=True, truncation=True, return_tensors='pt', max_length=512)
    ds = IMDBDataset(text_tokens)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    model.eval()
    results = []
    with torch.no_grad():
        for batch in dl:
            pred = model(**batch)
            prob = F.softmax(pred[0], dim=-1)
            result = prob[0].tolist()

            score_pos = result[1]
            score_neg = result[0]
            label = "Positive" if score_pos > score_neg else "Negative"

            results.append({"Class": label, "Score_Positive": score_pos, "Score_Negative": score_neg})

    return results









    





