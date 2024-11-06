print("This might take a while to run", end="\n\n")

print("IMPORTING LIBRARIES", end="\n")

import time
import pandas as pd
import torch
from torch import cuda
from torch.utils.data import Dataset, DataLoader
from transformers import BartTokenizer, BartForConditionalGeneration
import os
import shutil

device = "cuda" if cuda.is_available() else "cpu"
print(f"Using device: {device}", end="\n\n")

folder_path = "./model"

if os.path.exists(folder_path):
    shutil.rmtree(folder_path)
os.makedirs(folder_path)

torch.backends.cudnn.deterministic = True

new_repo = "text_summarizer"
repo_name = "EducativeCS2023/bart-base-summarization"

print("RETRIEVING TRAINING DATA", end="\n\n")

df = pd.read_csv("BBCarticles.csv", encoding="latin-1")
df = df[["Text", "Summary"]]
df.Text = "summarize: " + df.Text
split_ratio = 0.025
train_dataset = df.sample(frac=split_ratio).reset_index(drop=True)
eval_dataset = (
    df.drop(train_dataset.index).sample(frac=split_ratio).reset_index(drop=True)
)

print("Training Dataset Size:", train_dataset.shape, end="\n\n")
print("Evaluation Dataset Size:", eval_dataset.shape, end="\n\n")
print(df.head(3), end="\n\n")


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, source_len, summ_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.source_len = source_len
        self.summ_len = summ_len
        self.Summary = self.data.Summary
        self.Text = self.data.Text

    def __len__(self):
        return len(self.Summary)

    def __getitem__(self, index):
        Text = str(self.Text[index])
        Text = " ".join(Text.split())

        Summary = str(self.Summary[index])
        Summary = " ".join(Summary.split())
        source_encoded = self.tokenizer(
            Text,
            max_length=self.source_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        target_encoded = self.tokenizer(
            Summary,
            max_length=self.summ_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        source_ids = source_encoded["input_ids"].squeeze()
        source_mask = source_encoded["attention_mask"].squeeze()
        target_ids = target_encoded["input_ids"].squeeze()

        return {
            "source_ids": source_ids.to(dtype=torch.long),
            "source_mask": source_mask.to(dtype=torch.long),
            "target_ids": target_ids.to(dtype=torch.long),
        }


tokenizer = BartTokenizer.from_pretrained(repo_name)
training_set = CustomDataset(train_dataset, tokenizer, 512, 150)
eval_set = CustomDataset(eval_dataset, tokenizer, 512, 150)
training_loader = DataLoader(training_set, batch_size=2, shuffle=True, num_workers=0)
eval_loader = DataLoader(eval_set, batch_size=2, shuffle=False, num_workers=0)

print("INITIALISING BASE MODEL", end="\n\n")

model = BartForConditionalGeneration.from_pretrained(repo_name)
model = model.to(device)
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-4)


def train(epoch, tokenizer, model, device, loader, optimizer):
    model.train()
    for batch_index, data in enumerate(loader, 0):
        y = data["target_ids"].to(device, dtype=torch.long)
        y_ids = y[:, :-1].contiguous()
        labels = y[:, 1:].clone().detach()
        labels[y[:, 1:] == tokenizer.pad_token_id] = -100
        ids = data["source_ids"].to(device, dtype=torch.long)
        mask = data["source_mask"].to(device, dtype=torch.long)
        outputs = model(
            input_ids=ids, attention_mask=mask, decoder_input_ids=y_ids, labels=labels
        )
        loss = outputs[0]

        if batch_index % 500 == 0:
            print(f"Epoch: {epoch}, Loss:  {loss.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


print("TRAINING MODEL", end="\n\n")

for epoch in range(2):
    print(f"Training epoch: {epoch+1}/{2}")
    train(epoch, tokenizer, model, device, training_loader, optimizer)

model.save_pretrained("./model")
tokenizer.save_pretrained("./model")


def predict(tokenizer, model, device, loader):
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for batch_index, data in enumerate(loader, 0):
            ids = data["source_ids"].to(device, dtype=torch.long)
            mask = data["source_mask"].to(device, dtype=torch.long)
            y = data["target_ids"].to(device, dtype=torch.long)
            generated_ids = model.generate(
                input_ids=ids,
                attention_mask=mask,
                max_length=150,
                num_beams=2,
                repetition_penalty=2.5,
                length_penalty=1.0,
                early_stopping=True,
            )
            preds = [
                tokenizer.decode(
                    g, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for g in generated_ids
            ]
            target = [
                tokenizer.decode(
                    t, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                for t in y
            ]

            if batch_index % 100 == 0:
                print(f"Completed {batch_index} batches")

            predictions.extend(preds)
            actuals.extend(target)

    return predictions, actuals


print("PREDICTING OUTPUTS", end="\n\n")

start_time = time.time()
model = BartForConditionalGeneration.from_pretrained("./model")
tokenizer = BartTokenizer.from_pretrained("./model")
predictions, actuals = predict(tokenizer, model, device, eval_loader)
results = pd.DataFrame({"predictions": predictions, "actuals": actuals})
results.to_csv("results.csv")
end_time = time.time()
time_taken = end_time - start_time
print(f"Time taken for predictions: {time_taken:.2f} seconds")
print(results.head())

print("EVALUATING MODEL", end="\n\n")

import evaluate

rouge_score = evaluate.load("rouge")
scores = rouge_score.compute(
    predictions=results["predictions"], references=results["actuals"]
)
rouge_scores_df = pd.DataFrame([scores]).transpose()
print(rouge_scores_df.head())
