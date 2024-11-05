from transformers import BartForConditionalGeneration, BartTokenizer
import re

model = BartForConditionalGeneration.from_pretrained("shivamgutgutia/text_summarizer")
tokenizer = BartTokenizer.from_pretrained("shivamgutgutia/text_summarizer")

def abstractiveSummarize(text):
    text = re.sub(r"\s+", " ", text) 
    text = re.sub(r"\n+", " ", text)  
    text = re.sub(r"\r+", " ", text)  
    text = re.sub(r"[^a-zA-Z0-9 .,?!]", "", text)
    text = text.strip()
    inputText = "summarize: " + text
    inputs = tokenizer(
        inputText,
        max_length=512,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to("cpu")

    summaryIds = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=150,
        num_beams=2,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True,
    )

    summary = tokenizer.decode(
        summaryIds[0], skip_special_tokens=True, clean_up_tokenization_spaces=True
    )
    return summary
