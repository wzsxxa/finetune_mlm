from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

checkpoint = "distilbert-base-uncased"
model_before_training = AutoModelForMaskedLM.from_pretrained(checkpoint)
tokenizers_before_training = AutoTokenizer.from_pretrained(checkpoint)
model_after_training = AutoModelForMaskedLM.from_pretrained('./model-finetuned-imdb')
tokenizers_after_training = AutoTokenizer.from_pretrained('./model-finetuned-imdb')
f(model_before_training, tokenizers_before_training)
print(f'>>>after training---------------------------------------------------')
f(model_after_training, tokenizers_after_training)

def f(model, tokenizer):
    text = "This is a great [MASK]."
    inputs = tokenizers_before_training(text, return_tensors='pt')
    token_logits = model(**inputs).logits
    # Find the location of [MASK] and extract its logits
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    mask_token_logits = token_logits[0, mask_token_index, :]
    # Pick the [MASK] candidates with the highest logits
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

    for token in top_5_tokens:
        print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")