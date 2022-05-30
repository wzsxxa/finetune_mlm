from transformers import AutoTokenizer
from datasets import load_dataset

# data = load_dataset('super_glue', 'rte')
# print(data['train']['premise'])
# print(type(data['train']['premise']))
batch_sentences = ["Hello I'm a single sentence",
                   "And another sentence",
                   "And the very very last one"]
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased", use_fast=False)
# encoded_inputs = tokenizer("Hello I'm a single sentence", padding=True, return_tensors='pt')
# print(encoded_inputs)
encoded_inputs = tokenizer("Hello I'm a [UNK] sentence", padding=True, return_tensors='pt',
                           return_special_tokens_mask=True)
print(encoded_inputs)