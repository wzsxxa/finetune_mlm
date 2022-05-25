from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
import collections
from transformers import default_data_collator
import numpy as np
from transformers import TrainingArguments, Trainer
import copy
import math

checkpoint = "distilbert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(checkpoint)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
imdb_dataset = load_dataset('imdb', cache_dir="/home/wanxl/datasets")
train_size = 10_000
test_size = 1000
imdb_dataset = imdb_dataset["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)
# print(imdb_dataset)


def tokenize_function(examples):
    result = tokenizer(examples['text'])
    if tokenizer.is_fast:
        result['word_ids'] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result


tokenized_datasets = imdb_dataset.map(tokenize_function, batched=True, remove_columns=["text", "label"])
# print(tokenized_datasets)
chunk_size = 128
tokenized_samples = tokenized_datasets["train"][:3]
# print(type(tokenized_samples["input_ids"]))


def group_texts(examples):
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(concatenated_examples.keys())[0]])
    total_length = (total_length // chunk_size) * chunk_size
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


lm_datasets = tokenized_datasets.map(group_texts, batched=True)
print(lm_datasets)
# print(lm_datasets)
# print(lm_datasets["train"][0])
# print(type(lm_datasets["train"][0]))
# samples = [lm_datasets["train"][i] for i in range(2)]
# print(type(lm_datasets['train']))
# print(type(samples))
# print(samples)
# print(type(samples[0]))
# model.eval()
# with torch.no_grad():
#     outputs = model(input_ids=samples[0]['input_ids'], attention_mask=samples[0]['attention_mask'],
#      labels=samples[0]['labels'])
#     loss = outputs.loss
#     print(loss)
# ssamples = copy.deepcopy(samples)
# for sample in samples:
#     _ = sample.pop("word_ids")
# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.9)
# batch_after_collator = data_collator(samples)
# print(type(batch_after_collator))
# print(batch_after_collator)
wwm_probability = 0.2

def whole_word_masking_data_collator(features):
    for feature in features:
        word_ids = feature.pop("word_ids")
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id
    # print(type(features))
    # print(features)
    # print(features[0].keys())
    return default_data_collator(features)


batch_size = 6
training_args = TrainingArguments(
    output_dir='./model-finetuned-imdb',
    overwrite_output_dir=True,
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    weight_decay=0.01,
    num_train_epochs=1.0,
    per_device_eval_batch_size=batch_size,
    per_device_train_batch_size=batch_size,
    remove_unused_columns=False,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_datasets['train'],
    eval_dataset=lm_datasets['test'],
    data_collator=whole_word_masking_data_collator,
)
eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
trainer.train()
eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")



