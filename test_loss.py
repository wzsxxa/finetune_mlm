from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel
import torch
import torch.nn as nn

# target = torch.empty(3, dtype=torch.long).random_(5)
# print(type(target[0]))

checkpoint = "distilbert-base-uncased"
head_model = AutoModelForMaskedLM.from_pretrained(checkpoint)
head_model.eval()
# model = AutoModel.from_pretrained(checkpoint)
# model.eval()
tokenizers = AutoTokenizer.from_pretrained(checkpoint)

text = 'what a good [MASK]!'
labels = torch.tensor([[33, 10, 100, 1000,  103,  999,  1100]])
inputs = tokenizers(text, return_tensors='pt')
# print(inputs)
outputs = head_model(**inputs, labels=labels)
print(outputs.loss)
logits = outputs.logits
loss_fct = nn.CrossEntropyLoss()
loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
print(loss)
# print(outputs.logits.shape)
# outputs = model(**inputs)
# print(outputs)
# print(outputs.last_hidden_state.shape)
# logits = outputs.logits[0]
# labels = labels[0]
# loss = torch.tensor(0.)
# sfm = nn.Softmax(dim=0)
# cel = nn.CrossEntropyLoss()
# for i in range(len(logits)):
#     sfm_embedding = sfm(logits[i])
#     print(sfm_embedding.shape, labels[i].shape)
#     part_loss = cel(sfm_embedding.unsqueeze(0), labels[i].unsqueeze(0))
#     print(f">>>{i} part_loss:{part_loss:.2f}")
#     loss += part_loss
#
# print(loss)
