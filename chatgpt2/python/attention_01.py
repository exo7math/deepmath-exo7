
# Importing necessary modules
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Loading pre-trained GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)
# model = GPT2LMHeadModel.from_pretrained('gpt2')

# Encoding input text
sentence = "The dog is black"
input_ids = tokenizer.encode(sentence, return_tensors='pt')

# Generating model output with attention information
output = model.generate(
    input_ids,
    max_length=len(input_ids[0])+1,   # Should be one more than the number of tokens in the input
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    output_attentions=True,
    return_dict_in_generate=True,
)

# Extracting attention tensors
attn = output.attentions



# Printing the shape of the attention tensors
print("Length of the input sequence:", len(input_ids[0]))
print("Number of attention layer:", len(attn))
print("This is the number of new tokens in the sequence (just one for us)")
print("For us only one new word (max_length must one more word than)")

t = 0  # only one new token
# print("For the new token")
print("The number of attention layer is:", len(attn[t]))

layer = 4
print("We consider the layer number:", layer)
print("The shape of this attention tensor is:", attn[t][layer].shape)
print(attn[t][layer])

head = 11
print("We consider the head number:", head)
print("The shape of this attention tensor is:", attn[t][layer][0][head].shape)
print(attn[t][layer][0][head])

# print("For the new token, the attention tensor has the shape: ", attn[0].shape)
