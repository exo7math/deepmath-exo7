
from transformers import TFGPT2LMHeadModel
from transformers import GPT2Tokenizer

import tensorflow as tf
import numpy as np

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = TFGPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)



text = "What are you doing after you have finished working?"

input_ids=tokenizer.encode(text,return_tensors='tf')
beam_output=model.generate(input_ids,max_length=100,num_beams=5,no_repeat_ngram_size=2,early_stopping=True)
output=tokenizer.decode(beam_output[0],skip_special_tokens=True,clean_up_tokenization_spaces=True)

print("Output by GPT2:")
print(output)

# generated = tokenizer.encode(text)
# context = tf.constant([generated])
# past = None
# for i in range(100):
#     output, past = model(context, past = past)
#     logits = output[0, -1, :]
#     tok = tf.argmax(logits)
#     generated.append(tok.numpy())
#     context = tf.expand_dims(tf.expand_dims(tok, 0), 0)
# sequence = tokenizer.decode(generated)



# tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)



# inputs = tokenizer("Hello, my dog is cute")

# outputs = model(**inputs)


# sentence = 'What is Elasticsearch?' 
# input_ids = tokenizer.encode(sentence)

# print(input_ids)

# outputs = model.generate(input_ids,
#                             max_length=1000,
#                             do_sample=True,
#                             top_k=50,
#                             top_p=0.95,
#                             num_return_sequences=3)

# print(outputs)

# # output = model.generate(input_ids, 
# #                             max_length = 1000, 
# #                             num_beams = 5,
# #                             no_repeat_ngram_size = 2,
# #                             early_stopping = True)



# # # outputs = model(input_ids, labels=input_ids)
# # # loss, logits = outputs[:2]
# # # print(loss, logits)

# # # outputs = model.generate(input_ids, max_length=100, do_sample=True)
# # # print(outputs)
