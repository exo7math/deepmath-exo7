import numpy as np
import torch

from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer



tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)


def example1(N):
    """ Generate the end of a sentence (with N words)"""
    text = "This dog is"

    def next_word(text):
        indexed_tokens = tokenizer.encode(text)
        tokens_tensor = torch.tensor([indexed_tokens])
        model.eval()

        with torch.no_grad():
            outputs = model(tokens_tensor)
            predictions = outputs[0]
            
        predicted_index = torch.argmax(predictions[0, -1, :]).item()
        predicted_text = tokenizer.decode(indexed_tokens + [predicted_index])
        return predicted_text

    for i in range(N):
        text = next_word(text)
        print(text)

    return

example1(10)


# For introduction to chapter 1 of ChatGPT
def example2():
    """ Probability distribution over the next word """
    # https://stackoverflow.com/questions/76397904/

    text = "This dog is"

    encoded_text = tokenizer(text, return_tensors="pt")

    #1. step to get the logits of the next token
    with torch.inference_mode():
        outputs = model(**encoded_text)

    next_token_logits = outputs.logits[0, -1, :]
    print(next_token_logits.shape)
    print(next_token_logits)

    # 2. step to convert the logits to probabilities
    next_token_probs = torch.softmax(next_token_logits, -1)

    # 3. step to get the top 10
    topk_next_tokens= torch.topk(next_token_probs, 10)

    #putting it together
    print("Next tokens and their probabilities:")
    print("After sentence: ", text)
    print(*[(tokenizer.decode(idx), float(prob.numpy())) for idx, prob in zip(topk_next_tokens.indices, topk_next_tokens.values)], sep="\n")

    return

example2()
