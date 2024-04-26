import numpy as np
import torch

from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer



tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)


def example1():
    """ Generate the end of a sentence """
    text = "Anna visits the"

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

    for i in range(1):
        text = next_word(text)
        print(text)

    return

# example1()

def example2():
    """ Probability distribution over the next word """
    # https://stackoverflow.com/questions/76397904/

    text = "The cat sat on the"   # cat / dog / child

    encoded_text = tokenizer(text, return_tensors="pt")

    #1. step to get the logits of the next token
    with torch.inference_mode():
        outputs = model(**encoded_text)

    next_token_logits = outputs.logits[0, -1, :]

    print("Taille du token prédit: ", next_token_logits.shape)
    print("Vecteur logit du token prédit: ")
    print(next_token_logits)

    # 2. step to convert the logits to probabilities
    next_token_probs = torch.softmax(next_token_logits, -1)

    print("Vecteur de probabilité du token prédit: ")
    print(next_token_probs)

    # 3. step to get the top 10
    topk_next_tokens= torch.topk(next_token_probs, 10)

    # 4. putting it together

    result = [(int(idx), tokenizer.decode(idx), float(prob.numpy())) for idx, prob in zip(topk_next_tokens.indices, topk_next_tokens.values)]


    print("Plus grande probabilité pour le token suivant :")
    print("Phrase à compléter: ", text)
    print("  token id, token, probabilité ")
    for idx, prob in zip(topk_next_tokens.indices, topk_next_tokens.values):
        idx = int(idx)
        prob = float(prob.numpy())
        print("  ", idx, tokenizer.decode(idx), "{:.2f}".format(100*prob), "%  ", prob)


    return

example2()