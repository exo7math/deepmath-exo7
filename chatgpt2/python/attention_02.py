from transformers import AutoTokenizer, AutoModel, utils
utils.logging.set_verbosity_error()  # Suppress standard warnings
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModel.from_pretrained("gpt2", output_attentions=True)

# sentence = "The dog is black"
sentence = "To be or not to be that is the question"

inputs = tokenizer.encode(sentence, return_tensors='pt')
outputs = model(inputs)
attention = outputs[-1]  # Output includes attention weights when output_attentions=True
tokens = tokenizer.convert_ids_to_tokens(inputs[0]) 

from bertviz import head_view, model_view
# head_view(attention, tokens)
# head_view(attention, tokens, layer=11, heads=[4])
# layer=4, heads=[11]  ## previous token
# layer=0, heads=[3]   ## token itself
# layer=0, heads=[9]   ## direct to all previous tokens
# layer=5, heads=[1]   ## Focuses on repeated words in the sentence (???)
# layer=11, heads=[2]  ## Focus on comma (???)

#model_view(attention, tokens)

my_layer = 0
my_head = 5

html_head_view = head_view(attention, tokens, layer=my_layer, heads=[my_head], html_action='return')

with open("head_view.html", 'w') as file:
    file.write(html_head_view.data)