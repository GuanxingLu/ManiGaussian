
import torch
from termcolor import cprint


def create_language_model(name, device=None):
    if name == 'T5':
        return T5EmbeddingExtractor(model_name='t5-base')
    elif name == 'CLIP':
        return CLIP(model_name='RN50', device=device)
    else:
        raise ValueError("Unknown language model")


class T5EmbeddingExtractor:
    def __init__(self, model_name='t5-base'):
        from transformers import T5Tokenizer, T5EncoderModel
        # Load T5 tokenizer and model
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=512)
        self.model = T5EncoderModel.from_pretrained(model_name).cuda()
        cprint(f"T5 model loaded: {model_name}", "green")

    def extract(self, text):
        # pad to 77 tokens
        text = text + " <pad>" * (77 - len(self.tokenizer.encode(text)))
        # Tokenize the input text
        input_ids = self.tokenizer(text, return_tensors='pt').input_ids.cuda()

        # Generate embeddings from the T5 model
        outputs = self.model(input_ids=input_ids)
        embeddings = outputs.last_hidden_state
        sentence_embedding = torch.zeros(1, 1024).cuda()
        return sentence_embedding, embeddings

class CLIP:
    def __init__(self, model_name='RN50', device=None):
        from .clip.core.clip import build_model, load_clip, tokenize
        self.tokenizer = tokenize
        model, _ = load_clip(model_name, jit=False, device="cpu")
        language_model = build_model(model.state_dict())
        del model
        self.device = device
        self.model = language_model.to(device) if device is not None else language_model.cuda()
        cprint(f"CLIP model loaded: {model_name}", "green")
    
    # @torch.no_grad()
    def extract(self, text):
        tokens = self.tokenizer([text]).numpy()
        token_tensor = torch.from_numpy(tokens).to(self.device) if self.device is not None else torch.from_numpy(tokens).cuda()
        sentence_emb, token_embs = self.model.encode_text_with_embeddings(token_tensor)
        return sentence_emb, token_embs


if __name__ == "__main__":
    model = T5EmbeddingExtractor()
    text = "This is a test sentence"
    sentence_emb, token_embs = model.extract(text)
    print(token_embs.shape)


    model = CLIP()
    sentence_emb, token_embs = model.extract(text)
    print(token_embs.shape)
    

        