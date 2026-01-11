import torch
from transformer import TransformerSeq2Seq
from attention_masks import create_causal_mask
import pickle

def generate_text(model, vocab, inv_vocab, input_text, max_len=20):
    model.eval()
    src_indices = [vocab[word] for word in input_text.split() if word in vocab]
    
    # Handle unknown words roughly (skip them or error)
    if not src_indices:
        return "[Error: No known words in input]"
    
    src = torch.tensor([src_indices])
    generated = torch.tensor([[vocab["<SOS>"]]])
    
    for _ in range(max_len):
        tgt_mask = create_causal_mask(generated.size(1))
        with torch.no_grad():
            logits, _ = model(src, generated, None, tgt_mask)
            next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(0)
            
            if next_token.item() == vocab["<EOS>"]:
                break
                
            generated = torch.cat([generated, next_token], dim=1)
    
    result = [inv_vocab[i.item()] for i in generated[0][1:]] # Skip <SOS>
    return " ".join(result)

def main():
    try:
        with open("vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
    except FileNotFoundError:
        print("Error: vocab.pkl not found. Run train.py first.")
        return

    inv_vocab = {v: k for k, v in vocab.items()}
    d_model = 128
    model = TransformerSeq2Seq(len(vocab), len(vocab), d_model=d_model)
    
    try:
        model.load_state_dict(torch.load("transformer_seq2seq.pth"))
    except FileNotFoundError:
         print("Error: Model file not found. Run train.py first.")
         return

    print("Model Loaded. Vocabulary Size:", len(vocab))
    
    test_inputs = [
        "AI improves healthcare",
        "Transformers process data in parallel",
        "What is self-attention?",
        "Why is positional encoding required?",
        "In the future, AI will"
    ]
    
    print("\n--- Inference Results ---")
    for txt in test_inputs:
        output = generate_text(model, vocab, inv_vocab, txt)
        print(f"Input : {txt}")
        print(f"Output: {output}")
        print("-" * 30)

if __name__ == "__main__": 
    main()