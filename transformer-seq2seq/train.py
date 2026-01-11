import torch
import torch.optim as optim
import torch.nn as nn
from transformer import TransformerSeq2Seq
from attention_masks import create_causal_mask
import pickle

# Dataset based on User Inputs S36-S70
raw_data = [
    ("AI improves healthcare", "AI enhances medical diagnosis and treatment"),
    ("Transformers process data in parallel", "Transformers handle sequences simultaneously"),
    ("What is self-attention?", "Self-attention relates each word to every other word"),
    ("Why is positional encoding required?", "Positional encoding provides word order information"),
    ("In the future, AI will", "In the future, AI will automate decision systems"),
    ("Deep learning models can", "Deep learning models can learn abstract features"),
    ("Machine learning helps", "Machine learning helps in data-driven decisions"),
    ("Attention improves NLP accuracy", "Attention mechanisms increase NLP performance"),
    ("What is autoregression?", "Autoregression generates output tokens sequentially"),
    ("Transformers are useful because", "Transformers are useful because they capture global context")
]

# Simple Tokenizer (whitespace)
vocab = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2}
for src, tgt in raw_data:
    # Use a set to avoid duplicates in this loop? No, just straightforward check
    for word in src.split():
        if word not in vocab: vocab[word] = len(vocab)
    for word in tgt.split():
        if word not in vocab: vocab[word] = len(vocab)
            
inv_vocab = {v: k for k, v in vocab.items()}

def to_tensor(sentence):
    return [vocab[word] for word in sentence.split() if word in vocab]

def train():
    d_model = 128
    model = TransformerSeq2Seq(len(vocab), len(vocab), d_model=d_model)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss(ignore_index=0) # Ignore PAD

    print(f"Training on {len(raw_data)} samples with vocab size {len(vocab)}...")

    for epoch in range(200): # training enough to memorize
        total_loss = 0
        for src_text, tgt_text in raw_data:
            src = torch.tensor([to_tensor(src_text)])
            tgt_seq = to_tensor(tgt_text)
            
            tgt_in = torch.tensor([[vocab["<SOS>"]] + tgt_seq])
            tgt_out = torch.tensor([tgt_seq + [vocab["<EOS>"]]])

            optimizer.zero_grad()
            tgt_mask = create_causal_mask(tgt_in.size(1))
            
            # Forward
            preds, _ = model(src, tgt_in, None, tgt_mask)
            
            loss = criterion(preds.view(-1, len(vocab)), tgt_out.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(raw_data):.4f}")
    
    torch.save(model.state_dict(), "transformer_seq2seq.pth")
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    print("Training Complete. Model and Vocab Saved.")

if __name__ == "__main__": train()