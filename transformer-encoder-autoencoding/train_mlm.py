import torch
import torch.nn as nn
import torch.optim as optim
from encoder import EncoderLayer
from positional_encoding import PositionalEncoding
import os
import pickle

# Dataset based on User Inputs S1-S35 Sample
# (ID, Topic, Masked Input, Reconstructed Output)
data_samples = [
    ("SAMPLE1", "AI", "Transformers use [MASK] attention", "Transformers use self attention"),
    ("SAMPLE2", "Space", "Mars is called the [MASK] planet", "Mars is called the red planet"),
    ("SAMPLE3", "Education", "Online learning improves [MASK] access", "Online learning improves educational access"),
    ("SAMPLE4", "Health", "Exercise improves [MASK] health", "Exercise improves mental health"),
    ("SAMPLE5", "Sports", "Cricket is a [MASK] sport", "Cricket is a popular sport"),
    ("SAMPLE6", "Computing", "Python is a [MASK] language", "Python is a programming language"),
    ("SAMPLE7", "AI", "Neural networks have [MASK] layers", "Neural networks have hidden layers"),
    ("SAMPLE8", "Environment", "Trees reduce [MASK] pollution", "Trees reduce air pollution"),
    ("SAMPLE9", "Robotics", "Robots perform [MASK] tasks", "Robots perform repetitive tasks"),
    ("SAMPLE10", "Energy", "Solar power is a [MASK] source", "Solar power is a renewable source")
]

# Build Vocabulary
vocab = {"[PAD]": 0, "[MASK]": 1}
for _, _, _, target_sent in data_samples:
    for word in target_sent.split():
        if word not in vocab: 
            vocab[word] = len(vocab)
            
rev_vocab = {v: k for k, v in vocab.items()}

class TransformerMLM(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.encoder = EncoderLayer(d_model, num_heads)
        self.decoder = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)
        x, attn = self.encoder(x)
        return self.decoder(x), attn # Returns [batch, seq_len, vocab_size], attn_weights

def tokenize(sentence):
    return [vocab.get(word, vocab.get(word, 0)) for word in sentence.split()]

def train_and_evaluate():
    model = TransformerMLM(len(vocab))
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"Vocab size: {len(vocab)}")
    print("Training MLM Reconstruction...")
    
    # Training Loop
    for epoch in range(100):
        total_loss = 0
        for _, _, masked_txt, target_txt in data_samples:
            inputs = torch.tensor([tokenize(masked_txt)])
            targets = torch.tensor([tokenize(target_txt)])
            
            optimizer.zero_grad()
            output, _ = model(inputs)
            
            # Reshape for loss: [batch*seq_len, vocab_size] vs [batch*seq_len]
            loss = criterion(output.view(-1, len(vocab)), targets.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}, Loss: {total_loss/len(data_samples):.4f}")

    # Save Model
    if not os.path.exists('results'): os.makedirs('results')
    torch.save(model.state_dict(), 'results/mlm_model.pth')
    with open('results/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    # Inference & Visualization Data Collection
    model.eval()
    print("\n--- Reconstruction Results ---")
    
    # We will pick SAMPLE1 for detailed attention visualization
    viz_data = {}
    
    with torch.no_grad():
        for _, _, masked_txt, target_txt in data_samples:
            inputs = torch.tensor([tokenize(masked_txt)])
            pred_logits, attn = model(inputs)
            
            # Get the tokens
            pred_ids = pred_logits.argmax(dim=-1).squeeze().tolist()
            reconstructed = [rev_vocab[id] for id in pred_ids]
            
            print(f"Input : {masked_txt}")
            print(f"Pred  : {' '.join(reconstructed)}")
            
            # Find the Mask index to see if it filled it correctly
            # (Just printing full reconstruction is enough for verification)
            print("-" * 30)

            # Save attention for the first sample ("Transformers use [MASK] attention")
            if "Transformers" in masked_txt:
                viz_data['tokens'] = masked_txt.split()
                viz_data['attention'] = attn # [batch, num_heads, seq_len, seq_len]
    
    # Save visualization data
    torch.save(viz_data, 'results/viz_data.pt')
    print("Training Complete. Results saved.")

if __name__ == "__main__":
    train_and_evaluate()