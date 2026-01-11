import torch
import matplotlib.pyplot as plt
import pickle
from transformer import TransformerSeq2Seq
from attention_masks import create_causal_mask
import os
import numpy as np

def plot_attention(attention, sentence, predicted_sentence, filename="samples/attention_heatmap.png"):
    """
    Visualizes the attention weights.
    attention: (num_heads, tgt_len, src_len) - using the first head or averaging
    """
    attn = attention.squeeze(0) # [num_heads, tgt_len, src_len]
    
    # Use the first head for visualization
    attn_head = attn[0].detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.imshow(attn_head, cmap='viridis', aspect='equal')
    fig.colorbar(cax, ax=ax)

    # Set ticks
    ax.set_xticks(np.arange(len(sentence)))
    ax.set_yticks(np.arange(len(predicted_sentence)))
    
    # Set labels
    ax.set_xticklabels(sentence)
    ax.set_yticklabels(predicted_sentence)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    ax.set_xlabel('Source Sequence')
    ax.set_ylabel('Target Sequence')
    ax.set_title('Encoder-Decoder Attention Map (Head 0)')
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Attention map saved to {filename}")
    plt.close()

def visualize():
    # Load Dicts
    try:
        with open("vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
    except FileNotFoundError:
        print("Run train.py first.")
        return

    inv_vocab = {v: k for k, v in vocab.items()}
    
    # Load Model
    model = TransformerSeq2Seq(len(vocab), len(vocab), d_model=128)
    try:
        model.load_state_dict(torch.load("transformer_seq2seq.pth"))
    except FileNotFoundError:
        print("Model file not found.")
        return
        
    model.eval()

    # Input Data
    input_text = "AI improves healthcare"
    src_tokens = input_text.split()
    src_indices = [vocab[w] for w in src_tokens if w in vocab]
    
    if not src_indices:
        print("Input words not in vocab")
        return

    src = torch.tensor([src_indices])
    
    print(f"Generating for: '{input_text}'...")
    
    generated_indices = [vocab["<SOS>"]]
    decoded_words = []
    
    # Generation Loop
    for _ in range(20):
        tgt = torch.tensor([generated_indices])
        tgt_mask = create_causal_mask(tgt.size(1))
        
        with torch.no_grad():
            output, attn_weights = model(src, tgt, None, tgt_mask)
            
            next_token_logits = output[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1).item()
            
            if next_token == vocab["<EOS>"]:
                break
            
            generated_indices.append(next_token)
            decoded_words.append(inv_vocab[next_token])

    # Final forward pass to get full attention
    tgt = torch.tensor([generated_indices])
    tgt_mask = create_causal_mask(tgt.size(1))
    
    with torch.no_grad():
        _, attn_weights = model(src, tgt, None, tgt_mask)
    
    tgt_labels = ["<SOS>"] + decoded_words
    
    # Filter src_tokens if some were skipped (though our logic above skipped unknown words)
    # The plot logic assumes strict alignment.
    
    plot_attention(attn_weights, src_tokens, tgt_labels)

if __name__ == "__main__":
    if not os.path.exists("samples"):
        os.makedirs("samples")
    visualize()
