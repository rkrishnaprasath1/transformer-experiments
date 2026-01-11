import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize():
    if not os.path.exists('results/viz_data.pt'):
        print("Run train_mlm.py first.")
        return

    data = torch.load('results/viz_data.pt')
    tokens = data['tokens']
    attn_weights = data['attention'] # [batch, num_heads, seq_len, seq_len]
    
    # Take first item in batch, first head
    # attn_weights shape: [1, 8, seq_len, seq_len]
    attn = attn_weights.squeeze(0)[0].detach().cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.imshow(attn, cmap='viridis', aspect='equal')
    fig.colorbar(cax, ax=ax)

    # Set ticks
    ax.set_xticks(np.arange(len(tokens)))
    ax.set_yticks(np.arange(len(tokens)))
    
    # Set labels
    ax.set_xticklabels(tokens)
    ax.set_yticklabels(tokens)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    ax.set_xlabel('Key Sequence')
    ax.set_ylabel('Query Sequence')
    ax.set_title('Self-Attention Map (Head 0)')
    
    plt.tight_layout()
    plt.savefig('results/attention_heatmap.png')
    print("Attention heatmap saved to results/attention_heatmap.png")
    plt.close()

if __name__ == "__main__":
    visualize()
