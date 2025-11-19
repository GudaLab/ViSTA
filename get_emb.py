# get_emb.py

import sys
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoConfig, BertForSequenceClassification
from peft import PeftModel, PeftConfig

# ----------------------------
# ======== SETTINGS ==========
# ----------------------------

# Path to base DNABERT-2 model
base_model_dir = "/pt_ViSTA"

# Path to LoRA adapter (fine-tuned)
adapter_dir = "/ft_ViSTA/best_model"

# ----------------------------
# ======== FUNCTIONS =========
# ----------------------------

def load_model_and_tokenizer():
    peft_config = PeftConfig.from_pretrained(adapter_dir)
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir)

    config = AutoConfig.from_pretrained(base_model_dir)
    config.num_labels = 4  # Adjust if needed

    base_model = BertForSequenceClassification.from_pretrained(
        base_model_dir, config=config, ignore_mismatched_sizes=True
    )
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()
    return model, tokenizer

def get_cls_embedding(sequence, layer=-1, max_length=512):
    inputs = tokenizer(sequence, return_tensors="pt", padding="max_length",
                       truncation=True, max_length=max_length)
    inputs['output_hidden_states'] = True

    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.hidden_states
        selected_hidden = hidden_states[layer]  # shape: [1, seq_len, hidden_dim]
        cls_embedding = selected_hidden[0, 0, :].cpu().numpy()
    return cls_embedding

# ----------------------------
# ======== MAIN ==============
# ----------------------------

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("\nUsage:")
        print("  python get_emb.py \"ATCGTCGT\" layer_number max_length output.csv")
        print("\nArguments:")
        print("  sequence     - DNA sequence (e.g., 'AGTCAGTCA')")
        print("  layer_number - Transformer layer (e.g., -1 for last, 0 for embeddings, 1-12 for internal layers)")
        print("  max_length   - Max length for tokenizer padding/truncation (e.g., 512)")
        print("  output.csv   - Path to save output embedding as CSV")
        sys.exit(1)

    sequence = sys.argv[1]
    layer = int(sys.argv[2])
    max_length = int(sys.argv[3])
    output_path = sys.argv[4]

    model, tokenizer = load_model_and_tokenizer()
    embedding = get_cls_embedding(sequence, layer=layer, max_length=max_length)

    # Print to console
    print(f"\n[CLS] Embedding (Layer {layer}, MaxLen {max_length})")
    print(f"Shape: {embedding.shape}")
    print("Embedding (first 10 dims):", embedding[:10])

    # Save to CSV
    dim_names = [f"dim_{i}" for i in range(len(embedding))]
    df = pd.DataFrame([embedding], columns=dim_names)
    df.insert(0, "input_sequence", sequence)
    df.insert(1, "layer", layer)
    df.insert(2, "max_length", max_length)

    df.to_csv(output_path, index=False)
    print(f"\n? Full embedding saved to: {output_path}")
