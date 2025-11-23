import pickle
def load_vocabulary(vocab_dir='../Data/vocabulary'):
    #Loading vocabulary mappings
    with open(f'{vocab_dir}/vocabulary.pkl', 'rb') as f:
        vocab = pickle.load(f)
    return vocab

def encode_text(text, char_to_idx, add_sos_eos=False, unk_idx=3):
    #Encode text to indices
    encoded = [char_to_idx.get(char, unk_idx) for char in text]
    
    if add_sos_eos:
        sos_idx = char_to_idx.get('<SOS>', 1)
        eos_idx = char_to_idx.get('<EOS>', 2)
        encoded = [sos_idx] + encoded + [eos_idx]
    
    return encoded

def decode_indices(indices, idx_to_char, remove_special=True):
    #Decode indices back to text
    decoded = [idx_to_char.get(idx, '<UNK>') for idx in indices]
    
    if remove_special:
        # Removing special tokens
        special = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
        decoded = [char for char in decoded if char not in special]
    
    return ''.join(decoded)

if __name__ == '__main__':
    vocab = load_vocabulary()

    text = "ABC123"
    encoded = encode_text(text, vocab['char_to_idx'])
    print(f"Text: {text}")
    print(f"Encoded: {encoded}")

    decoded = decode_indices(encoded, vocab['idx_to_char'])
    print(f"Decoded: {decoded}")