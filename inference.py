import torch
from torch.nn import functional as F
from collections import defaultdict

from src.transformer import Transformer

def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    hyperparams = checkpoint['hyperparameters']
    model = Transformer(
        src_vocab_size=hyperparams['src_vocab_size'],
        trg_vocab_size=hyperparams['trg_vocab_size'],
        d_model=hyperparams['d_model'],
        num_heads=hyperparams['num_heads'],
        num_layers=hyperparams['num_layers'],
        dropout=hyperparams['dropout']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, checkpoint['src_vocab'], checkpoint['trg_vocab']

def create_reverse_vocab(vocab):
    """Create a reverse vocabulary mapping (index to token)"""
    return {idx: token for token, idx in vocab.items()}

def preprocess_sentence(sentence, vocab, max_length=50):
    """Convert sentence to tensor using vocabulary"""
    tokens = sentence.lower().split()#simple whitespace tok

    # Add special tokens
    tokens = ['<sos>'] + tokens + ['<eos>']

    # Convert to indices
    indices = [vocab.get(token, vocab['<unk>']) for token in tokens]

    # Pad or truncate if needed
    if len(indices) > max_length:
        indices = indices[:max_length-1] + [vocab['<eos>']]
    else:
        indices = indices + [vocab['<pad>']] * (max_length - len(indices))

    return torch.LongTensor(indices).unsqueeze(0)  # Add batch dimension

def translate_sentence(model, sentence, src_vocab, trg_vocab, trg_reverse_vocab, device, max_length=50):
    """Translate a single sentence"""
    src_tensor = preprocess_sentence(sentence, src_vocab).to(device)

    with torch.no_grad():
        encoder_output = model.encode(src_tensor)

    trg_indices = [trg_vocab['<sos>']]

    for i in range(max_length):
        trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model.decode(trg_tensor, encoder_output)
            output = model.linear(output)
            probs = F.softmax(output, dim=-1)
            # Get the most likely next token
            next_token = output.argmax(2)[:, -1].item()
            trg_indices.append(next_token)
            # Stop if it predicts <eos>
            if next_token == trg_vocab['<eos>']:
                break

    trg_tokens = [trg_reverse_vocab[idx] for idx in trg_indices[1:-1]]  # Remove <sos> and <eos>

    return ' '.join(trg_tokens)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model, src_vocab, trg_vocab = load_model('transformer.pt', device)
    trg_reverse_vocab = create_reverse_vocab(trg_vocab)

    test_sentence = "Hello world"
    translation = translate_sentence(
        model,
        test_sentence,
        src_vocab,
        trg_vocab,
        trg_reverse_vocab,
        device
    )
    print(f"Source: {test_sentence}")
    print(f"Translation: {translation}")
    print("-" * 50)

if __name__ == "__main__":
    main()
