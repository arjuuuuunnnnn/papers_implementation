import torch

def translate_sentence(sentence, src_vocab, trg_vocab, model, device, max_len=50):
    model.eval()

    # Tokenize and numericalize
    tokens = [token.lower() for token in sentence.split()]
    tokens = ['<sos>'] + tokens + ['<eos>']
    src_indices = [src_vocab.get(token, src_vocab['<unk>']) for token in tokens]

    src_tensor = torch.LongTensor(src_indices).unsqueeze(0).to(device)

    with torch.no_grad():
        encoder_output = model.encode(src_tensor)

    trg_indices = [trg_vocab['<sos>']]

    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indices).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model.decode(trg_tensor, encoder_output)
            output = model.linear(output)

        pred_token = output.argmax(2)[:,-1].item()
        trg_indices.append(pred_token)

        if pred_token == trg_vocab['<eos>']:
            break

    trg_tokens = [trg_vocab.get_idx_for_token(i) for i in trg_indices]

    return ' '.join(trg_tokens[1:-1])
