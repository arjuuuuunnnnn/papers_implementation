from tokenizers import Tokenizer
from tokenizers.processors import TemplateProcessing
from tokenizers import normalizers
from tokenizers.normalizers import Lowercase, NFD, StripAccents
from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from transformers import AutoTokenizer

def get_english_tokenizer(data, vocab_size):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.Sequence([NFD(), StripAccents(), Lowercase()])
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[BOS]", "[EOS]"])

    def batch_iterator(batch_size=10000):
        for i in range(0, len(data), batch_size):
            yield [item[0] for item in data[i : i + batch_size]]  # Only English

    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer, length=len(data))
    
    tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A [EOS]",
        special_tokens=[
            ("[BOS]", tokenizer.token_to_id("[BOS]")),
            ("[EOS]", tokenizer.token_to_id("[EOS]")),
        ],
    )
    return tokenizer

def get_kannada_tokenizer(model_name):
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

def get_translation_tokenizers(data, english_vocab_size, kannada_model_name):
    english_tokenizer = get_english_tokenizer(data, english_vocab_size)
    kannada_tokenizer = get_kannada_tokenizer(kannada_model_name)
    return english_tokenizer, kannada_tokenizer
