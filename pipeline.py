# Importing Dependencies
from datasets import load_dataset
import re

from tokenizers import Tokenizer, trainers, pre_tokenizers, decoders, processors
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel

# Cleaning Function
# Cleaning Function
def clean_text(text):
    # Remove non-Hindi characters
    text = re.sub(r"[^ऀ-ॿ\s]", "", text)
    # Remove extra whitespaces
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Loading the dataset
dataset = load_dataset("oscar-corpus/OSCAR-2201",
                        use_auth_token=True,
                        language="hi",
                        streaming=True,
                        split="train")

# Clean the dataset and make a generator
def cleaned_dataset_generator(dataset):
    for d in dataset:
        yield clean_text(d['text'])

cleaned_dataset = cleaned_dataset_generator(dataset)

# Creating the tokenizer
tokenizer = Tokenizer(BPE())

# Initialize a pre-tokenizer
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    ByteLevel()
])

# Initialize a normalizer
tokenizer.normalizer = Sequence([
    NFKC(),
    Lowercase()
])

# Initialize a decoder
tokenizer.decoder = decoders.ByteLevel()

# Initialize a post-processor
tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

# Initialize a trainer
trainer = trainers.BpeTrainer(vocab_size=50000, min_frequency=2, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Train the tokenizer
tokenizer.train_from_iterator(cleaned_dataset, trainer)

# Saving the tokenizer
tokenizer.save("bpe_hindi.json")

print("Done")