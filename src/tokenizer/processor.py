import tokenizers
from loader.processor import Loader
from models.processor import Processor
from loguru import logger


class Tokenizer(Processor):

    def compute(self) -> None:
        tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
        tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=True)
        tokenizer.decoder = tokenizers.decoders.ByteLevel()
        VOCAB_SIZE = 10000
        trainer = tokenizers.trainers.BpeTrainer(
            vocab_size=VOCAB_SIZE,
            special_tokens=["[pad]", "[eos]"],
            show_progress=True
        )
        text = Loader(name="loader").compute()
        tokenizer.train_from_iterator(text, trainer=trainer)
        tokenizer.enable_padding(pad_id=tokenizer.token_to_id("[pad]"), pad_token="[pad]")
        # Save the trained tokenizer
        tokenizer.save("gutenberg_tokenizer.json", pretty=True)
        logger.debug("Tokenizer json file has been trained!")

if __name__ == "__main__":
    p = Tokenizer(name="tokenizer")
    p.compute()