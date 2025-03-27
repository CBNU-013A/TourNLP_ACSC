import torch
import glob
import re
import os

from model.src.processor import Processor, convert_examples_to_features
from common.schema import InputExample
from common.utils import(
    CONFIG_CLASSES,
    TOKENIZER_CLASS,
    MODEL_FOR_TOKEN_CLASSIFICATION
)

class Tester:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

        self.processor = Processor(args)
        self.categories = self.processor._get_category()
        self.label_map = {0: "pos", 1: "neg", 2: "neu", 3: "none"}

        # Find latest checkpoint
        checkpoint_dirs = glob.glob(os.path.join(args.output_dir, "checkpoint-*"))
        latest_ckpt = sorted(
            checkpoint_dirs, key=lambda x: int(re.findall(r"checkpoint-(\d+)", x)[0])
        )[-1] if checkpoint_dirs else args.output_dir

        self.tokenizer = TOKENIZER_CLASS[args.model_type].from_pretrained(args.model_name_or_path)
        self.model = MODEL_FOR_TOKEN_CLASSIFICATION[args.model_type].from_pretrained(latest_ckpt)
        self.model.to(self.device)
        self.model.eval()

    def test(self):
        while True:
            sentence = input("입력 문장 (종료하려면 엔터): ").strip()
            if not sentence:
                break

            examples = [
                InputExample(guid=f"test-{i}", sentence=sentence, category=cat, sentiment="none")
                for i, cat in enumerate(self.categories)
            ]

            features = convert_examples_to_features(self.args, examples, self.tokenizer, self.args.max_seq_len)
            for ex, feat in zip(examples, features):
                input_ids = torch.tensor([feat.input_ids], device=self.device)
                attention_mask = torch.tensor([feat.attention_mask], device=self.device)
                token_type_ids = torch.tensor([feat.token_type_ids], device=self.device)

                with torch.no_grad():
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids
                    )
                    pred = torch.argmax(outputs.logits, dim=1).item()
                    sentiment = self.label_map[pred]

                print(f"[{ex.category}] → {sentiment}")