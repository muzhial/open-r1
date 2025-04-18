from typing import Dict, Any
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from trl import setup_chat_format, DataCollatorForCompletionOnlyLM
from trl.data_utils import apply_chat_template, maybe_convert_to_chatml


def formart_func(example: Dict[str, Any]) -> Dict[str, Any]:
    keep_conv_keys = ["from", "value"]
    convs = example["conversations"]
    system = example["system"]
    new_convs = [
        {"from": "system", "value": system},
    ]
    for conv in convs:
        new_conv = {k: v for k, v in conv.items() if k in keep_conv_keys}
        new_conv["from"] = new_conv["from"].lower()
        new_convs.append(new_conv)
    return {
        "conversations": new_convs,
    }


def tokenize(example, tokenizer):
    return tokenizer(example["text"])


if __name__ == "__main__":
    model_name = "/mnt/nas_data2/chenjn_workspace/datasets/HF/llama/Llama-3.2-3B-Instruct"
    model_name = "/mnt/nas_data2/chenjn_workspace/datasets/HF/Qwen/Qwen2.5-3B-Instruct"
    datafile = "examples/datas/multi_turn_conversation.json"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        print("pad_token is None, set pad_token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    else:
        print(f"pad_token is {tokenizer.pad_token}")

    dataset = load_dataset("json", data_files=datafile, split="train")

    dataset = dataset.map(
        formart_func,
        remove_columns=["system", "mask", "type", "category"]
    )
    dataset = dataset.map(
        maybe_convert_to_chatml,
        remove_columns="conversations" if "conversations" in dataset.column_names else None,
    )
    dataset = dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns="messages" if "messages" in dataset.column_names else None,
    )
    dataset = dataset.map(
        tokenize,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns="text" if "text" in dataset.column_names else None,
    )

    # llama3.1 and llama3.2
    # response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    # instruction_template = "<|start_header_id|>user<|end_header_id|>\n\n"
    # qwen
    response_template = "<|im_start|>assistant\n"
    instruction_template = "<|im_start|>user\n"

    # data_collator = DataCollatorForLanguageModeling(
    #     tokenizer=tokenizer,
    #     mlm=False,
    # )

    data_collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        instruction_template=instruction_template,
        tokenizer=tokenizer,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        collate_fn=data_collator,
    )
    for batch in dataloader:
        print("=" * 13)
        input_text = tokenizer.decode(batch["input_ids"][0].tolist())
        label_ids = batch["labels"][0][batch["labels"][0] != -100]
        output_text = tokenizer.decode(label_ids.tolist())
        print("-" * 13)
        print(input_text)
        print("-" * 13)
        print(output_text)
