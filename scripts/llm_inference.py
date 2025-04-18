from dataclasses import dataclass, field, asdict

from pprint import pprint
from transformers import (
    HfArgumentParser,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)


def load_model(
    model_name: str,
    device: str = "auto",
) -> tuple["AutoModelForCausalLM", "AutoTokenizer"]:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        print(f"pad_token is None, set to eos_token: {tokenizer.eos_token}")
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


@dataclass
class ScriptArgs:
    model_name: str = field(default="/mnt/nas_data2/chenjn_workspace/datasets/HF/Qwen/Qwen2.5-3B-Instruct")
    prompt: str = field(default="朋友托我买房子，吃差价算违法吗")
    system_prompt: str = field(default=None)
    device: str = field(default="auto")
    max_new_tokens: int = field(default=512)
    do_sample: bool = field(default=True)
    top_p: float = field(default=0.9)
    temperature: float = field(default=0.6)


def main():
    parser = HfArgumentParser(ScriptArgs)
    args = parser.parse_args_into_dataclasses()[0]
    print(f"=> ScriptArgs:")
    pprint(asdict(args))

    model, tokenizer = load_model(args.model_name, args.device)

    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    if args.prompt:
        messages.append({"role": "user", "content": args.prompt})

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    print("-" * 42)
    print(text)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        top_p=args.top_p,
        temperature=args.temperature,
    )

    generated_ids = model.generate(
        **model_inputs,
        generation_config=generation_config
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print("-" * 42)
    print(response)


if __name__ == "__main__":
    main()
