import argparse
import json
import logging
import os


FASHION_STYLES_DICT = {
    "americancasual": "american casual",
    "gorpcore": "outdoor",
}
LOGGER_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"

logger = logging.getLogger(__name__)
formatter = logging.Formatter(LOGGER_FORMAT)
logger.setLevel(logging.DEBUG)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, default="/opt/ml/processing")
    parser.add_argument("--prompt-prefix", type=str, default=None)
    parser.add_argument("--prompt-suffix", type=str, default=None)

    args, _ = parser.parse_known_args()

    inputs_dir = f"{args.base_dir}/inputs"
    output_dir = f"{args.base_dir}/output"

    metadata = {}
    with open(
        os.path.join(output_dir, "metadata.jsonl"), "w", encoding="utf-8"
    ) as output_file:
        for filename in os.listdir(inputs_dir):
            if filename.endswith(".out"):
                prompt_prefix = (
                    "" if args.prompt_prefix is None else f"{args.prompt_prefix} "
                )
                prompt_suffix = (
                    "" if args.prompt_suffix is None else f" {args.prompt_suffix}"
                )

                with open(
                    os.path.join(inputs_dir, filename), "r", encoding="utf-8"
                ) as input_file:
                    metadata["file_name"] = f"images/{filename[:-4]}"
                    split = filename.split("_")

                    if len(split) > 3:
                        fashion_style = FASHION_STYLES_DICT.get(split[2], split[2])
                        prompt_suffix += f", {fashion_style} fashion style"

                    prediction = json.load(input_file)
                    text = f"{prompt_prefix}{prediction['text']}{prompt_suffix}"
                    metadata["text"] = text.strip()
                    output_file.write(json.dumps(metadata) + "\n")
                    logger.info("The line has been written: %s", metadata)
