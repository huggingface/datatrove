from datatrove.pipeline.readers import CSVReader, JsonlReader, ParquetReader, WarcReader
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.tokens.counter import TokensCounter
from datatrove.pipeline.filters.language_filter import LanguageFilter


# For ablation the non-truncated should yield 8B tokens,
# one doc has 6.1k tokens, so 8B / 6.1k = 1.3M docs
# For the truncated, we should yield the rest == 22B tokens
# one doc has 22895 tokens, so 22B / 22895 = 1M docs

class UpdateDocumentText(PipelineStep):
    def __init__(self):
        super().__init__()
    
    def run(self, data, rank: int = 0, world_size: int = 1):
        for sample in data:
            document_text = ""
            last_offset = 0
            if "pdf_metadata" not in sample.media[0].metadata or "page_offsets" not in sample.media[0].metadata["pdf_metadata"]:
                continue

            for offset in sample.media[0].metadata["pdf_metadata"]["page_offsets"]:
                text = sample.text[last_offset:offset]
                if not "<---- stop_reason_" in text:
                    document_text += text
                last_offset = offset
            sample.text = document_text
            yield sample

class Accumulator(PipelineStep):
    def __init__(self, unit: str):
        super().__init__()
    
    def run(self, data, rank: int = 0, world_size: int = 1):
        total_tokens = 0
        docs = 0
        for i,sample in enumerate(data):
            docs += 1
            tokens = sample.metadata.get("token_count", 0)
            total_tokens += tokens
            if i % 1000 == 0:
                print(f"Rank {rank} has {total_tokens} tokens, on average {total_tokens/docs} tokens per document")
            yield sample


def main():
    import argparse
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--input_folder", type=str, required=True)
    args = args_parser.parse_args()
    LocalPipelineExecutor(
        [
            JsonlReader(args.input_folder),
            UpdateDocumentText(),
            LanguageFilter(languages=["en"]),
            TokensCounter(count_eos_token=True, tokenizer_name_or_path="hynky/Llama-3.2-1B-no-bos"),
            Accumulator(unit="token"),
        ]
    ).run()

if __name__ == "__main__":
    main()
