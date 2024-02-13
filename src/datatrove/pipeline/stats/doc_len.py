from datatrove.pipeline.base import DocumentsPipeline, PipelineStep


class DocLenStats(PipelineStep):
    """Pipeline step to compute the length of each document in a pipeline.
    Will add a "length" metadata to each document with the length of the text in characters and (if available) tokens.
    """

    type = "📊 - STATS"
    name = "🤓 document length"

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        for doc in data:
            self.update_doc_stats(doc)
            yield doc
