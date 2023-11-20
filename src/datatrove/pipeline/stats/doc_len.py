from datatrove.pipeline.base import DocumentsPipeline, PipelineStep


class DocLenStats(PipelineStep):
    type = "📊 - STATS"
    name = "🤓 document length"

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        for doc in data:
            self.update_doc_stats(doc)
            yield doc
