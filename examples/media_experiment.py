from datatrove.executor.base import PipelineExecutor
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.filters import VideoFrozenFilter
from datatrove.pipeline.readers import VideoTripletReader


def run_step_1():
    video_triplet_reader = VideoTripletReader(
        data_folder="s3://amotoratolins/datatrovetest/", metadata_origin="youtube"
    )

    video_frozen_filter = VideoFrozenFilter()

    pipeline_1 = [video_triplet_reader, video_frozen_filter]

    # Create the executor with the pipeline
    executor_1: PipelineExecutor = LocalPipelineExecutor(pipeline=pipeline_1, workers=1, tasks=1)

    # Execute the pipeline
    # result = executor_1.run()
    executor_1.run()


#    # Additional debugging
#    for document in video_triplet_reader.read_file(None):
#        print(f"Document ID: {document.id}")
#        print(f"Text: {document.text[:100]}...")  # Print first 100 characters of text
#        print(f"Media: {document.media}")
#        print(f"Metadata: {document.metadata}")
#        print("-" * 80)


# Run the testing pipeline
run_step_1()
