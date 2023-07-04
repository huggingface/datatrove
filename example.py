from datatrove.data import Document
from datatrove.executor.base import PipelineExecutor
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.io import LocalOutputDataFolder
from datatrove.pipeline.tokens.tokenizer import DocumentTokenizer


samples = [
    Document(
        content=f"""{x}
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nam ut aliquam ante. Integer felis elit, auctor non volutpat ac, tempor in velit. Suspendisse nec turpis ante. Maecenas pretium ipsum ut arcu rhoncus placerat. Nunc semper mauris mattis, pretium nisi eget, bibendum ex. Quisque blandit, ex sit amet suscipit vestibulum, est nisi elementum nisl, id aliquam arcu lorem vitae arcu. Quisque interdum luctus mauris. Vestibulum tempor velit ac nibh tempus suscipit. Etiam eget risus lacus. Integer malesuada, nulla nec vulputate accumsan, ex eros placerat sem, vel viverra est lorem vel purus. Vestibulum volutpat purus a ex commodo, eget scelerisque nisl semper. Vestibulum nisi mauris, feugiat eu tincidunt non, faucibus id mauris. Fusce non massa tortor. Donec venenatis magna non elit tristique imperdiet.

Cras venenatis finibus lorem consequat consectetur. Nullam vehicula eget mauris ut fermentum. Nullam finibus quis lectus sed pellentesque. Phasellus consectetur consequat magna dignissim gravida. Maecenas commodo lacus eu ultrices finibus. Vivamus consectetur urna lacus, id condimentum ex bibendum non. Vestibulum neque mi, ultricies at ex in, interdum consectetur nisi. Donec in pulvinar quam. Donec placerat, dolor at varius vehicula, enim metus blandit lorem, sed condimentum turpis orci bibendum orci.

Fusce quam tellus, placerat eget ullamcorper eleifend, varius non mauris. Phasellus consectetur bibendum lacus, nec vulputate ligula interdum et. Nam at lorem iaculis, vestibulum lacus viverra, finibus ipsum. Aenean non tellus iaculis turpis malesuada efficitur. Aenean consectetur nunc ac mauris porttitor blandit. Integer cursus laoreet odio non laoreet. Etiam fermentum eu felis in ullamcorper. Donec a ipsum ac ante tincidunt suscipit. Nulla consectetur nunc metus, ut finibus arcu elementum id. Praesent sit amet elit semper, dignissim tortor in, pretium erat. Phasellus fermentum, nibh sed sagittis euismod, magna nibh imperdiet urna, scelerisque euismod urna sem in ex. Ut eu tellus pretium, egestas neque in, ullamcorper dui. Aliquam commodo pulvinar lectus, luctus tincidunt nisl luctus id. Vestibulum maximus laoreet erat et ullamcorper.

Donec orci lorem, cursus in urna eu, laoreet vulputate mauris. Aliquam ac urna vitae ante pretium gravida scelerisque eget nisl. Vestibulum id risus vitae libero luctus dignissim. Maecenas eget bibendum ligula, vel tristique leo. Nullam eu nisi sed dui condimentum faucibus. Pellentesque bibendum ligula non massa lacinia venenatis. Quisque sodales neque diam, vel tristique erat laoreet et. In sollicitudin ultricies pretium.

Aenean eget volutpat metus. Integer vitae sollicitudin sapien. Phasellus finibus risus vel dolor pellentesque consequat. Sed non posuere felis. In tortor justo, aliquet a libero et, bibendum laoreet lectus. Quisque sit amet justo nibh. Mauris aliquet, dui laoreet suscipit finibus, metus libero vehicula libero, vitae commodo felis leo vel urna. Proin urna lacus, hendrerit ac mauris et, suscipit tempor odio. Nunc eu ultricies nunc, vel mattis odio. Quisque efficitur facilisis tortor. Aliquam feugiat odio non ex malesuada porttitor. Aliquam sit amet augue pellentesque, porttitor elit nec, faucibus neque. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Suspendisse potenti. """,
        data_id=str(i),
        metadata={"mid": i % 10},
    )
    for i, x in enumerate(["sample 1", "some text", "more text"] * 500)
]

# output = LocalOutputDataFolder(path="/home/gui/hf_dev/datatrove/test_folder")
# print(output)
#
# with JsonlWriter(output, output_filename="${mid}.jsonl.gz") as writer:
#     for doc in samples:
#         writer.write(doc)
#
pipeline = [samples, DocumentTokenizer(LocalOutputDataFolder(path="/home/gui/hf_dev/datatrove/tokenized_test"))]

executor: PipelineExecutor = LocalPipelineExecutor(pipeline=pipeline, tasks=1)
executor.run()
