# DataTrove

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

DataTrove 是一个专为超大规模文本数据处理、过滤与去重而设计的 Python 核心基础库。它提供了一系列开箱即用、预置的高频数据处理模块，并辅以高度灵活的框架，便于开发者轻松扩展自定义功能。

DataTrove 的处理流水线具有平台无关（platform-agnostic）的特性，无论是本地开发机还是大型 Slurm 集群，代码均可即开即用、无缝运行。其对内存占用的极致优化（相对极低）以及流水线分步设计，使其成为应对海量数据负载的理想之选（例如处理前沿大语言模型 LLM 的万亿级预训练语料）。

得益于 [fsspec](https://filesystem-spec.readthedocs.io/en/latest/) 的底层支持，DataTrove 原生支持本地文件系统、远程对象存储（如 S3、HF Buckets）及各类分布式存储系统。

## 目录（Table of contents）

<!-- toc -->

- [安装指南（Installation）](#installation)
- [快速上手示例（Quickstart examples）](#quickstart-examples)
- [核心术语（Terminology）](#terminology)
- [流水线（Pipeline）](#pipeline)
  * [DataTrove Document 文档对象](#datatrove-document)
  * [流水线处理模块类型](#types-of-pipeline-blocks)
  * [完整流水线构建](#full-pipeline)
- [执行器（Executors）](#executors)
  * [LocalPipelineExecutor 本地执行器](#localpipelineexecutor)
  * [SlurmPipelineExecutor Slurm 集群执行器](#slurmpipelineexecutor)
  * [RayPipelineExecutor Ray 集群执行器](#raypipelineexecutor)
  * [JobsPipelineExecutor HF Jobs 执行器](#jobspipelineexecutor)
- [日志系统（Logging）](#logging)
- [DataFolder 与路径规范（DataFolder / paths）](#datafolder--paths)
- [实践指南（Practical guides）](#practical-guides)
  * [数据读取（Reading data）](#reading-data)
  * [合成数据生成（Synthetic data generation）](#synthetic-data-generation)
    + [自定义 Rollout 执行函数](#custom-rollouts)
    + [开箱即用的大规模生成脚本](#ready-to-use-generation-script)
    + [进阶配置与断点恢复](#advanced-configuration)
    + [进度监控与数据集卡片](#progress-monitoring)
    + [性能基准测试（Benchmarking）](#benchmarking)
  * [文本提取（Extracting text）](#extracting-text)
  * [数据过滤（Filtering data）](#filtering-data)
  * [数据保存（Saving data）](#saving-data)
  * [数据去重（Deduplicating data）](#deduplicating-data)
  * [数据汇总统计（Summary Statistics）](#summary-statistics)
  * [自定义处理模块（Custom blocks）](#custom-blocks)
    + [直接传入数据集合](#simple-data)
    + [自定义函数处理](#custom-function)
    + [完整自定义模块开发](#custom-block)
- [参与贡献（Contributing）](#contributing)
- [引用（Citation）](#citation)

<!-- tocstop -->

<span id="installation"></span>
## 安装指南（Installation）

运行环境要求 Python 3.10+。

```bash
uv sync
```

支持按需安装不同功能特性包（可重复使用 `--extra` 组合多个特性，例如 `uv sync --extra processing --extra s3`）：
- `all` 安装所有依赖项：`uv sync --extra all`
- `io` 读取 `warc/arc/wet` 文件及 Arrow/Parquet/[优化版 Parquet](https://huggingface.co/docs/hub/en/datasets-libraries#optimized-parquet-files) 格式所需依赖：`uv sync --extra io`
- `processing` 文本提取、规则过滤及词元化（Tokenization）相关依赖：`uv sync --extra processing`
- `s3` AWS S3 对象存储支持：`uv sync --extra s3`
- `cli` 命令行实用工具：`uv sync --extra cli`
- `ray` Ray 分布式计算引擎支持：`uv sync --extra ray`
- `inference` 大模型推理与合成数据生成流水线：`uv sync --extra inference`
- `decont` 借助 lighteval 进行基准测试数据去污染（Decontamination）：`uv sync --extra decont`
- `multilingual` 多语言文本处理依赖：`uv sync --extra multilingual`

<span id="quickstart-examples"></span>
## 快速上手示例（Quickstart examples）

可参考以下精选官方[示例程序](examples)：
- [fineweb.py](examples/fineweb.py)：完整复现顶尖开源高质量预训练数据集 [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) 的全流程流水线
- [process_common_crawl_dump.py](examples/process_common_crawl_dump.py)：读取 Common Crawl WARC 归档文件、提取网页正文文本、执行质量过滤并将清洗结果存储至 S3 的完整端到端流水线（支持在 Slurm 上运行）
- [tokenize_c4.py](examples/tokenize_c4.py)：直接从 Hugging Face Hub 流式读取数据，使用 `gpt2` 分词器对 C4 数据集的英文部分进行大规模词元化
- [estimate_tokens.py](examples/estimate_tokens.py)：估算超大规模 Hugging Face 数据集的词元（Token）总量——在从数万亿级 Token 语料库中创建随机打乱子集（例如抽取 100B Token）时，用于精确设置 `SamplerFilter` 采样率。通过流式读取各数据集少量样本收敛得到每篇文档的平均 Token 数，再乘以总样本行数
- [smol_data.py](examples/smol_data.py)：针对多个大型 Hugging Face 数据集构建 ~100B Token 子集、50-30-20 混合比例配比及洗牌（Shuffled）变体
- [minhash_deduplication.py](examples/minhash_deduplication.py)：基于 MinHash（最小哈希）对海量文本执行高效近似去重的完整流水线
- [sentence_deduplication.py](examples/sentence_deduplication.py)：基于句子级别的精确去重示例
- [exact_substrings.py](examples/exact_substrings.py)：运行 ExactSubstr（精确子串去重）的示例（依赖 [该开源仓库](https://github.com/google-research/deduplicate-text-datasets)）
- [finephrase.py](examples/inference/finephrase.py)：使用多提示词模板在大规模集群上批量生成高质量合成数据集的独立示例

<span id="terminology"></span>
## 核心术语（Terminology）

- `pipeline`（流水线）：待按序执行的数据处理步骤列表（例如读取数据、规则过滤、写入磁盘等）。
- `executor`（执行器）：在特定计算环境（如本地多核机器、Slurm 集群、Ray 集群等）上调度并运行特定流水线的执行引擎。
- `job`（作业）：在指定执行器上运行某个完整流水线的生命周期实例。
- `task`（任务）：一个 `job` 通常由多个并行的 `task` 组成，用于实现大规模并行化执行，通常每个 `task` 分配处理一个数据分片（`shard`）。DataTrove 会自动追踪已成功完成的任务；当因异常中断重新启动时，仅会重新执行尚未完成的任务。
- `file`（单个文件）：单个输入文件（如 `.json`、`.csv`、`.parquet` 等）。
> [!TIP]
> **提示**：每个文件由单个 `task` 独立处理。DataTrove 不会自动将单个大文件拆分为多个子部分。因此，为了充分发挥多核/多节点并行能力，**建议组织为若干大小适中的文件，而非一个巨大的单个文件**。
- `shard`（数据分片）：分配给特定 `task` 处理的一组输入数据（通常是一组 `file` 文件列表）。每个 `task` 将从完整输入文件列表中获取互不重叠的数据分片。
- `worker`（工作节点/执行单元）：同时执行单个任务的计算资源单元。例如，如果你拥有一台 50 核 CPU 的机器，你可以配置 `workers=50` 的 `LocalPipelineExecutor`，同时并发执行 50 个 `task`（每个 CPU 核心处理一个任务）。某个 `worker` 完成当前任务后，会自动领取下一个等待执行的任务。

> [!TIP]
> `tasks` 的数量决定了任务的并行度以及每个独立处理单元耗时。如果任务数量较少（即每个任务分配处理大量文件），一旦中途出错中断，重新执行将不得不从头处理该大批次；反之，若划分为较多的小型任务，每个失败任务重新执行所需的时间将大幅缩减。

> [!CAUTION]
> 如果设置的 `tasks` 数量大于文件总数 `files`，部分任务将分配不到任何数据文件，因此通常**无需将 `tasks` 设置得比 `files` 更多**。

### 示例场景（Example）

假设需要处理 **10,000** 个数据文件（`files`），机器拥有 **100** 个 CPU 核心（即可用计算 `workers` 为 100）。若选择设置 **1,000** 个任务（`tasks`），则每个任务分配处理一个包含 10 个文件的数据分片（`shard`）。配置 `workers=100` 意味着集群可以同时并行处理 **100** 个任务。

<span id="pipeline"></span>
## 流水线（Pipeline）

<span id="datatrove-document"></span>
### DataTrove Document 文档对象

流水线中的各个处理组件均基于 DataTrove 标准的 [`Document`](src/datatrove/data.py) 格式流式处理数据：
- `text`：每个样本的实际正文文本内容（字符串）
- `id`：当前样本的全局唯一 ID 标识（字符串）
- `metadata`：键值字典，用于存储与该文档关联的任意扩展元数据

<span id="types-of-pipeline-blocks"></span>
### 流水线处理模块类型

流水线中的每个处理组件（Block）均以 `Document` 的生成器（Generator）作为输入，经过相应处理后，返回另一个 `Document` 生成器：
- **[readers（数据读取）](src/datatrove/pipeline/readers)**：从各种数据存储格式中读取样本，并逐个生成 `Document`
- **[writers（数据写出）](src/datatrove/pipeline/writers)**：将处理后的 `Document` 以指定格式持久化到磁盘或对象存储中
- **[extractors（文本抽取）](src/datatrove/pipeline/extractors)**：从原始多媒体或标记格式（如网页 HTML）中抽取纯文本内容
- **[filters（质量过滤）](src/datatrove/pipeline/filters)**：根据设定的启发式规则或分类模型过滤并剔除不合格的 `Document`
- **[stats（统计分析）](src/datatrove/pipeline/stats)**：用于对整个数据集提取全方位统计特征与质量画像的组件
- **[tokens（分词处理）](src/datatrove/pipeline/tokens)**：负责文档词元化（Tokenize）及 Token 数量统计
- **[dedup（数据去重）](src/datatrove/pipeline/dedup)**：负责执行精准或模糊去重的专用处理组件

<span id="full-pipeline"></span>
### 完整流水线构建

流水线在代码中直接以 Python 列表的形式进行声明。例如，以下流水线将从磁盘读取 CSV 数据、随机过滤淘汰一部分文档，并将保留结果写入磁盘为 JSONL 格式：

```python
from datatrove.pipeline.readers import CSVReader
from datatrove.pipeline.filters import SamplerFilter
from datatrove.pipeline.writers import JsonlWriter

pipeline = [
    CSVReader(
        data_folder="/my/input/path"
    ),
    SamplerFilter(rate=0.5),
    JsonlWriter(
        output_folder="/my/output/path"
    )
]
```

<span id="executors"></span>
## 执行器（Executors）

DataTrove 的处理流水线具有平台无关性，同一套流水线步骤无需修改即可无缝迁移到不同的运行环境中。不同的计算基础设施对应各自专门的 `PipelineExecutor`。

所有执行器通用的核心参数：
- `pipeline`：包含待运行流水线各个处理步骤的列表
- `logging_dir`：日志输出路径（DataFolder 格式），用于存放日志文件、统计信息与完成标记。**请勿在不同流水线或作业间复用同一文件夹**，否则会覆盖统计数据、日志及完成标记
- `skip_completed`（布尔值，默认为 `True`）：DataTrove 会在本地/远程记录已完成任务；重新发起作业时将自动跳过已完成项。设为 `False` 可禁用该行为强制重跑
- `randomize_start_duration`（整数，默认为 `0`）：各任务启动时的最大随机延迟秒数，避免所有并发任务在同一瞬间同时启动对存储或 API 造成突发冲击

调用执行器对象的 `run()` 方法即可启动流水线。

> [!TIP]
> DataTrove 通过在 `${logging_dir}/completions` 目录下创建以任务编号命名的空标记文件来记录已成功完成的任务。作业执行结束后，如果某些任务不幸失败，你**只需直接重新运行完全相同的执行器配置**，DataTrove 会自动识别并仅重新执行此前未成功的任务。

> [!CAUTION]
> 在因部分任务失败而重新拉起流水线时，**请切勿更改任务总数 `tasks`**，否则会导致输入文件的分片哈希映射发生变动，破坏断点续跑的一致性。

<span id="localpipelineexecutor"></span>
### LocalPipelineExecutor 本地执行器

在单机本地环境上调度运行流水线。
参数选项：
- `tasks`：需要运行的任务总数
- `workers`：同时并发运行的任务数。若设为 `-1`，则不设限制；若 `> 1`，则采用 Python 多进程（Multiprocessing）并发执行各任务
- `start_method`：启动多进程池的方式（例如 `"fork"`、`"spawn"`）。当 `workers` 为 1 时忽略

<details>
  <summary>查看本地执行器示例代码</summary>

```python
from datatrove.executor import LocalPipelineExecutor
executor = LocalPipelineExecutor(
    pipeline=[
        ...
    ],
    logging_dir="logs/",
    tasks=10,
    workers=5
)
executor.run()
```
</details>

<details>
  <summary>多节点跨机并行模式（Multi-node parallelism）</summary>

你可以通过 `local_tasks` 和 `local_rank_offset` 参数，让不同计算节点分别承载总任务的指定区间。在每台机器上，使用以下参数启动：
- `tasks`：全局跨所有机器执行的总任务数。**该值在各节点上必须严格保持一致，否则各节点分片分配可能重叠！** 示例：500
- `local_tasks`：当前这台具体机器负责执行的任务数量（各机器可配置不同值）。示例：100
- `local_rank_offset`：当前机器所执行任务的起始 Rank 偏移量。例如当前是启动作业的第 3 台机器，前两台机器分别执行了 250 和 150 个任务，则本机器应配置为 `400`。

若需获取全量合并统计数据，只需在包含所有机器统计输出的路径上手动调用 `merge_stats` 脚本进行合并。
</details>

<span id="slurmpipelineexecutor"></span>
### SlurmPipelineExecutor Slurm 集群执行器

在 Slurm 高性能计算集群上调度流水线，利用 Slurm 作业阵列（Job Arrays）组织和管理大规模任务。
核心参数：
- `tasks`：运行的任务总数（**必填**）
- `time`：Slurm 任务执行时间限制字符串（**必填**，例如 `"10:00:00"`）
- `partition`：Slurm 分区名称（**必填**）
- `workers`：同时并发运行的最大任务数。若为 `-1`，则不设并发限制（由 Slurm 调度器自行决定；默认：`-1`）
- `job_name`：Slurm 作业名称（默认：`"data_processing"`）
- `depends`：另一个 `SlurmPipelineExecutor` 实例，表示流水线之间的依赖关系（当前流水线仅在前置依赖流水线成功执行完成后才会开始运行）
- `sbatch_args`：传递给 `sbatch` 命令的附加参数字典
- `slurm_logs_folder`：Slurm 日志存放路径。如果 `logging_dir` 为本地路径，日志保存在 `logging_dir/slurm_logs`；否则将作为当前工作目录的子目录保存

<details>
  <summary>更多高级 Slurm 配置选项</summary>

- `cpus_per_task`：每个任务分配的 CPU 核心数（默认：`1`）
- `qos`：Slurm QOS 服务质量等级（默认：`"normal"`）
- `mem_per_cpu_gb`：每个 CPU 核心分配的内存容量（GB，默认：`2`）
- `env_command`：激活 Python 虚拟环境的自定义 shell 命令（如有需要）
- `condaenv`：待激活的 Conda 环境名称
- `venv_path`：待激活的 Python 虚拟环境路径
- `max_array_size`：由 `$ scontrol show config` 查得的 _MaxArraySize_ 上限。当任务数超出此限制时，会自动拆分为多个阵列作业提交（默认：`1001`）
- `max_array_launch_parallel`：因超额需要拆分多个作业时，是一次性全部提交（并行）还是按序等待执行（默认：`False`）
- `stagger_max_array_jobs`：当 `max_array_launch_parallel=True` 时，各并行作业提交之间的缓冲间隔秒数（默认：`0`）
- `run_on_dependency_fail`：当前置依赖作业执行失败时是否仍然继续执行（默认：`False`）
- `randomize_start`：将各任务的启动时间在一个 ~3 分钟的时间窗口内打散随机化。在大量任务同时高并发访问 S3 对象存储等场景下尤为实用（默认：`False`）
</details>

<details>
  <summary>查看 Slurm 执行器示例代码</summary>

```python
from datatrove.executor import SlurmPipelineExecutor
executor1 = SlurmPipelineExecutor(
    pipeline=[
        ...
    ],
    job_name="my_cool_job1",
    logging_dir="logs/job1",
    tasks=500,
    workers=100,  # 省略该参数则一次性全部提交
    time="10:00:00",  # 10 小时
    partition="hopper-cpu"
)
executor2 = SlurmPipelineExecutor(
    pipeline=[
        ...
    ],
    job_name="my_cool_job2",
    logging_dir="logs/job2",
    tasks=1,
    time="5:00:00",  # 5 小时
    partition="hopper-cpu",
    depends=executor1  # 本流水线仅在 executor1 顺利完成后才会启动
)
# executor1.run()
executor2.run()  # 调用此处将自动级联触发依赖的 executor1，无需重复显式启动
```
</details>

<span id="raypipelineexecutor"></span>
### RayPipelineExecutor Ray 集群执行器

在 Ray 集群上启动流水线，通过 Ray Task 实现分布式弹性并发执行。
参数选项：
- `tasks`：需要执行的任务总数
- `workers`：同时并行执行的最大任务数。若为 `-1`，则不设限制（默认：`-1`）
- `depends`：依赖的前置 `RayPipelineExecutor` 实例（确保按序执行）

<details>
  <summary>更多 Ray 配置选项</summary>

- `cpus_per_task`：每个任务分配的 CPU 核心数（默认：`1`）
- `mem_per_cpu_gb`：每个 CPU 分配的内存（GB，默认：`2`）
- `ray_remote_kwargs`：传递给 `@ray.remote` 装饰器的附加关键字参数字典
</details>

<details>
  <summary>查看 Ray 执行器示例代码</summary>

```python
import ray
from datatrove.executor import RayPipelineExecutor
ray.init()
executor = RayPipelineExecutor(
    pipeline=[
        ...
    ],
    logging_dir="logs/",
    tasks=500,
    workers=100,  # 省略该参数则一次性全部拉起
)
executor.run()
```
</details>

<span id="jobspipelineexecutor"></span>
### JobsPipelineExecutor HF Jobs 执行器

**实验性功能（未来可能会调整或移除）**：在 [Hugging Face Jobs](https://huggingface.co/docs/huggingface_hub/en/guides/jobs) 上调度运行流水线。其协同模型与 Slurm 类似，但每个任务块均在独立的云端托管 Job 容器中执行。详情请参阅 [`JobsPipelineExecutor` 文档注释](src/datatrove/executor/jobs.py)及 `*_jobs.py` [示例文件](examples)。

<span id="logging"></span>
## 日志系统（Logging）

对于 `logging_dir` 配置为 **mylogspath/exp1** 的流水线，DataTrove 会自动生成如下结构清晰的日志与状态目录：

<details>
  <summary>查看日志目录树结构</summary>

```
└── mylogspath/exp1
    │── executor.json ⟵ 执行器配置选项与流水线步骤的 JSON 归档快照
    │── launch_script.slurm ⟵ 生成并用于提交任务的 Slurm 脚本（若在 Slurm 上运行）
    │── executor.pik ⟵ 执行器序列化二进制对象（若在 Slurm 上运行）
    │── ranks_to_run.json ⟵ 当前正在调度的任务列表
    │── logs/
    │   └──[task_00000.log, task_00001.log, task_00002.log, ...] ⟵ 每个任务独立的日志文件
    │── completions/
    │   └──[00004, 00007, 00204, ...] ⟵ 标记任务已成功完成的空标记文件。在断点重跑/恢复作业时使用（自动仅重跑未完成的任务）
    │── stats/
    │   └──[00000.json, 00001.json, 00002.json, ...] ⟵ 各任务的独立统计数据（处理样本数、过滤数、移除数等）
    └── stats.json ⟵ 汇总所有任务的全局指标数据
```
</details>

### 控制台与日志着色（Colorization）

日志支持 ANSI 彩色高亮。默认情况下，控制台输出会自动检测并启用着色，而磁盘日志文件（`logs/task_XXXXX.log`）默认禁用颜色代码以保持纯文本整洁。
可通过以下环境变量显式覆盖默认着色行为：
- `DATATROVE_COLORIZE_LOGS`：设置为 `"1"` 强制在控制台输出中添加 ANSI 颜色，设置为 `"0"` 禁用
- `DATATROVE_COLORIZE_LOG_FILES`：设置为 `"1"` 强制将 ANSI 颜色代码写入磁盘日志文件

<span id="datafolder--paths"></span>
## DataFolder 与路径规范（DataFolder / paths）

得益于 [fsspec](https://filesystem-spec.readthedocs.io/en/latest/)，DataTrove 原生兼容丰富多元的输入与输出数据源。

在向 DataTrove 组件传递路径参数时（如 `input_folder`、`logging_dir`、`data_folder` 等），支持以下几种灵活形式：
- `str`：最直接的方式是传递字符串路径。示例：`/home/user/mydir`、`s3://mybucket/myinputdata`、`hf://buckets/myorg/my-bucket/raw/`、`hf://datasets/allenai/c4/en/`

> 对于原始数据与中间产物，建议使用 `hf://buckets/...`（类似 S3 的可变对象存储，无版本控制冗余开销）；对于最终清洗完毕、准备对外发布的成熟数据集，使用 `hf://datasets/...`。详见下方 [HF 存储桶章节](#hf-storage-buckets)。

- `(str, fsspec 文件系统实例)`：元组形式，包含字符串路径与已初始化的文件系统对象。示例：`("s3://mybucket/myinputdata", S3FileSystem(client_kwargs={"endpoint_url": endpoint_uri}))`
- `(str, dict)`：元组形式，包含字符串路径与用于初始化文件系统的配置字典。示例（等同于上一行）：`("s3://mybucket/myinputdata", {"client_kwargs": {"endpoint_url": endpoint_uri}})`
- `DataFolder`：直接实例化 [DataFolder](src/datatrove/io.py) 对象并作为参数传递

底层均由 [`get_datafolder`](src/datatrove/io.py#116) 函数自动完成统一解析与适配。

<span id="practical-guides"></span>
## 实践指南（Practical guides）

<span id="reading-data"></span>
### 数据读取（Reading data）

通常情况下，流水线的第一步为 [Reader 数据读取模块](src/datatrove/pipeline/readers)。
大部分读取模块均接收一个 `data_folder` 参数，指向待处理数据存放的文件夹路径。

系统会自动将该目录下的文件分发至各个任务中。若配置了 `N` 个任务，编号为 `i`（从 0 开始）的任务将负责处理索引为 `i, i+N, i+2N, i+3N,...` 的文件。

在内部，读取模块会先将数据反序列化为字典对象，随后封装为标准 `Document` 实例。

多数读取模块通用的常用参数：
- `text_key`：指定样本字典中作为正文文本的字段名（默认：`"text"`）
- `id_key`：指定作为样本唯一标识的字段名（默认：`"id"`）
- `default_metadata`：字典格式，用于为所有样本预先注入默认元数据（例如数据来源、批次编号等）
- `recursive`：是否递归扫描 `data_folder` 下的所有子文件夹
- `glob_pattern`：用于精确匹配指定文件的通配符模式。例如 `glob_pattern="*/warc/*.warc.gz"` 可精准匹配各子目录下 `warc/` 文件夹内的 `.warc.gz` 格式文件
- `adapter`：自定义适配器函数，接收读取器解析出的原始字典，并返回包含 `Document` 字段标准的字典。开发者可根据需要重写该函数（默认实现见 [_default_adapter](src/datatrove/pipeline/readers/base.py)）
- `limit`：仅读取指定数量的样本。在本地测试与调试流水线时非常实用

<span id="synthetic-data-generation"></span>
### 合成数据生成（Synthetic data generation）

通过 `uv sync --extra inference` 安装推理扩展，引入轻量级异步 HTTP 客户端、断点检查点支持以及基于 SQLite 的异步请求缓存。

通过 [InferenceRunner 模块](src/datatrove/pipeline/inference/run_inference.py)，DataTrove 原生支持 [vLLM](https://github.com/vllm-project/vllm)、[SGLang](https://github.com/sgl-project/sglang)、OpenAI 兼容的 HTTPS 端点以及本地 `dummy` 模拟测试服务。每个 DataTrove 任务既可以拉起专属于当前进程的服务副本（适用于 `vllm`、`sglang` 或 `dummy`），也可以直接连接外部集中部署的推理端点，同时通过异步并发批处理保持 GPU 利用率处于峰值。

<span id="custom-rollouts"></span>
#### 自定义 Rollout 执行函数

核心抽象是 **Rollout 函数**——一个标准的 Python 异步可调用对象（async callable），接收一个 `Document`、一个 `generate(payload)` 异步回调以及来自 `shared_context` 的任意额外上下文参数。你可以在 Rollout 内部自由编排多个串行或并行的 `generate` 请求，从而完全掌控提示词的动态构建方式与多轮生成结果的聚合逻辑。示例可参考 [inference_chunked.py](examples/inference/inference_chunked.py)：
- 基础的单次请求 Rollout
- 将超长文档自动切片并缝合多次生成结果的 Chunked 分块 Rollout
- 借助 `shared_context` 使用进程池处理 CPU 密集型预处理任务
- 多节点分布式协同推理

配置 `rollouts_per_document` 参数可自动对每个样本重复运行多次相同的 Rollout；生成器会将成功的输出集合统一归集在 `document.metadata["rollout_results"]` 中。

<span id="ready-to-use-generation-script"></span>
#### 开箱即用的大规模生成脚本

若需要开箱即用的大规模合成数据生产脚本（支持 1B 到 1T 参数量级模型、本地与 Slurm 弹性调度以及跨节点集群部署），请参阅 [`generate_data.py`](examples/inference/generate_data.py)。该脚本支持基于提示词的批量生成，并提供灵活的 System Prompt 与模板配置。

> 对于模型输出的原始生成数据，建议先写入 `hf://buckets/<org>/<bucket>/...`（或使用 `HuggingFaceBucketWriter`）；仅在完成后续数据过滤、清洗并确认发布时，再提升归档至 `hf://datasets/...`。详见 [HF 存储桶说明](#hf-storage-buckets)。

<span id="advanced-configuration"></span>
#### 进阶配置与断点恢复

`shared_context` 允许你在每个任务启动时注入全局共享状态，支持传入：
- 字典（直接作为关键字参数解包传入）
- 返回字典的可调用对象（便于惰性初始化耗时资源）
- 上下文管理器（Context Manager）或返回上下文管理器的函数（适用于管理进程池、GPU 分配器、临时目录等）。上下文管理器将在每个任务生命周期内准确执行一次 enter 和 exit

高可靠性断点续传（Recoverable generation）：
- 同时配置 `checkpoints_local_dir` 与 `records_per_chunk`，流水线会将每个 `Document` 实时写入本地分块文件（请在输出文件名模板中包含 `${chunk_index}`），并通过指定的写入器上传。若某个任务中途意外失败，重新拉起时将**自动从最后一个已完成的分块无缝恢复**。
- 启用检查点时，内置的 SQLite `RequestCache` 会通过请求载荷的哈希值自动对每次 Rollout 进行去重（依赖 `xxhash` 与 `aiosqlite`），因此在任务重试期间绝不会对同一请求重复调用推理后端。
- 在 `InferenceRunner` 上设置 `skip_bad_requests=True`，可在遇到服务端返回的 `BadRequestError`（例如上下文窗口溢出）时优雅跳过该错误样本，确保整体流水线持续稳定推进。

通过 `max_concurrent_generations` 调节并发请求批大小；在预处理或后处理较为繁重时，适当调大 `max_concurrent_documents`，允许更多 Rollout 协程在网络请求排队期间提前构建 Prompt 载荷。

<details>
  <summary>查看端到端最小示例代码</summary>

```python
from datatrove.data import Document
from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.inference.run_inference import InferenceConfig, InferenceRunner
from datatrove.pipeline.writers import JsonlWriter

async def simple_rollout(doc: Document, generate):
    payload = {"messages": [{"role": "user", "content": [{"type": "text", "text": doc.text}]}], "max_tokens": 2048}
    return await generate(payload)

documents = [Document(text="What's the weather in Tokyo?", id=str(i)) for i in range(1005)]
config = InferenceConfig(server_type="vllm", model_name_or_path="google/gemma-3-27b-it", rollouts_per_document=1, max_concurrent_generations=500)

LocalPipelineExecutor(
    pipeline=[
        documents,
        InferenceRunner(
            rollout_fn=simple_rollout,
            config=config,
            skip_bad_requests=True,
            records_per_chunk=500,
            checkpoints_local_dir="/fsx/.../translate-checkpoints",
            output_writer=JsonlWriter("s3://.../final_output_data", output_filename="${rank}_chunk_${chunk_index}.jsonl"),
        ),
    ],
    logging_dir="/fsx/.../inference_logs",
    tasks=1,
    ).run()
```
</details>

完整的进阶示例（单/多 Rollout 流程、带检查点恢复及跨 Rollout 共享进程池）可参阅 [inference_chunked.py](examples/inference/inference_chunked.py)。

<span id="progress-monitoring"></span>
#### 进度监控与数据集卡片

对于耗时较长的大规模离线推理作业，可使用 `InferenceProgressMonitor` 定期自动刷新 Hugging Face 数据集卡片中的进度条与预计完成时间（ETA）。推理全量完成后，`InferenceDatasetCardGenerator` 会自动生成包含详尽统计指标的最终版数据集卡片。

```python
from datatrove.pipeline.inference import InferenceDatasetCardParams, InferenceProgressMonitor, InferenceDatasetCardGenerator

params = InferenceDatasetCardParams(
    output_repo_id="your-username/output-dataset",
    input_dataset_name="simplescaling/s1K-1.1",
    input_dataset_split="train",
    model_name="Qwen/Qwen3-0.6B",
    # ... 其他可选参数
)

# 进度监控流水线（在 Slurm 上与推理作业并行运行）
monitor_pipeline = [InferenceProgressMonitor(params=params, update_interval=3600)]

# 最终卡片生成流水线（在推理全部完成后执行）
datacard_pipeline = [InferenceDatasetCardGenerator(params=params)]
```

完整并集成 Slurm 的可运行示例请参阅 [progress_monitoring.py](examples/inference/progress_monitoring.py)。

<span id="benchmarking"></span>
#### 性能基准测试（Benchmarking）

若需评估不同模型和架构配置（张量并行 TP、流水线并行 PP、投机解码 Speculative Decoding）下的 vLLM 实际吞吐，可使用官方提供的[基准测试工具集](examples/inference/benchmark/README.md)：
- **`launch_experiments.py`**：基于 YAML 配置文件批量发起超参数扫描实验，支持自动向 Slurm 提交调度作业
- **`analyze_results.py`**：自动解析服务日志并生成 CSV 汇总报表，直观呈现每张 GPU 的 tokens/s 吞吐及处理 1B Tokens 所需的 GPU·天（GPU-days）

<span id="extracting-text"></span>
### 文本提取（Extracting text）

可以使用 [extractors 提取组件](src/datatrove/pipeline/extractors) 从原始网页 HTML 中提取高质量正文文本。DataTrove 中最常用的提取器是 [Trafilatura](src/datatrove/pipeline/extractors/trafilatura.py)，底层基于成熟的 [trafilatura](https://trafilatura.readthedocs.io/en/latest/) 库构建。

<span id="filtering-data"></span>
### 数据过滤（Filtering data）

[Filters 过滤组件](src/datatrove/pipeline/filters) 是整个数据清洗流水线中最为关键的环节。DataTrove 的过滤组件接收单个 `Document` 并返回布尔值（`True` 表示保留该样本，`False` 表示剔除）。被淘汰的样本将终止流转，不会进入后续处理阶段。此外，你还可以将一个 [Writer 写入器](src/datatrove/pipeline/writers) 传递给 `exclusion_writer` 参数，将所有被淘汰的废弃样本另存至指定路径，便于后续分析过滤原因。

<span id="saving-data"></span>
### 数据保存（Saving data）

在数据处理和清洗完成后，通常需要将最终成果持久化保存。为此可以使用对应的 [writer 写入器](src/datatrove/pipeline/writers/jsonl.py)。
写入模块需要指定 `output_folder`（保存数据的目标路径）。你可以自由选择 `compression` 压缩算法（默认：`"gzip"`）以及文件的命名模板规则。
对于 `output_filename`，支持在模板中包含以下动态占位变量：
- `${rank}`：自动替换为当前任务的编号。**注意：若文件名模板中未包含此变量，不同并发任务可能会因写入同名文件而产生冲突覆盖！**
- `${id}`：替换为当前样本的 ID
- 自定义元数据：模板中的任意 `${tag}` 都会被自动替换为对应的 `document.metadata['tag']` 元数据取值

以下为根据样本元数据中的 `lang` 语种字段进行按语言自动分流存储的示例：
```python
JsonlWriter(
    f"{MAIN_OUTPUT_PATH}/non_english/",
    output_filename="${language}/" + DUMP + "/${rank}.jsonl.gz",  # 产出目录结构：language/dump/file
)
```

#### Hugging Face Hub 上的数据存储规范

当把数据持久化到 Hugging Face Hub 时，推荐**将原始/中间数据写入 Storage Buckets**，而**面向社区公开分享的最终发布版才写入 Datasets**。两者均提供原生写入器支持：

```python
from datatrove.pipeline.writers import HuggingFaceBucketWriter, HuggingFaceDatasetWriter

# 推荐：处理大规模流水线时用于存放原始及中间数据：
HuggingFaceBucketWriter(
    bucket="myorg/my-bucket",
    prefix="v1/raw",                # 存储桶内的相对路径前缀
    private=True,
    overwrite=True,                  # 写入前先清理该前缀下的已有文件（默认：False = 追加）
)

# 当把最终成果集成为正式发布的 HF 数据集时使用：
HuggingFaceDatasetWriter(
    dataset="myorg/my-dataset",
    private=True,
)
```

关于直连 fsspec 读写、`hf-mount` 本地挂载与 HF Jobs 卷挂载的完整对比，请参阅下文的 [HF 存储桶章节](#hf-storage-buckets)。

<span id="hf-storage-buckets"></span>
### HF Storage Buckets 存储桶

Hugging Face [Storage Buckets 存储桶](https://huggingface.co/docs/hub/storage-buckets)是底层基于 Xet 构建、类似 S3 的可变对象存储服务。这是存放原始数据及中间处理产物的首选推荐介质；而最终清洗完毕且准备发布的成熟版本，再提升推送到正式数据集（`hf://datasets/...`）。

你可通过以下四种方式读写存储桶，可根据自身基础设施按需选用：

| 接入方式 | 适用场景 |
| :--- | :--- |
| `HuggingFaceBucketWriter` | 大规模数据集、分阶段 Xet 快速上传、自动创建存储桶。支持 `overwrite=True` 覆盖替换已有文件 |
| 在任意读取器/写入器中使用原生 fsspec 路径（`hf://buckets/...`） | 通过 `HfFileSystem` 简单透明地读写，无需额外的前置安装配置 |
| [`hf-mount`](https://huggingface.co/docs/hub/storage-buckets-mount) (FUSE/NFS) | 零代码改动即可获得极佳读取吞吐；将云端存储桶直接挂载为本地常规目录访问 |
| [HF Jobs 卷挂载（Volume Mounts）](https://huggingface.co/docs/huggingface_hub/main/guides/jobs#volume-mounts) | 在 Hugging Face 官方云端设施上运行任务时的零配置方式；存储桶会自动挂载到指定容器路径 |

数据读取可直接使用现有的 `ParquetReader` 或 `JsonlReader`——无需单独的专用存储桶读取器，因为存储桶本质上即是标准的对象存储。有关这四种接入方式的直观对比示例，请参阅 [`examples/bucket_synthetic_data.py`](examples/bucket_synthetic_data.py)。

```python
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.writers import HuggingFaceBucketWriter

# 读取数据：在任意支持的读取组件中直接传入 hf://buckets/... 路径即可：
reader = ParquetReader(data_folder="hf://buckets/myorg/my-bucket/raw/")

# 写出数据：HuggingFaceBucketWriter 会在本地分片暂存后通过 Xet 高效推送到云端：
# 设置 overwrite=True 会在推送前自动清空当前前缀下的历史文件（默认：append 追加）
writer = HuggingFaceBucketWriter(
    bucket="myorg/my-bucket",
    prefix="v1/filtered",
    private=True,
    cleanup=True,
    overwrite=True,
)
```

存储桶 URL 同样可以直接用作任意执行器的 `logging_dir` 日志路径，例如 `logging_dir="hf://buckets/myorg/my-bucket/logs/v1"`。

<span id="deduplicating-data"></span>
### 数据去重（Deduplicating data）

有关去重技术的具体应用与完整流水线，请参考以下官方范例：
- [minhash_deduplication.py](examples/minhash_deduplication.py)：海量数据 MinHash 局部敏感哈希近似去重
- [sentence_deduplication.py](examples/sentence_deduplication.py)：基于句子的精确去重
- [exact_substrings.py](examples/exact_substrings.py)：超长公共子串精确去重

<span id="summary-statistics"></span>
### 数据汇总统计（Summary Statistics）

若需对语料质量进行全方位透视并提取画像，可使用 [Stats 统计组件](https://github.com/huggingface/datatrove/tree/main/src/datatrove/pipeline/stats)。这些组件提供了一种分布式、高并发收集数据集特征画像的标准方案。整体执行分为两步：
1. 针对各个数据分片遍历文档，并按照以下分组模式之一聚合并收集统计量：`summary`（所有文档统一计入 summary 键）、`fqdn`（按完全限定域名分组，如网站域名）、`suffix`（按 URL 路径后缀分组）或 `histogram`（基于数值区间的直方图分组）。
2. 将各分片产出的统计结果合并归纳为单个全局汇总文件。

详细代码演示请参阅 [summary_stats.py](examples/summary_stats.py)。

每个统计结果均存储在遵循特定目录结构的独立文件中：`output_folder/{fqdn,suffix,summary,histogram}/{stat_name}/metric.json`

上述生成的 JSON 文件是一个标准的 `MetricStatsDict` 对象，可使用如下代码轻松反序列化加载并分析：

```python
from datatrove.pipeline.stats.summary_stats import MetricStatsDict
import json

stats = MetricStatsDict.from_dict(json.load(open("fqdn/length/metric.json")))

# 示例：获取 nytimes.com 域名下所有文档的总长度
stats["nytimes.com"].total

# 示例：获取 cnn.com 域名下文档长度的平均值
stats["cnn.com"].mean
```

内置支持的统计模块包括：
- `contamination_stats.py`：`word_contamination_{words[0]}`：统计指定单词在文档中的污染出现频次。
- `doc_stats.py`：`length`（文档总字符长度）、`white_space_ratio`（空白字符占比）、`non_alpha_digit_ratio`（非字母数字字符占比）、`digit_ratio`（数字字符占比）、`uppercase_ratio`（大写字母占比）、`elipsis_ratio`（省略号占比）、`punctuation_ratio`（标点符号占比）。
- `lang_stats.py`：`fasttext_{language}`：利用 FastText 模型评估文档属于特定 `language` 语言的置信度评分。
- `line_stats.py`：`n_lines`（每篇文档行数）、`avg_line_length`（平均行长度）、`long_line_ratio_chars_{chars}`（长度大于特定字符数的长行比例）、`short_line_ratio_chars_{chars}`（短行比例）、`bullet_point_lines_ratio`（以项目符号开头的行占比）、`line_duplicates`（重复行比例）、`line_char_duplicates`（重复行中所包含字符占全部字符的比例）。
- `paragraph_stats.py`：`n_paragraphs`（段落数量）、`avg_paragraph_length`（平均段落长度）、`short_paragraph_ratio_{chars}`（短段落比例，`<{chars}` 字符）、`long_paragraph_ratio_{chars}`（长段落比例，`>{chars}` 字符）。
- `perplexity_stats.py`：`ccnet_perplexity_{model_dataset}_{language}`：利用 CCNet 模型评估在特定语言和数据集上的困惑度（Perplexity）。
- `sentence_stats.py`：`n_sentences`（句子数量）、`avg_sentence_length`（平均句子长度）、`short_sentence_ratio_{chars}`（短句比例）、`long_sentence_ratio_{chars}`（长句比例）。
- `token_stats.py`：`token_count`：文档包含的 Token 词元总量。
- `word_stats.py`：`n_words`（单词总数）、`avg_word_length`（平均词长）、`avg_words_per_line`（每行平均词数）、`short_word_ratio_{chars}`（短词比例）、`stop_word_ratio`（停用词占比）、`long_word_ratio_{chars}`（长词比例）、`type_token_ratio`（词汇多样性：不重复词数 / Token 总数）、`capitalized_word_ratio`（首字母大写词比例）、`uppercase_word_ratio`（全大写词比例）。

<span id="custom-blocks"></span>
### 自定义处理模块（Custom blocks）

<span id="simple-data"></span>
#### 直接传入数据集合

你可以将一个包含 [`Document`](src/datatrove/data.py) 的可迭代对象直接作为流水线的第一步：

```python
from datatrove.data import Document
from datatrove.pipeline.filters import SamplerFilter
from datatrove.pipeline.writers import JsonlWriter

pipeline = [
    [
        Document(text="some data", id="0"),
        Document(text="some more data", id="1"),
        Document(text="even more data", id="2"),
    ],
    SamplerFilter(rate=0.5),
    JsonlWriter(
        output_folder="/my/output/path"
    )
]
```

请注意，该列表数据不会被自动分片（若启动了多个并发任务，每个任务都会完整收到全部样本）。该用法非常适合小规模数据处理或功能验证。

<span id="custom-function"></span>
#### 自定义函数处理

对于轻量级的数据变换逻辑，只需传入一个满足如下签名的自定义生成器函数即可：

```python
from datatrove.data import DocumentsPipeline

def uppercase_everything(data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
    """
        `data` 是一个 Document 生成器。你同样必须使用 yield 逐个返回 Document 生成器。
        可按需使用 `rank` 和 `world_size` 实现自定义的分片与路由逻辑。
    """
    for document in data:
        document.text = document.text.upper()
        yield document

pipeline = [
    ...,
    uppercase_everything,
    ...
]
```

> [!TIP]
> 在多进程并发环境下，有时可能会因外部模块引用而产生 Pickle 序列化问题。若遇到此类情况，只需将相关的 `import` 语句移动至该函数体内部即可解决。

<span id="custom-block"></span>
#### 完整自定义模块开发

若需要实现完备且带状态的处理组件，可继承基类 [`PipelineStep`](src/datatrove/pipeline/base.py) 或其派生子类：

```python
from datatrove.pipeline.base import PipelineStep
from datatrove.data import DocumentsPipeline
from datatrove.io import DataFolderLike, get_datafolder


class UppercaserBlock(PipelineStep):
    def __init__(self, some_folder: DataFolderLike, some_param: int = 5):
        super().__init__()
        # 接收任意参数并保存为实例属性
        self.some_param = some_param
        # 使用 get_datafolder() 统一加载与适配路径
        self.some_folder = get_datafolder(some_folder)

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        # 场景 1：从自定义存储路径 `some_folder` 中读取并生成数据：
        for filepath in self.some_folder.get_shard(rank, world_size):  # 支持传入 glob 通配符模式等
            with self.some_folder.open(filepath, "rt") as f:
                # 处理文件内容
                ...
                yield doc

        #
        # 场景 2：流式处理来自前置步骤的数据（`data`）
        #

        for doc in data:
            with self.track_time():
                # 可以用 track_time() 上下文包裹核心处理逻辑，自动度量各文档的处理耗时
                nr_uppercase_letters = sum(map(lambda c: c.isupper(), doc.text))
                # 还可以使用 stat_update 记录文档级指标
                self.stat_update("og_upper_letters", value=nr_uppercase_letters)
                doc.text = doc.text.upper()
            # 注意：请务必确保 yield 位于 track_time 代码块外部，避免把下游等待时间计入当前步骤耗时
            yield doc

        #
        # 场景 3：直接将数据持久化保存到磁盘
        #

        with self.some_folder.open("myoutput", "wt") as f:
            for doc in data:
                f.write(doc...)
```

在流水线中使用该自定义组件：
```python
pipeline = [
    ...,
    UppercaserBlock("somepath"),
    ...
]
```

你也可以根据功能定位，直接继承专用基类：[`BaseExtractor`](src/datatrove/pipeline/extractors/base.py)（文本提取器）、[`BaseFilter`](src/datatrove/pipeline/filters/base_filter.py)（质量过滤器）、[`BaseReader`/`BaseDiskReader`](src/datatrove/pipeline/readers/base.py)（数据读取器）或 [`DiskWriter`](src/datatrove/pipeline/writers/disk_base.py)（磁盘写出器）。

<span id="contributing"></span>
## 参与贡献（Contributing）

本地开发环境搭建：
```bash
git clone git@github.com:huggingface/datatrove.git && cd datatrove
uv sync --extra dev
```

安装 pre-commit 代码规范钩子：
```bash
pre-commit install
```

执行代码规范与格式化检查：
```bash
# 快速本地检查循环（仅检查发生变动的 Python 文件）
make quality
make style

# 全量仓库检查（与 GitHub CI 范围一致）
make quality-full
make style-full
```

运行单元测试：
```bash
pytest -sv ./tests/
```

<span id="citation"></span>
## 引用（Citation）

如果在研究中使用了 DataTrove，欢迎引用如下 BibTeX 条目：

```bibtex
@misc{penedo2024datatrove,
  author = {Penedo, Guilherme and Kydlíček, Hynek and Cappelli, Alessandro and Sasko, Mario and Wolf, Thomas},
  title = {DataTrove: large scale data processing},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/huggingface/datatrove}
}
```

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年10月23日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
