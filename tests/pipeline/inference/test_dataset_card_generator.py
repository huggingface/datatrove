"""Tests for dataset card generation and config/prompt patching."""

import json
from pathlib import Path

import pytest

from datatrove.pipeline.inference import dataset_card_generator as dcg
from datatrove.pipeline.inference.dataset_card_generator import (
    InferenceDatasetCardGenerator,
    InferenceDatasetCardParams,
    JobStats,
    _add_all_config,
    _build_config_entry,
    _decode_prompt_html_content,
    _extract_prompt_templates,
    _fetch_existing_configs,
    _format_block,
    _merge_configs,
    _render_configs_block,
    _render_job_stats,
    _render_load_dataset_example,
    _render_prompt_pre,
    _render_user_prompt_info,
    _size_category,
    build_and_upload_dataset_card,
    format_number,
    load_job_stats,
    patch_readme_configs,
    patch_readme_prompt,
)


def _make_card_params(
    *,
    stats_path: str,
    prompt_template: str | None = "Document: [[DOCUMENT]]",
    prompt_template_name: str = "default",
) -> InferenceDatasetCardParams:
    return InferenceDatasetCardParams(
        output_repo_id="org/repo",
        input_dataset_name="org/input",
        input_dataset_split="train",
        input_dataset_config=None,
        prompt_column="text",
        prompt_template=prompt_template,
        prompt_template_name=prompt_template_name,
        system_prompt="You are a helpful assistant.",
        model_name="org/model",
        model_revision="main",
        generation_kwargs={
            "model_max_context": 4096,
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": 50,
            "max_tokens": 256,
        },
        spec_config=None,
        stats_path=stats_path,
    )


class TestBuildConfigEntry:
    def test_basic(self):
        entry = _build_config_entry("math", "train", "math/**/*.parquet")
        assert entry == {
            "config_name": "math",
            "data_files": [{"split": "train", "path": "math/**/*.parquet"}],
        }

    def test_default_config(self):
        entry = _build_config_entry("default", "train", "default/**/*.parquet")
        assert entry["config_name"] == "default"
        assert entry["data_files"][0]["path"] == "default/**/*.parquet"


class TestMergeConfigs:
    def test_add_new_config_to_empty(self):
        new = _build_config_entry("math", "train", "math/**/*.parquet")
        result = _merge_configs([], new)
        assert len(result) == 1
        assert result[0]["config_name"] == "math"

    def test_add_second_config(self):
        existing = [_build_config_entry("math", "train", "math/**/*.parquet")]
        new = _build_config_entry("faq", "train", "faq/**/*.parquet")
        result = _merge_configs(existing, new)
        assert len(result) == 2
        assert result[0]["config_name"] == "faq"
        assert result[1]["config_name"] == "math"

    def test_replace_existing_config(self):
        existing = [
            _build_config_entry("math", "train", "math/old/*.parquet"),
            _build_config_entry("faq", "train", "faq/**/*.parquet"),
        ]
        new = _build_config_entry("math", "train", "math/**/*.parquet")
        result = _merge_configs(existing, new)
        assert len(result) == 2
        math_cfg = next(c for c in result if c["config_name"] == "math")
        assert math_cfg["data_files"][0]["path"] == "math/**/*.parquet"

    def test_sorting_with_many_configs(self):
        configs = []
        for name in ["tutorial", "math", "faq", "table"]:
            new = _build_config_entry(name, "train", f"{name}/**/*.parquet")
            configs = _merge_configs(configs, new)
        names = [c["config_name"] for c in configs]
        assert names == ["faq", "math", "table", "tutorial"]


class TestAddAllConfig:
    def test_skips_when_fewer_than_two_named(self):
        configs = [_build_config_entry("math", "train", "math/**/*.parquet")]
        result = _add_all_config(configs, "train")
        assert len(result) == 1
        assert result[0]["config_name"] == "math"

    def test_adds_all_with_two_named(self):
        configs = [
            _build_config_entry("faq", "train", "faq/**/*.parquet"),
            _build_config_entry("math", "train", "math/**/*.parquet"),
        ]
        result = _add_all_config(configs, "train")
        assert len(result) == 3
        all_cfg = result[0]
        assert all_cfg["config_name"] == "all"
        # Single data_files entry with a list of paths (no duplicate splits)
        assert len(all_cfg["data_files"]) == 1
        assert all_cfg["data_files"][0]["split"] == "train"
        assert isinstance(all_cfg["data_files"][0]["path"], list)
        assert set(all_cfg["data_files"][0]["path"]) == {"faq/**/*.parquet", "math/**/*.parquet"}

    def test_all_config_has_no_duplicate_splits(self):
        """Regression test: HuggingFace rejects data_files with duplicate split names."""
        configs = [
            _build_config_entry("faq", "train", "faq/**/*.parquet"),
            _build_config_entry("math", "train", "math/**/*.parquet"),
            _build_config_entry("table", "train", "table/**/*.parquet"),
            _build_config_entry("tutorial", "train", "tutorial/**/*.parquet"),
        ]
        result = _add_all_config(configs, "train")
        all_cfg = next(c for c in result if c["config_name"] == "all")
        splits = [df["split"] for df in all_cfg["data_files"]]
        assert len(splits) == 1, f"Expected 1 data_files entry, got {len(splits)} with splits: {splits}"

    def test_ignores_default_and_all_configs(self):
        configs = [
            _build_config_entry("default", "train", "default/**/*.parquet"),
            _build_config_entry("faq", "train", "faq/**/*.parquet"),
        ]
        result = _add_all_config(configs, "train")
        # Only 1 named config (faq), so no "all" config should be added
        assert not any(c["config_name"] == "all" for c in result)


class TestRenderConfigsBlock:
    def test_single_default_config(self):
        configs = [_build_config_entry("default", "train", "default/**/*.parquet")]
        block = _render_configs_block(configs, "train")
        assert "config_name: default" in block
        assert "path: default/**/*.parquet" in block
        assert "split: train" in block
        assert "train-eval-index:" in block

    def test_multiple_named_configs(self):
        configs = [
            _build_config_entry("faq", "train", "faq/**/*.parquet"),
            _build_config_entry("math", "train", "math/**/*.parquet"),
        ]
        block = _render_configs_block(configs, "train")
        assert "config_name: faq" in block
        assert "config_name: math" in block
        assert "config: faq" in block

    def test_renders_list_paths_for_all_config(self):
        """Verify that the 'all' config with list paths renders valid YAML."""
        configs = [
            _build_config_entry("faq", "train", "faq/**/*.parquet"),
            _build_config_entry("math", "train", "math/**/*.parquet"),
        ]
        configs = _add_all_config(configs, "train")
        block = _render_configs_block(configs, "train")
        assert "config_name: all" in block
        assert "    path:" in block
        assert "    - faq/**/*.parquet" in block
        assert "    - math/**/*.parquet" in block


class TestRenderLoadDatasetExample:
    def test_default_config(self):
        configs = [_build_config_entry("default", "train", "default/**/*.parquet")]
        example = _render_load_dataset_example("org/repo", configs)
        assert 'load_dataset("org/repo")' in example
        assert '"default"' not in example

    def test_named_configs(self):
        configs = [
            _build_config_entry("faq", "train", "faq/**/*.parquet"),
            _build_config_entry("math", "train", "math/**/*.parquet"),
        ]
        example = _render_load_dataset_example("org/repo", configs)
        assert 'load_dataset("org/repo", "faq")' in example
        assert 'load_dataset("org/repo", "math")' in example
        assert "ds_faq" in example
        assert "ds_math" in example


class TestFetchExistingConfigs:
    def test_nonexistent_repo_returns_empty(self):
        result = _fetch_existing_configs("nonexistent-org/nonexistent-repo-12345")
        assert result == []


class TestLoadJobStats:
    def test_parses_dict_stats(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        stats_path = tmp_path / "stats.json"
        stats_path.write_text(
            json.dumps(
                [
                    {"stats": {"doc_len": {"n": 10, "mean": 42.5}, "documents": {"total": 12}}},
                    {
                        "stats": {
                            "prompt_tokens": {"total": 100, "mean": 10.0},
                            "completion_tokens": {"total": 80, "mean": 8.0},
                        }
                    },
                ]
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(dcg.time, "sleep", lambda _seconds: None)

        stats = load_job_stats(stats_path, timeout=1)

        assert stats is not None
        assert stats.document_count == 12
        assert stats.mean_doc_len == 42.5
        assert stats.prompt_tokens_total == 100
        assert stats.completion_tokens_total == 80
        assert stats.prompt_tokens_mean == 10.0
        assert stats.completion_tokens_mean == 8.0

    def test_parses_scalar_doc_len(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        stats_path = tmp_path / "stats.json"
        stats_path.write_text(json.dumps([{"stats": {"doc_len": 7}}]), encoding="utf-8")
        monkeypatch.setattr(dcg.time, "sleep", lambda _seconds: None)

        stats = load_job_stats(stats_path, timeout=1)

        assert stats is not None
        assert stats.document_count == 7
        assert stats.mean_doc_len is None
        assert stats.prompt_tokens_total is None
        assert stats.completion_tokens_total is None

    def test_times_out_when_file_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        missing_path = tmp_path / "missing.json"
        now = {"value": 0.0}

        def fake_time() -> float:
            now["value"] += 1.0
            return now["value"]

        monkeypatch.setattr(dcg.time, "time", fake_time)
        monkeypatch.setattr(dcg.time, "sleep", lambda _seconds: None)

        assert load_job_stats(missing_path, timeout=0) is None

    def test_returns_none_when_no_doc_entry(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        stats_path = tmp_path / "stats.json"
        stats_path.write_text(json.dumps([{"stats": {"some_other_key": 42}}]), encoding="utf-8")
        monkeypatch.setattr(dcg.time, "sleep", lambda _seconds: None)

        assert load_job_stats(stats_path, timeout=1) is None

    def test_returns_none_on_json_parse_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        stats_path = tmp_path / "stats.json"
        stats_path.write_text("not valid json", encoding="utf-8")
        monkeypatch.setattr(dcg.time, "sleep", lambda _seconds: None)

        assert load_job_stats(stats_path, timeout=1) is None

    def test_handles_documents_as_dict_with_total(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        stats_path = tmp_path / "stats.json"
        stats_path.write_text(
            json.dumps([{"stats": {"doc_len": {"n": 5, "mean": 10.0}, "documents": {"total": 99}}}]),
            encoding="utf-8",
        )
        monkeypatch.setattr(dcg.time, "sleep", lambda _seconds: None)

        stats = load_job_stats(stats_path, timeout=1)
        assert stats is not None
        assert stats.document_count == 99

    def test_handles_documents_as_plain_int(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        stats_path = tmp_path / "stats.json"
        stats_path.write_text(
            json.dumps([{"stats": {"doc_len": {"n": 5, "mean": 10.0}, "documents": 42}}]),
            encoding="utf-8",
        )
        monkeypatch.setattr(dcg.time, "sleep", lambda _seconds: None)

        stats = load_job_stats(stats_path, timeout=1)
        assert stats is not None
        assert stats.document_count == 42

    def test_falls_back_to_doc_len_n_when_no_documents_key(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        stats_path = tmp_path / "stats.json"
        stats_path.write_text(
            json.dumps([{"stats": {"doc_len": {"n": 7, "mean": 10.0}}}]),
            encoding="utf-8",
        )
        monkeypatch.setattr(dcg.time, "sleep", lambda _seconds: None)

        stats = load_job_stats(stats_path, timeout=1)
        assert stats is not None
        assert stats.document_count == 7

    def test_waits_for_file_to_appear(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Test the wait-and-retry loop when file appears after one sleep cycle."""
        stats_path = tmp_path / "stats.json"

        def create_file_on_sleep(_seconds: float) -> None:
            if not stats_path.exists():
                stats_path.write_text(
                    json.dumps([{"stats": {"doc_len": {"n": 3, "mean": 5.0}}}]),
                    encoding="utf-8",
                )

        monkeypatch.setattr(dcg.time, "sleep", create_file_on_sleep)
        monkeypatch.setattr(dcg.time, "time", lambda: 0.0)

        stats = load_job_stats(stats_path, timeout=60)
        assert stats is not None
        assert stats.document_count == 3


class TestSizeCategory:
    def test_none(self):
        assert _size_category(None) == "unknown"

    def test_small(self):
        assert _size_category(500) == "n<1K"

    def test_1k_to_10k(self):
        assert _size_category(5_000) == "1K<n<10K"

    def test_10k_to_100k(self):
        assert _size_category(50_000) == "10K<n<100K"

    def test_100k_to_1m(self):
        assert _size_category(500_000) == "100K<n<1M"

    def test_over_1m(self):
        assert _size_category(5_000_000) == "n>1M"


class TestFormatNumber:
    def test_none_treated_as_zero(self):
        assert format_number(None) == "0"

    def test_small_number(self):
        assert format_number(42) == "42"

    def test_thousands(self):
        assert format_number(1_500) == "1,500"

    def test_millions(self):
        result = format_number(1_500_000)
        assert "1,500,000" in result
        assert "≈1.5M" in result

    def test_billions(self):
        result = format_number(2_500_000_000)
        assert "2,500,000,000" in result
        assert "≈2.5B" in result

    def test_trillions(self):
        result = format_number(1_200_000_000_000)
        assert "≈1.2T" in result


class TestRenderJobStats:
    def test_none_returns_fallback(self):
        assert _render_job_stats(None) == "Job statistics could not be collected."

    def test_full_stats(self):
        stats = JobStats(
            document_count=1000,
            mean_doc_len=128.5,
            prompt_tokens_total=10_000,
            completion_tokens_total=20_000,
            prompt_tokens_mean=10.0,
            completion_tokens_mean=20.0,
        )
        result = _render_job_stats(stats)
        assert "| Documents processed | 1,000 |" in result
        assert "| Avg. source chars | 128.50 |" in result
        assert "| Total prompt tokens | 10,000 |" in result
        assert "| Mean prompt tokens | 10.00 |" in result

    def test_minimal_stats_omits_optional_rows(self):
        stats = JobStats(
            document_count=100,
            mean_doc_len=None,
            prompt_tokens_total=None,
            completion_tokens_total=None,
            prompt_tokens_mean=None,
            completion_tokens_mean=None,
        )
        result = _render_job_stats(stats)
        assert "Documents processed" in result
        assert "n/a" in result
        assert "Avg. source chars" not in result
        assert "Mean prompt tokens" not in result


class TestFormatBlock:
    def test_empty_values_returns_fallback(self):
        assert _format_block([], "- fallback") == "- fallback"

    def test_filters_empty_strings(self):
        assert _format_block(["", "", ""], "- fallback") == "- fallback"

    def test_formats_values_as_list(self):
        result = _format_block(["a", "b"], "- fallback")
        assert result == "- a\n- b"


class TestRenderPromptPre:
    def test_escapes_html(self):
        result = _render_prompt_pre("<script>alert('xss')</script>")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_encodes_newlines_as_br(self):
        result = _render_prompt_pre("line1\nline2")
        assert "<br/>" in result
        assert "\n" not in result.split(">")[1].split("<")[0]

    def test_roundtrips_with_decode(self):
        original = "Line 1\nLine 2 with <html> & stuff"
        encoded = _render_prompt_pre(original)
        # Extract body between <pre ...> and </pre>
        body = encoded.split(">", 1)[1].rsplit("</pre>", 1)[0]
        decoded = _decode_prompt_html_content(body)
        assert decoded == original


class TestExtractPromptTemplates:
    def test_extracts_single_prompt(self):
        content = (
            '<details>\n<summary><b>math</b> prompt</summary>\n'
            '<pre style="white-space: pre-wrap;">Math prompt text</pre>\n'
            "</details>"
        )
        result = _extract_prompt_templates(content)
        assert "math" in result
        assert result["math"] == "Math prompt text"

    def test_extracts_multiple_prompts(self):
        content = (
            '<details>\n<summary><b>math</b> prompt</summary>\n'
            '<pre style="white-space: pre-wrap;">Math</pre>\n</details>\n'
            '<details>\n<summary><b>faq</b> prompt</summary>\n'
            '<pre style="white-space: pre-wrap;">FAQ</pre>\n</details>'
        )
        result = _extract_prompt_templates(content)
        assert len(result) == 2
        assert result["math"] == "Math"
        assert result["faq"] == "FAQ"

    def test_returns_empty_for_no_prompts(self):
        assert _extract_prompt_templates("# No prompts here") == {}

    def test_handles_div_tag(self):
        content = (
            '<details>\n<summary><b>math</b> prompt</summary>\n'
            '<div style="white-space: pre-wrap;">Math text</div>\n'
            "</details>"
        )
        result = _extract_prompt_templates(content)
        assert result["math"] == "Math text"


class TestRenderUserPromptInfo:
    def test_no_template_default_config(self):
        result = _render_user_prompt_info("default", None, "text", {})
        assert "Column `text`" in result
        assert "<details>" not in result

    def test_single_default_with_template(self):
        result = _render_user_prompt_info("default", "Prompt: [[DOCUMENT]]", "text", {})
        assert "Template with content from column `text`" in result
        assert "<details>" in result
        assert "Prompt template" in result

    def test_named_config_renders_details_block(self):
        result = _render_user_prompt_info("math", "Math prompt", "text", {})
        assert "<b>math</b> prompt" in result
        assert "from column `text`" in result

    def test_merges_with_existing_prompts(self):
        existing = {"faq": "FAQ prompt"}
        result = _render_user_prompt_info("math", "Math prompt", "text", existing)
        assert "<b>math</b>" in result
        assert "<b>faq</b>" in result


class TestRenderLoadDatasetExampleAllConfig:
    def test_all_config_uses_ds_variable(self):
        configs = _add_all_config(
            [
                _build_config_entry("faq", "train", "faq/**/*.parquet"),
                _build_config_entry("math", "train", "math/**/*.parquet"),
            ],
            "train",
        )
        example = _render_load_dataset_example("org/repo", configs)
        assert 'ds = load_dataset("org/repo", "all")  # all subsets combined' in example
        assert "ds_faq" in example
        assert "ds_math" in example


class TestBuildAndUploadDatasetCard:
    def test_progress_update_skips_loading_stats(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        params = _make_card_params(stats_path=str(tmp_path / "stats.json"))
        uploaded: dict[str, str] = {}

        def fail_load_job_stats(_stats_path: Path, timeout: int = 300) -> JobStats | None:
            raise AssertionError("load_job_stats should not be called for progress updates")

        def fake_upload_file(*, path_or_fileobj: str, path_in_repo: str, repo_id: str, repo_type: str) -> None:
            uploaded["path_in_repo"] = path_in_repo
            uploaded["repo_id"] = repo_id
            uploaded["repo_type"] = repo_type
            uploaded["content"] = Path(path_or_fileobj).read_text(encoding="utf-8")

        monkeypatch.setattr(dcg, "load_job_stats", fail_load_job_stats)
        monkeypatch.setattr(
            dcg,
            "fetch_source_dataset_metadata",
            lambda _dataset_name: {"license": "apache-2.0", "languages": ["en"], "tags": ["tag-a"]},
        )
        monkeypatch.setattr(dcg, "whoami", lambda: {"name": "tester"})
        monkeypatch.setattr(dcg, "_parse_existing_prompts", lambda _repo_id: {})
        monkeypatch.setattr(dcg, "_fetch_existing_configs", lambda _repo_id: [])
        monkeypatch.setattr(dcg, "upload_file", fake_upload_file)

        build_and_upload_dataset_card(params=params, progress_section="## Generation Progress\n\n")

        assert uploaded["path_in_repo"] == "README.md"
        assert uploaded["repo_id"] == "org/repo"
        assert uploaded["repo_type"] == "dataset"
        assert "## Generation Progress" in uploaded["content"]
        assert "path: default/**/*.parquet" in uploaded["content"]

    def test_with_input_config_and_spec(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        params = InferenceDatasetCardParams(
            output_repo_id="org/repo",
            input_dataset_name="org/input",
            input_dataset_split="train",
            input_dataset_config="sample-350BT",
            prompt_column="text",
            prompt_template="Prompt: [[DOCUMENT]]",
            prompt_template_name="math",
            system_prompt="Be helpful",
            model_name="org/model",
            model_revision="v2",
            generation_kwargs={
                "model_max_context": 8192,
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": 50,
                "max_tokens": 2048,
            },
            spec_config='{"method": "suffix"}',
            stats_path=str(tmp_path / "stats.json"),
        )
        uploaded: dict[str, str] = {}

        def fail_load_job_stats(_stats_path, timeout=300):
            raise AssertionError("should not be called for progress updates")

        monkeypatch.setattr(dcg, "load_job_stats", fail_load_job_stats)
        monkeypatch.setattr(
            dcg,
            "fetch_source_dataset_metadata",
            lambda _: {"license": "mit", "languages": "en", "tags": []},
        )
        monkeypatch.setattr(dcg, "whoami", lambda: {"name": "joel"})
        monkeypatch.setattr(dcg, "_parse_existing_prompts", lambda _: {})
        monkeypatch.setattr(dcg, "_fetch_existing_configs", lambda _: [])
        monkeypatch.setattr(
            dcg,
            "upload_file",
            lambda *, path_or_fileobj, path_in_repo, repo_id, repo_type: uploaded.update(
                {"content": Path(path_or_fileobj).read_text(encoding="utf-8")}
            ),
        )

        build_and_upload_dataset_card(params=params, progress_section="## Progress")

        content = uploaded["content"]
        assert "`sample-350BT` config" in content
        assert "joel" in content
        assert '{"method": "suffix"}' in content
        assert "Be helpful" in content

    def test_whoami_failure_uses_fallback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        params = _make_card_params(stats_path=str(tmp_path / "stats.json"))
        uploaded: dict[str, str] = {}

        monkeypatch.setattr(dcg, "load_job_stats", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(
            dcg,
            "fetch_source_dataset_metadata",
            lambda _: {"license": "apache-2.0", "languages": ["en"], "tags": []},
        )
        monkeypatch.setattr(dcg, "whoami", lambda: (_ for _ in ()).throw(RuntimeError("no token")))
        monkeypatch.setattr(dcg, "_parse_existing_prompts", lambda _: {})
        monkeypatch.setattr(dcg, "_fetch_existing_configs", lambda _: [])
        monkeypatch.setattr(
            dcg,
            "upload_file",
            lambda *, path_or_fileobj, path_in_repo, repo_id, repo_type: uploaded.update(
                {"content": Path(path_or_fileobj).read_text(encoding="utf-8")}
            ),
        )

        build_and_upload_dataset_card(params=params, progress_section="## Progress")

        assert "hf_user" in uploaded["content"]

    def test_final_update_uses_loaded_stats(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        params = _make_card_params(stats_path=str(tmp_path / "stats.json"))
        uploaded: dict[str, str] = {}

        monkeypatch.setattr(
            dcg,
            "load_job_stats",
            lambda _stats_path: JobStats(
                document_count=1000,
                mean_doc_len=128.0,
                prompt_tokens_total=10_000,
                completion_tokens_total=20_000,
                prompt_tokens_mean=10.0,
                completion_tokens_mean=20.0,
            ),
        )
        monkeypatch.setattr(
            dcg,
            "fetch_source_dataset_metadata",
            lambda _dataset_name: {"license": "apache-2.0", "languages": ["en"], "tags": ["tag-a"]},
        )
        monkeypatch.setattr(dcg, "whoami", lambda: {"name": "tester"})
        monkeypatch.setattr(dcg, "_parse_existing_prompts", lambda _repo_id: {})
        monkeypatch.setattr(dcg, "_fetch_existing_configs", lambda _repo_id: [])
        monkeypatch.setattr(
            dcg,
            "upload_file",
            lambda *, path_or_fileobj, path_in_repo, repo_id, repo_type: uploaded.update(
                {
                    "path_in_repo": path_in_repo,
                    "repo_id": repo_id,
                    "repo_type": repo_type,
                    "content": Path(path_or_fileobj).read_text(encoding="utf-8"),
                }
            ),
        )

        build_and_upload_dataset_card(params=params)

        assert uploaded["path_in_repo"] == "README.md"
        assert uploaded["repo_id"] == "org/repo"
        assert uploaded["repo_type"] == "dataset"
        assert "The run produced 1,000 samples and generated 20,000 tokens." in uploaded["content"]
        assert "| Documents processed | 1,000 |" in uploaded["content"]


class TestInferenceDatasetCardGeneratorStep:
    def test_rank_zero_runs_card_generation(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        params = _make_card_params(stats_path=str(tmp_path / "stats.json"))
        step = InferenceDatasetCardGenerator(params=params)
        calls: list[dict[str, object]] = []

        monkeypatch.setattr(
            dcg,
            "build_and_upload_dataset_card",
            lambda *, params, progress_section: calls.append({"params": params, "progress_section": progress_section}),
        )

        passthrough = list(step.run(data=["a", "b"], rank=0))

        assert passthrough == ["a", "b"]
        assert len(calls) == 1
        assert calls[0]["params"] == params
        assert calls[0]["progress_section"] == ""

    def test_nonzero_rank_skips_card_generation(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        params = _make_card_params(stats_path=str(tmp_path / "stats.json"))
        step = InferenceDatasetCardGenerator(params=params)
        calls: list[dict[str, object]] = []

        monkeypatch.setattr(
            dcg,
            "build_and_upload_dataset_card",
            lambda *, params, progress_section: calls.append({"params": params, "progress_section": progress_section}),
        )

        passthrough = list(step.run(data=["x"], rank=1))

        assert passthrough == ["x"]
        assert calls == []

    def test_errors_are_caught(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        params = _make_card_params(stats_path=str(tmp_path / "stats.json"))
        step = InferenceDatasetCardGenerator(params=params)

        def raise_error(*, params: InferenceDatasetCardParams, progress_section: str) -> None:
            raise RuntimeError("boom")

        monkeypatch.setattr(dcg, "build_and_upload_dataset_card", raise_error)

        assert list(step.run(rank=0)) == []


class TestPatchReadmePrompt:
    SAMPLE_README = (
        " * User prompts (from column `text`):\n"
        "\n"
        "   <details>\n"
        "   <summary><b>table</b> prompt</summary>\n"
        "\n"
        '   <pre style="white-space: pre-wrap;">Table prompt here</pre>\n'
        "\n"
        "   </details>\n"
        "\n"
        "## 🔄 Generation Progress\n"
    )

    def test_inserts_missing_prompt(self):
        result = patch_readme_prompt(self.SAMPLE_README, "math", "Math prompt here")
        assert "<b>math</b> prompt" in result
        assert "Math prompt here" in result
        assert "<b>table</b> prompt" in result

    def test_updates_existing_prompt(self):
        result = patch_readme_prompt(self.SAMPLE_README, "table", "New table prompt")
        assert result.count("<b>table</b> prompt") == 1
        assert "New table prompt" in result

    def test_no_user_prompts_section(self):
        readme = "# No prompts here\n"
        result = patch_readme_prompt(readme, "math", "Math prompt")
        assert result == readme

    def test_blank_line_before_next_section(self):
        result = patch_readme_prompt(self.SAMPLE_README, "math", "Math prompt")
        idx = result.rfind("</details>")
        after = result[idx:]
        assert "\n\n##" in after or "\n\n\n##" in after

    def test_encodes_newlines_in_prompt_body(self):
        result = patch_readme_prompt(self.SAMPLE_README, "math", "Line 1\nDocument: [[DOCUMENT]]")
        assert "<br/>Document: [[DOCUMENT]]" in result
        assert "Line 1\nDocument: [[DOCUMENT]]</pre>" not in result

    def test_supports_hyphenated_config_names(self):
        readme = (
            " * User prompts (from column `text`):\n"
            "\n"
            "   <details>\n"
            "   <summary><b>math-v2</b> prompt</summary>\n"
            "\n"
            '   <pre style="white-space: pre-wrap;">Math prompt here</pre>\n'
            "\n"
            "   </details>\n"
            "\n"
            "## 🔄 Generation Progress\n"
        )
        result = patch_readme_prompt(readme, "faq-v2", "FAQ prompt here")
        assert "<b>math-v2</b> prompt" in result
        assert "<b>faq-v2</b> prompt" in result


class TestPatchReadmeConfigs:
    SAMPLE_README = (
        "---\n"
        "configs:\n"
        "- config_name: table\n"
        "  data_files:\n"
        "  - split: train\n"
        "    path: table/**/*.parquet\n"
        "train-eval-index:\n"
        "- config: table\n"
        "  task: text-generation\n"
        "  task_id: language-modeling\n"
        "  splits:\n"
        "    train_split: train\n"
        "    eval_split:\n"
        "  col_mapping:\n"
        "    text: text\n"
        "---\n"
        "\n"
        "# Dataset Card\n"
        "\n"
        "You can load the dataset using\n"
        "```python\n"
        "from datasets import load_dataset\n"
        "\n"
        'ds_table = load_dataset("org/repo", "table")\n'
        "```\n"
    )

    def test_adds_new_config(self):
        result = patch_readme_configs(self.SAMPLE_README, "org/repo", "math", "train")
        assert "config_name: math" in result
        assert "config_name: table" in result

    def test_adds_all_config_when_multiple(self):
        result = patch_readme_configs(self.SAMPLE_README, "org/repo", "math", "train")
        assert "config_name: all" in result

    def test_updates_load_example(self):
        result = patch_readme_configs(self.SAMPLE_README, "org/repo", "math", "train")
        assert 'load_dataset("org/repo", "math")' in result
        assert 'load_dataset("org/repo", "table")' in result
        assert 'load_dataset("org/repo", "all")' in result

    def test_skips_existing_config(self):
        result = patch_readme_configs(self.SAMPLE_README, "org/repo", "table", "train")
        assert result == self.SAMPLE_README

    def test_no_frontmatter_updates_load_example(self):
        readme = (
            "# Dataset Card\n\n"
            "You can load the dataset using\n"
            "```python\n"
            "from datasets import load_dataset\n"
            "\n"
            'ds = load_dataset("org/repo")\n'
            "```\n"
        )
        result = patch_readme_configs(readme, "org/repo", "math", "train")
        assert 'load_dataset("org/repo", "math")' in result


class TestPatchReadmePromptEdgeCases:
    def test_prompt_section_at_end_of_file(self):
        readme = (
            " * User prompts (from column `text`):\n"
            "\n"
            "   <details>\n"
            "   <summary><b>table</b> prompt</summary>\n"
            "\n"
            '   <pre style="white-space: pre-wrap;">Table prompt here</pre>\n'
            "\n"
            "   </details>\n"
        )
        result = patch_readme_prompt(readme, "math", "Math prompt")
        assert "<b>math</b> prompt" in result
        assert "<b>table</b> prompt" in result
