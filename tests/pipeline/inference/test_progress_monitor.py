"""Tests for progress monitor rendering and monitor-step behavior."""

from pathlib import Path

import pytest

from datatrove.pipeline.inference import progress_monitor as pm
from datatrove.pipeline.inference.dataset_card_generator import InferenceDatasetCardParams
from datatrove.pipeline.inference.progress_monitor import (
    InferenceProgressMonitor,
    _append_progress_section,
    _bounded_completed,
    _extract_config_names,
    _render_bar_and_counts,
    _upsert_config_progress_line,
    calculate_eta,
    create_progress_section_markdown,
    format_completion_datetime,
    format_time_remaining,
    patch_readme_progress,
    render_progress_bar,
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


class TestBoundedCompleted:
    def test_negative_total_returns_zero(self):
        assert _bounded_completed(50, -1) == 0

    def test_zero_total_returns_zero(self):
        assert _bounded_completed(50, 0) == 0

    def test_negative_completed_returns_zero(self):
        assert _bounded_completed(-10, 100) == 0

    def test_over_total_clamps(self):
        assert _bounded_completed(200, 100) == 100

    def test_normal_passthrough(self):
        assert _bounded_completed(50, 100) == 50

    def test_at_boundary(self):
        assert _bounded_completed(0, 100) == 0
        assert _bounded_completed(100, 100) == 100


class TestFormatCompletionDatetime:
    def test_formats_utc(self):
        from datetime import datetime, timezone

        ts = datetime(2026, 3, 15, 14, 30, tzinfo=timezone.utc).timestamp()
        result = format_completion_datetime(ts)
        assert "Mar 15 2026" in result
        assert "14:30 UTC" in result


class TestRenderBarAndCounts:
    def test_zero_total(self):
        result = _render_bar_and_counts(0, 0)
        assert "0%" in result
        assert "0/0 docs" in result

    def test_half_done(self):
        result = _render_bar_and_counts(500, 1000)
        assert "50%" in result
        assert "●" * 10 in result
        assert "○" * 10 in result

    def test_complete(self):
        result = _render_bar_and_counts(1000, 1000)
        assert "100%" in result
        assert "●" * 20 in result
        assert "○" not in result

    def test_clamps_values_over_total(self):
        result = _render_bar_and_counts(2500, 1000)
        assert "100%" in result
        assert "1,000/1,000 docs" in result

    def test_negative_completed_clamped(self):
        result = _render_bar_and_counts(-5, 100)
        assert "0%" in result
        assert "0/100 docs" in result


class TestRenderProgressBar:
    def test_in_progress_shows_eta(self):
        result = render_progress_bar(50, 100, 0.0, 100.0)
        assert "⏱️" in result
        assert "📅" in result
        assert "remaining" in result

    def test_zero_completed_shows_waiting_for_first_upload(self):
        result = render_progress_bar(0, 100, 0.0, 10.0)
        assert "waiting for first shard upload" in result

    def test_complete_shows_checkmark(self):
        result = render_progress_bar(100, 100, 0.0, 100.0)
        assert "✅ Complete" in result

    def test_overflow_is_clamped_to_complete(self):
        result = render_progress_bar(180, 100, 0.0, 100.0)
        assert "100/100 docs" in result
        assert "✅ Complete" in result

    def test_br_separates_bar_from_eta(self):
        result = render_progress_bar(50, 100, 0.0, 100.0)
        assert "<br>" in result
        parts = result.split("<br>")
        assert len(parts) == 2
        assert "docs" in parts[0]
        assert "⏱️" in parts[1]

    def test_br_in_complete(self):
        result = render_progress_bar(100, 100, 0.0, 100.0)
        assert "<br>" in result

    def test_br_in_waiting(self):
        result = render_progress_bar(0, 100, 0.0, 10.0)
        assert "<br>" in result


class TestCreateProgressSectionMarkdown:
    def test_default_config(self):
        result = create_progress_section_markdown("default", 500, 1000, 0.0, 100.0)
        assert "Generation Progress" in result
        assert "**default**:" in result
        assert "●" in result

    def test_named_config(self):
        result = create_progress_section_markdown("math", 300, 1000, 0.0, 100.0)
        assert "**math**:" in result
        assert "remaining" in result

    def test_zero_shows_waiting_for_first_upload(self):
        result = create_progress_section_markdown("math", 0, 1000, 0.0, 10.0)
        assert "waiting for first shard upload" in result

    def test_last_updated_timestamp(self):
        result = create_progress_section_markdown("default", 500, 1000, 0.0, 100.0)
        assert "Last updated:" in result


class TestPatchReadmeProgress:
    MULTI_CONFIG_README = (
        "# Dataset Card\n\n"
        "## 🔄 Generation Progress\n\n"
        "**faq**: [○○○○○○○○○○○○○○○○○○○○] 0% • 0/1,000 docs<br>⏱️ calculating...\n\n"
        "**math**: [●●●●●●●●●●○○○○○○○○○○] 50% • 500/1,000 docs<br>⏱️ 5m remaining • 📅 Feb 15, 21:00 UTC\n\n"
        "*Last updated: 2026-02-15 20:00:00 UTC*\n\n"
        "## Dataset Stats\n"
    )

    SINGLE_CONFIG_README = (
        "# Dataset Card\n\n"
        "## 🔄 Generation Progress\n\n"
        "**default**: [●●●●●●●●●●○○○○○○○○○○] 50% • 500/1,000 docs<br>⏱️ 5m remaining • 📅 Feb 15, 21:00 UTC\n\n"
        "*Last updated: 2026-02-15 20:00:00 UTC*\n\n"
        "## Dataset Stats\n"
    )

    def test_updates_only_owned_config(self):
        result = patch_readme_progress(self.MULTI_CONFIG_README, "math", 700, 1000, 0.0, 200.0)
        assert "700/1,000 docs" in result
        assert "**faq**: [○○○○○○○○○○○○○○○○○○○○] 0% • 0/1,000 docs" in result

    def test_timestamp_is_updated(self):
        result = patch_readme_progress(self.MULTI_CONFIG_README, "math", 700, 1000, 0.0, 200.0)
        assert "2026-02-15 20:00:00 UTC" not in result
        assert "Last updated:" in result

    def test_other_sections_untouched(self):
        result = patch_readme_progress(self.MULTI_CONFIG_README, "math", 700, 1000, 0.0, 200.0)
        assert "# Dataset Card" in result
        assert "## Dataset Stats" in result

    def test_inserts_new_config_line(self):
        result = patch_readme_progress(self.MULTI_CONFIG_README, "table", 100, 1000, 0.0, 50.0)
        assert "**table**:" in result
        assert "**faq**:" in result
        assert "**math**:" in result

    def test_inserted_config_on_own_line(self):
        result = patch_readme_progress(self.MULTI_CONFIG_README, "table", 100, 1000, 0.0, 50.0)
        lines = result.split("\n")
        config_lines = [i for i, line in enumerate(lines) if line.startswith("**") and "**:" in line]
        for idx in config_lines:
            prev = lines[idx - 1] if idx > 0 else ""
            assert prev == "" or prev.startswith("##"), f"Line {idx} not preceded by blank: {prev!r}"

    def test_complete_config(self):
        result = patch_readme_progress(self.MULTI_CONFIG_README, "math", 1000, 1000, 0.0, 200.0)
        lines = result.split("\n")
        math_line = next(line for line in lines if "**math**:" in line)
        assert "✅ Complete" in math_line

    def test_single_config_progress_update(self):
        result = patch_readme_progress(self.SINGLE_CONFIG_README, "default", 800, 1000, 0.0, 200.0)
        assert "**default**:" in result
        assert "800/1,000 docs" in result

    def test_single_config_timestamp_updated(self):
        result = patch_readme_progress(self.SINGLE_CONFIG_README, "default", 800, 1000, 0.0, 200.0)
        assert "2026-02-15 20:00:00 UTC" not in result
        assert "Last updated:" in result


class TestFormatTimeRemaining:
    def test_under_minute(self):
        assert format_time_remaining(30) == "< 1m"

    def test_minutes(self):
        assert format_time_remaining(90) == "1m"

    def test_hours_and_minutes(self):
        assert format_time_remaining(5400) == "1h 30m"

    def test_days_and_hours(self):
        assert format_time_remaining(25 * 3600) == "1d 1h"

    def test_weeks_and_days(self):
        assert format_time_remaining(10 * 24 * 3600) == "1w 3d"

    def test_months_and_days(self):
        result = format_time_remaining(45 * 24 * 3600)
        assert result.startswith("1mo")

    def test_years_and_months(self):
        result = format_time_remaining(400 * 24 * 3600)
        assert result.startswith("1y")

    def test_large_hours_use_days(self):
        result = format_time_remaining(3666 * 3600)
        assert "d" in result or "mo" in result or "w" in result


class TestCalculateEta:
    def test_basic_eta(self):
        seconds_remaining, _ = calculate_eta(50, 100, 100.0)
        assert seconds_remaining == pytest.approx(100.0)

    def test_zero_completed(self):
        seconds_remaining, _ = calculate_eta(0, 100, 100.0)
        assert seconds_remaining == 0.0

    def test_zero_elapsed(self):
        seconds_remaining, _ = calculate_eta(0, 100, 0.0)
        assert seconds_remaining == 0.0

    def test_fully_complete(self):
        seconds_remaining, _ = calculate_eta(100, 100, 50.0)
        assert seconds_remaining == pytest.approx(0.0)

    def test_returns_datetime_in_future(self):
        from datetime import datetime, timezone

        _, completion_dt = calculate_eta(50, 100, 100.0)
        assert completion_dt > datetime.now(timezone.utc)


class TestUpsertConfigProgressLine:
    def test_replaces_existing_config_line(self):
        readme = (
            "## 🔄 Generation Progress\n\n"
            "**math**: old progress\n\n"
            "*Last updated: 2026-01-01*\n"
        )
        result = _upsert_config_progress_line(readme, "math", "**math**: new progress")
        assert "**math**: new progress" in result
        assert "old progress" not in result

    def test_inserts_before_timestamp_when_config_missing(self):
        readme = (
            "## 🔄 Generation Progress\n\n"
            "**faq**: faq progress\n\n"
            "*Last updated: 2026-01-01*\n"
        )
        result = _upsert_config_progress_line(readme, "math", "**math**: new progress")
        assert result is not None
        assert "**math**: new progress" in result
        assert "**faq**: faq progress" in result
        # math should appear before the timestamp
        math_idx = result.index("**math**")
        ts_idx = result.index("*Last updated:")
        assert math_idx < ts_idx

    def test_returns_none_without_progress_section(self):
        readme = "# No progress section here\n"
        result = _upsert_config_progress_line(readme, "math", "**math**: new")
        assert result is None

    def test_appends_at_end_of_progress_section_without_timestamp(self):
        readme = (
            "## 🔄 Generation Progress\n\n"
            "**faq**: faq progress\n\n"
            "## Dataset Stats\n"
        )
        result = _upsert_config_progress_line(readme, "math", "**math**: new progress")
        assert result is not None
        assert "**math**: new progress" in result
        # Should be before the next section
        math_idx = result.index("**math**")
        stats_idx = result.index("## Dataset Stats")
        assert math_idx < stats_idx

    def test_appends_at_very_end_without_next_section_or_timestamp(self):
        readme = "## 🔄 Generation Progress\n\n**faq**: faq progress\n"
        result = _upsert_config_progress_line(readme, "math", "**math**: new progress")
        assert result is not None
        assert result.endswith("**math**: new progress\n\n")


class TestExtractConfigNames:
    def test_current_config_is_first(self):
        readme = (
            "---\n"
            "configs:\n"
            "- config_name: table\n"
            "- config_name: faq\n"
            "---\n"
        )
        result = _extract_config_names(readme, "math")
        assert result[0] == "math"

    def test_deduplicates_current_config(self):
        readme = (
            "---\n"
            "configs:\n"
            "- config_name: math\n"
            "- config_name: faq\n"
            "---\n"
        )
        result = _extract_config_names(readme, "math")
        assert result.count("math") == 1
        assert "faq" in result

    def test_skips_all_config(self):
        readme = (
            "---\n"
            "configs:\n"
            "- config_name: all\n"
            "- config_name: faq\n"
            "- config_name: math\n"
            "---\n"
        )
        result = _extract_config_names(readme, "faq")
        assert "all" not in result
        assert "faq" in result
        assert "math" in result

    def test_no_configs_returns_only_current(self):
        readme = "# Dataset Card\n"
        result = _extract_config_names(readme, "math")
        assert result == ["math"]


class TestAppendProgressSection:
    def test_appends_to_existing_readme(self):
        readme = "# Dataset Card\n\nSome content"
        section = "## 🔄 Generation Progress\n\n**math**: bar"
        result = _append_progress_section(readme, section)
        assert result.startswith("# Dataset Card")
        assert "## 🔄 Generation Progress" in result
        assert result.count("\n\n") >= 1

    def test_empty_readme(self):
        result = _append_progress_section("", "## Progress\n\n**math**: bar")
        assert result.startswith("## Progress")
        assert result.endswith("\n")

    def test_strips_trailing_whitespace(self):
        readme = "# Card\n\n   \n\n"
        section = "\n\n## Progress\n\n"
        result = _append_progress_section(readme, section)
        assert "# Card\n\n## Progress\n" == result


class TestPatchReadmeProgressEdgeCases:
    def test_returns_unchanged_when_no_progress_section(self):
        readme = "# No progress\n"
        result = patch_readme_progress(readme, "math", 50, 100, 0.0, 50.0)
        assert result == readme


class TestInferenceProgressMonitorStep:
    def test_nonzero_rank_only_passthrough(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        params = _make_card_params(stats_path=str(tmp_path / "stats.json"))
        monitor = InferenceProgressMonitor(params=params)

        def fail_total(*_args, **_kwargs) -> int:
            raise AssertionError("get_total_expected_documents should not be called for non-zero ranks")

        monkeypatch.setattr(pm, "get_total_expected_documents", fail_total)

        passthrough = list(monitor.run(data=["d1", "d2"], rank=1))

        assert passthrough == ["d1", "d2"]

    def test_updates_existing_progress_section(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        stats_path = tmp_path / "stats.json"
        params = _make_card_params(stats_path=str(stats_path), prompt_template="Prompt [[DOCUMENT]]")
        monitor = InferenceProgressMonitor(params=params, update_interval=1)

        uploads: list[tuple[str, str]] = []
        build_calls: list[dict[str, object]] = []

        monkeypatch.setattr(pm, "get_total_expected_documents", lambda *_args, **_kwargs: 100)
        monkeypatch.setattr(pm, "count_documents_in_repo", lambda *_args, **_kwargs: 25)
        monkeypatch.setattr(
            pm,
            "_download_readme",
            lambda _repo_id: "## 🔄 Generation Progress\n\n*Last updated: old*\n",
        )
        monkeypatch.setattr(pm, "patch_readme_progress", lambda content, *_args, **_kwargs: content + "\nprogress")
        monkeypatch.setattr(pm, "patch_readme_prompt", lambda content, *_args, **_kwargs: content + "\nprompt")
        monkeypatch.setattr(pm, "patch_readme_configs", lambda content, *_args, **_kwargs: content + "\nconfigs")
        monkeypatch.setattr(pm, "_upload_readme", lambda repo_id, content: uploads.append((repo_id, content)))
        monkeypatch.setattr(
            pm,
            "build_and_upload_dataset_card",
            lambda *, params, progress_section: build_calls.append(
                {"params": params, "progress_section": progress_section}
            ),
        )

        sleeps = {"count": 0}

        def fake_sleep(_seconds: float) -> None:
            sleeps["count"] += 1
            if sleeps["count"] == 1:
                stats_path.write_text("{}", encoding="utf-8")

        monkeypatch.setattr(pm.time, "sleep", fake_sleep)

        assert list(monitor.run(rank=0)) == []
        assert len(uploads) == 1
        assert uploads[0][0] == "org/repo"
        assert "progress" in uploads[0][1]
        assert "prompt" in uploads[0][1]
        assert "configs" in uploads[0][1]
        assert build_calls == []

    def test_builds_initial_progress_section_when_missing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        stats_path = tmp_path / "stats.json"
        params = _make_card_params(stats_path=str(stats_path))
        monitor = InferenceProgressMonitor(params=params, update_interval=1)

        uploads: list[tuple[str, str]] = []
        build_calls: list[dict[str, object]] = []

        monkeypatch.setattr(pm, "get_total_expected_documents", lambda *_args, **_kwargs: 100)
        monkeypatch.setattr(pm, "count_documents_in_repo", lambda *_args, **_kwargs: 10)
        monkeypatch.setattr(pm, "_download_readme", lambda _repo_id: None)
        monkeypatch.setattr(pm, "_upload_readme", lambda repo_id, content: uploads.append((repo_id, content)))
        monkeypatch.setattr(
            pm,
            "build_and_upload_dataset_card",
            lambda *, params, progress_section: build_calls.append(
                {"params": params, "progress_section": progress_section}
            ),
        )

        sleeps = {"count": 0}

        def fake_sleep(_seconds: float) -> None:
            sleeps["count"] += 1
            if sleeps["count"] == 1:
                stats_path.write_text("{}", encoding="utf-8")

        monkeypatch.setattr(pm.time, "sleep", fake_sleep)

        assert list(monitor.run(rank=0)) == []
        assert uploads == []
        assert len(build_calls) == 1
        assert build_calls[0]["params"] == params
        assert "## 🔄 Generation Progress" in str(build_calls[0]["progress_section"])

    def test_restores_missing_progress_section_from_existing_readme(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        stats_path = tmp_path / "stats.json"
        params = _make_card_params(
            stats_path=str(stats_path),
            prompt_template="Prompt [[DOCUMENT]]",
            prompt_template_name="faq",
        )
        monitor = InferenceProgressMonitor(params=params, update_interval=1)

        uploads: list[tuple[str, str]] = []
        build_calls: list[dict[str, object]] = []

        readme_without_progress = (
            "---\n"
            "configs:\n"
            "- config_name: table\n"
            "  data_files:\n"
            "  - split: train\n"
            "    path: table/**/*.parquet\n"
            "- config_name: tutorial\n"
            "  data_files:\n"
            "  - split: train\n"
            "    path: tutorial/**/*.parquet\n"
            "---\n\n"
            "# Dataset Card\n"
        )

        counts = {
            "faq": 100,
            "table": 50,
            "tutorial": 75,
        }

        def fake_count(_repo_id: str, config_name: str = "default") -> int:
            return counts[config_name]

        monkeypatch.setattr(pm, "get_total_expected_documents", lambda *_args, **_kwargs: 100)
        monkeypatch.setattr(pm, "count_documents_in_repo", fake_count)
        monkeypatch.setattr(pm, "_download_readme", lambda _repo_id: readme_without_progress)
        monkeypatch.setattr(pm, "patch_readme_prompt", lambda content, *_args, **_kwargs: content)
        monkeypatch.setattr(pm, "patch_readme_configs", lambda content, *_args, **_kwargs: content)
        monkeypatch.setattr(pm, "_upload_readme", lambda repo_id, content: uploads.append((repo_id, content)))
        monkeypatch.setattr(
            pm,
            "build_and_upload_dataset_card",
            lambda *, params, progress_section: build_calls.append(
                {"params": params, "progress_section": progress_section}
            ),
        )

        sleeps = {"count": 0}

        def fake_sleep(_seconds: float) -> None:
            sleeps["count"] += 1
            if sleeps["count"] == 1:
                stats_path.write_text("{}", encoding="utf-8")

        monkeypatch.setattr(pm.time, "sleep", fake_sleep)

        assert list(monitor.run(rank=0)) == []
        assert len(uploads) == 1
        assert uploads[0][0] == "org/repo"
        assert "**faq**:" in uploads[0][1]
        assert "**table**:" in uploads[0][1]
        assert "**tutorial**:" in uploads[0][1]
        assert build_calls == []

    def test_stops_if_inference_job_is_not_running(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        params = _make_card_params(stats_path=str(tmp_path / "stats.json"))
        monitor = InferenceProgressMonitor(params=params, inference_job_id="123", update_interval=1)

        monkeypatch.setattr(pm, "get_total_expected_documents", lambda *_args, **_kwargs: 100)
        monkeypatch.setattr(monitor, "_is_job_running", lambda _job_id: False)

        def fail_count(*_args, **_kwargs) -> int:
            raise AssertionError("count_documents_in_repo should not be called when job has stopped")

        monkeypatch.setattr(pm, "count_documents_in_repo", fail_count)

        assert list(monitor.run(rank=0)) == []

    def test_rank_zero_passes_through_data(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        stats_path = tmp_path / "stats.json"
        stats_path.write_text("{}", encoding="utf-8")  # stats present = immediate exit
        params = _make_card_params(stats_path=str(stats_path))
        monitor = InferenceProgressMonitor(params=params, update_interval=1)

        monkeypatch.setattr(pm, "get_total_expected_documents", lambda *_args, **_kwargs: 100)

        result = list(monitor.run(data=["a", "b"], rank=0))
        assert result == ["a", "b"]

    def test_catches_update_errors_and_continues(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        stats_path = tmp_path / "stats.json"
        params = _make_card_params(stats_path=str(stats_path))
        monitor = InferenceProgressMonitor(params=params, update_interval=1)

        monkeypatch.setattr(pm, "get_total_expected_documents", lambda *_args, **_kwargs: 100)
        monkeypatch.setattr(pm, "count_documents_in_repo", lambda *_args, **_kwargs: 10)

        def exploding_download(_repo_id: str) -> str:
            raise RuntimeError("network error")

        monkeypatch.setattr(pm, "_download_readme", exploding_download)

        sleeps = {"count": 0}

        def fake_sleep(_seconds: float) -> None:
            sleeps["count"] += 1
            if sleeps["count"] == 1:
                stats_path.write_text("{}", encoding="utf-8")

        monkeypatch.setattr(pm.time, "sleep", fake_sleep)

        # Should not raise, should complete after catching the error
        assert list(monitor.run(rank=0)) == []
        assert sleeps["count"] == 1

    def test_skips_prompt_patch_when_no_template(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        stats_path = tmp_path / "stats.json"
        params = _make_card_params(stats_path=str(stats_path), prompt_template=None)
        monitor = InferenceProgressMonitor(params=params, update_interval=1)

        uploads: list[tuple[str, str]] = []
        prompt_patch_called = {"called": False}

        monkeypatch.setattr(pm, "get_total_expected_documents", lambda *_args, **_kwargs: 100)
        monkeypatch.setattr(pm, "count_documents_in_repo", lambda *_args, **_kwargs: 25)
        monkeypatch.setattr(
            pm,
            "_download_readme",
            lambda _repo_id: "## 🔄 Generation Progress\n\n*Last updated: old*\n",
        )
        monkeypatch.setattr(pm, "patch_readme_progress", lambda content, *_args, **_kwargs: content)

        def track_prompt_patch(content, *_args, **_kwargs):
            prompt_patch_called["called"] = True
            return content

        monkeypatch.setattr(pm, "patch_readme_prompt", track_prompt_patch)
        monkeypatch.setattr(pm, "patch_readme_configs", lambda content, *_args, **_kwargs: content)
        monkeypatch.setattr(pm, "_upload_readme", lambda repo_id, content: uploads.append((repo_id, content)))

        sleeps = {"count": 0}

        def fake_sleep(_seconds: float) -> None:
            sleeps["count"] += 1
            if sleeps["count"] == 1:
                stats_path.write_text("{}", encoding="utf-8")

        monkeypatch.setattr(pm.time, "sleep", fake_sleep)

        list(monitor.run(rank=0))
        assert not prompt_patch_called["called"]
