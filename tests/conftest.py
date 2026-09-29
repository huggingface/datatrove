import os


try:
    from hypothesis import settings
except ImportError:  # hypothesis is only installed with the `testing` extra
    settings = None

if settings is not None:
    # ci: reproducible inputs (same tests and tool versions give the same examples), so CI does not fail at random.
    # dev: local default. thorough: explore before a release.
    # deadline=None because the first call to a tokenizer (spaCy, nltk) is slow.
    settings.register_profile("ci", max_examples=50, derandomize=True, deadline=None)
    settings.register_profile("dev", max_examples=100, deadline=None)
    settings.register_profile("thorough", max_examples=1000, deadline=None)
    # CI is set by GitHub Actions (and most CI systems); HYPOTHESIS_PROFILE or --hypothesis-profile overrides it
    settings.load_profile(os.getenv("HYPOTHESIS_PROFILE", "ci" if os.getenv("CI") else "dev"))
