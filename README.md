# datatrove

## Installation

```bash
# readability-lxml==0.8.3.dev0
pip install --upgrade --ignore-installed git+https://github.com/huggingface/python-readability

pip install -e ".[dev]"
```

Install pre-commit code style hooks:
```bash
pre-commit install
```

Run the tests:
```bash
pytest -n 4  --max-worker-restart=0 --dist=loadfile tests
```