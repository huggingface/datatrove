# datatrove

## Installation

```bash
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
