# Data Inspector

## Setup
1. Create a virtual environment and install dependencies:

```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Running the app

Sessions are auto-discovered under the `DB` folder (override with `--db-root`).

```
python main.py --db-root DB --session 260227_155409 --debug
```

Open http://127.0.0.1:8050 in your browser to explore the synchronized dashboard.
