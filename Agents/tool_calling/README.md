# Tool calling at the database level

Runnable code for the article on LLM tool calling against a SQLite database.
Each script is self-contained and matches the snippets in the article.

## Setup

Run the setup script. It creates a virtualenv, installs dependencies,
builds `shop.db`, and seeds a `.env` from the template:

```bash
./setup.sh
```

Then add your key to `.env` (the examples use `gpt-4o-mini`) and load it:

```bash
source .venv/bin/activate
export $(grep -v '^#' .env | xargs)   # OPENAI_API_KEY=sk-...
```

<details>
<summary>Manual setup</summary>

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="sk-..."   # the examples use gpt-4o-mini
python setup_db.py               # creates shop.db (2 customers, 3 orders)
```

</details>

## The examples

| File | What it shows |
|------|---------------|
| `setup_db.py` | Builds the tiny `shop.db` database |
| `01_single_tool.py` | One narrow tool: look up a customer's orders by email |
| `02_choose_tools.py` | The model picks between two tools (note the `status` enum) |
| `03_raw_sql.py` | The tempting shortcut: let the model write SQL, behind a read-only connection |

Run them in order:

```bash
python 01_single_tool.py
python 02_choose_tools.py
python 03_raw_sql.py
```

## The point

The model asks, your code decides. Every safety control — the bound SQL
parameter (`?`), the `enum`, the `mode=ro` read-only connection, the row
limit — lives in the gap between the model asking and your code acting.
Start with narrow tools. Only reach for raw SQL once the locks are in place.
