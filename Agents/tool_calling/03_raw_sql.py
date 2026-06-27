"""3. The tempting shortcut: one tool that accepts SQL.

Powerful, but riskier. The guardrails live in run_query: a read-only
connection, plus a row limit. Read-only protects data from being changed.
It does not protect data from being revealed, so treat this as a
demonstration of the pattern (and its dangers), not a production design.

Run setup_db.py first, then:

    export OPENAI_API_KEY="sk-..."
    python 03_raw_sql.py
"""

import json
import sqlite3

from openai import OpenAI

client = OpenAI()

tools = [
    {
        "type": "function",
        "function": {
            "name": "run_query",
            "description": "Run a read-only SQL query against the shop database and return the rows",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {
                        "type": "string",
                        "description": "A single SELECT statement"
                    }
                },
                "required": ["sql"]
            }
        }
    }
]

# The model needs to know the schema to write SQL. With narrow tools this
# stays hidden inside the function; with raw SQL we have to describe it.
SCHEMA = """
Tables:
  customers(id, name, email)
  orders(id, customer_id, product, quantity, status, order_date)
orders.customer_id references customers.id.
status is one of: processing, shipped, delivered.
"""


def run_query(sql: str):
    conn = sqlite3.connect("file:shop.db?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(sql).fetchall()
    finally:
        conn.close()
    return {
        "rows": [dict(row) for row in rows[:50]]
    }


def main():
    messages = [
        {
            "role": "system",
            "content": "You answer questions about the shop database. "
                       "Use the run_query tool with a single read-only "
                       "SELECT statement. Schema:\n" + SCHEMA
        },
        {"role": "user", "content": "Which product sold the most units?"}
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        tools=tools,
        messages=messages
    )

    tool_call = response.choices[0].message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    print(arguments["sql"])

    result = run_query(**arguments)

    messages.append(response.choices[0].message)
    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(result)
    })

    final_response = client.chat.completions.create(
        model="gpt-4o-mini",
        tools=tools,
        messages=messages
    )
    print(final_response.choices[0].message.content)


if __name__ == "__main__":
    main()
