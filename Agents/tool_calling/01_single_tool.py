"""1. A single safe tool.

The model can only provide a customer email. It cannot choose a table,
change the query, ask for every customer, or delete anything.

Run setup_db.py first, then:

    export OPENAI_API_KEY="sk-..."
    python 01_single_tool.py
"""

import json
import sqlite3

from openai import OpenAI

client = OpenAI()

# One tool. It takes one email address. That is all.
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_customer_orders",
            "description": "Look up all orders placed by a customer, given their email address",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_email": {
                        "type": "string",
                        "description": "The customer's email address, for example jane@example.com"
                    }
                },
                "required": ["customer_email"]
            }
        }
    }
]


def get_customer_orders(customer_email: str):
    conn = sqlite3.connect("shop.db")
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT product, quantity, status, order_date
        FROM orders o
        JOIN customers c ON c.id = o.customer_id
        WHERE c.email = ?
        ORDER BY order_date DESC
        """,
        (customer_email,)
    ).fetchall()
    conn.close()
    return {
        "email": customer_email,
        "orders": [dict(row) for row in rows]
    }


def main():
    messages = [
        {"role": "user", "content": "What has jane@example.com ordered?"}
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        tools=tools,
        messages=messages
    )
    print(response.choices[0].message)

    # The model asked us to run a function. Our code decides what to do.
    tool_call = response.choices[0].message.tool_calls[0]
    arguments = json.loads(tool_call.function.arguments)
    result = get_customer_orders(**arguments)

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
