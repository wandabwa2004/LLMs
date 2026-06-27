"""2. Letting the model choose between tools.

The model reads the user's request, looks at the tools available, decides
which one fits, and fills in the arguments. The status parameter uses an
enum, so the model cannot send a random status value.

Run setup_db.py first, then:

    export OPENAI_API_KEY="sk-..."
    python 02_choose_tools.py
"""

import json
import sqlite3

from openai import OpenAI

client = OpenAI()

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
                        "description": "The customer's email address"
                    }
                },
                "required": ["customer_email"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "count_orders_by_status",
            "description": "Count how many orders currently have a given status",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["processing", "shipped", "delivered"],
                        "description": "The order status to count"
                    }
                },
                "required": ["status"]
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


def count_orders_by_status(status: str):
    conn = sqlite3.connect("shop.db")
    count = conn.execute(
        "SELECT COUNT(*) FROM orders WHERE status = ?",
        (status,)
    ).fetchone()[0]
    conn.close()
    return {
        "status": status,
        "count": count
    }


# Map tool names to the functions our code is willing to run.
AVAILABLE_FUNCTIONS = {
    "get_customer_orders": get_customer_orders,
    "count_orders_by_status": count_orders_by_status,
}


def main():
    messages = [
        {"role": "user", "content": "How many orders are still processing?"}
    ]
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        tools=tools,
        messages=messages
    )

    tool_call = response.choices[0].message.tool_calls[0]
    print(tool_call.function.name)
    print(tool_call.function.arguments)

    # Run the tool the model picked, then let it phrase the final answer.
    function = AVAILABLE_FUNCTIONS[tool_call.function.name]
    arguments = json.loads(tool_call.function.arguments)
    result = function(**arguments)

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
