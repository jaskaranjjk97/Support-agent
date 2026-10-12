import json 
from app import config

def _load_accounts()->dict:
    with open (config.ACCOUNTS_FILE) as f:
        return json.load(f)


def _load_orders()->dict:
    with open(config.ORDERS_FILE) as f:
        return json.load(f)


def get_account_details(user_id: str)->dict:
    """Look up a TaskFlow account by user ID.

    Args:
        user_id: The unique user identifier (e.g., "U1001").

    Returns:
        Account details including name, email, plan, seats, billing status, renewal date.
        Returns {"error": "not_found"} if the user ID doesn't exist.
    """

    accounts = _load_accounts()
    if user_id not in accounts:
        return {"error": "not_found","user_id":user_id}

    return accounts[user_id]

def get_order_status(order_id: str) -> dict:
    """Look up the status of a TaskFlow order.

    Args:
        order_id: The unique order identifier (e.g., "O5001").

    Returns:
        Order details including item, amount, status, and date.
        Returns {"error": "not_found"} if the order ID doesn't exist.
    """
    orders = _load_orders()
    if order_id not in orders:
        return {"error": "not_found", "order_id": order_id}
    return orders[order_id]

def reset_password(user_id: str) -> dict:
    """Trigger a password reset email for a TaskFlow user.

    Args:
        user_id: The unique user identifier (e.g., "U1001").

    Returns:
        Confirmation with the email address the reset was sent to.
    """
    accounts = _load_accounts()
    if user_id not in accounts:
        return {"error": "not_found", "user_id": user_id}
    email = accounts[user_id]["email"]
    return {"status": "reset_email_sent", "to": email}