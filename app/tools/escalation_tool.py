import uuid
from datetime import UTC, datetime


def create_escalation_ticket(user_message: str, severity: str) -> str:
    """Create a Escalation ticket for human Intervention.

    Args:
         user_message:The original complaint message from the user.
         severity: How severe the case is , could be ("low","medium","high")


    Returns:
            Ticket details including a unique ticket_id , severity , created_at

    """
    # Creating a unique ticket with hexadecimal and using 8 characters as a ticket scheme
    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"

    return {
        "ticket_id": ticket_id,
        "severity": severity,
        "user_message": user_message,
        "created_at": datetime.now(UTC).isoformat(),
        "status": "open",
    }
