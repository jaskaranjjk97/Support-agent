from langchain_core.tools import StructuredTool
from app.tools.account_tool import (get_account_details,get_order_status,reset_password)

ACCOUNT_TOOLS=[
    StructuredTool.from_function(get_account_details),
    StructuredTool.from_function(get_order_status),
    StructuredTool.from_function(reset_password),
]
