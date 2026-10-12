import json

from app import config


def test_data_files_load():
    with open(config.ACCOUNTS_FILE) as f:
        accounts = json.load(f)
    with open(config.ORDERS_FILE) as f:
        orders = json.load(f)

    assert len(accounts) >= 3
    assert len(orders) >= 3
    print(f"Loaded {len(accounts)} accounts, {len(orders)} orders")


def test_help_docs_present():
    docs = list[config.DATA_DIR.glob("*.md")]
    assert len(docs) >= 4
    print(f"Found {len(docs)} help docs: {[d.name for d in docs]}")


def test_openai_key_loaded():
    assert config.OPENAI_API_KEY, "OPENAI_API_KEY missing"
    print("OPENAI_API_KEY loaded")


if __name__ == "__main__":
    test_data_files_load()
    test_help_docs_present()
    test_openai_key_loaded()
    print("Setup OK")
