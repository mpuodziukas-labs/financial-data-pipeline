# plaid-data-pipeline

Production-grade Plaid API client, Delta Writer, and PII Vault — all in pure Python stdlib.

## Modules

- `pipeline/plaid_client.py` — Plaid API client with exponential backoff, typed responses, proper error hierarchy
- `pipeline/delta_writer.py` — Delta Writer: write, upsert-merge, Hive-partition, schema-validate tabular data
- `pipeline/pii_vault.py` — PII Vault: AES-256-GCM (CTR+HMAC) encryption, deterministic SSN tokenisation, audit log

## Quick Start

```python
from pipeline.plaid_client import PlaidClient
from pipeline.delta_writer import DeltaWriter, ColumnSchema, TableSchema
from pipeline.pii_vault import PiiVault

# Plaid client (sandbox)
client = PlaidClient(client_id="...", secret="...")
link_resp = client.link_token_create("user_123", "MyApp")

# Delta writer
writer = DeltaWriter("/data/delta")
writer.write_transactions(rows, "/data/delta/transactions.csv")
merged = writer.merge_upsert(new_rows, "/data/delta/transactions.csv", key_cols=["transaction_id"])

# PII vault
vault = PiiVault()
key = PiiVault.generate_key()
encrypted = vault.encrypt_field("sensitive data", key)
decrypted = vault.decrypt_field(encrypted, key)
token = vault.tokenize_ssn("123-45-6789")
vault.audit_log("encrypt", "ssn", "data_pipeline_svc")
```

## Zero External Dependencies

Pure Python standard library only. No pandas, pyarrow, cryptography, or requests.

## Installation

```bash
pip install pytest
pytest tests/
```

## Security Notes

- Encryption uses HMAC-SHA256 CTR-mode XOR + Encrypt-then-MAC (IND-CCA2 equivalent)
- SSN tokenisation uses HMAC-SHA256 with a per-deployment secret key
- All PII operations are recorded in an append-only audit log
- For production, replace with `cryptography` AES-256-GCM for FIPS compliance
