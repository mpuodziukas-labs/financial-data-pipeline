"""
40+ tests for plaid-data-pipeline.

Covers plaid_client.py (mocked), delta_writer.py, and pii_vault.py.
No real Plaid API calls are made.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import tempfile
import unittest.mock as mock
import urllib.error

import pytest

from pipeline.plaid_client import (
    Account,
    AccountsGetResponse,
    Identity,
    IdentityGetResponse,
    ItemPublicTokenExchangeResponse,
    LinkTokenCreateResponse,
    PlaidAuthError,
    PlaidClient,
    PlaidError,
    PlaidInvalidRequestError,
    PlaidItemError,
    PlaidNetworkError,
    PlaidRateLimitError,
    RetryConfig,
    Transaction,
    TransactionsGetResponse,
)
from pipeline.delta_writer import (
    ColumnSchema,
    DataFrame,
    DeltaWriter,
    SchemaValidationError,
    TableSchema,
)
from pipeline.pii_vault import (
    AuditEntry,
    PiiVault,
)


# ===========================================================================
# Fixtures and helpers
# ===========================================================================

def _make_client(retry_config: RetryConfig | None = None) -> PlaidClient:
    return PlaidClient(
        client_id="test_client",
        secret="test_secret",
        base_url="https://sandbox.plaid.com",
        retry_config=retry_config or RetryConfig(max_retries=0),
    )


def _mock_http_response(data: dict) -> mock.MagicMock:
    """Create a mock urllib response that returns *data* as JSON."""
    m = mock.MagicMock()
    m.__enter__ = mock.Mock(return_value=m)
    m.__exit__ = mock.Mock(return_value=False)
    m.read.return_value = json.dumps(data).encode("utf-8")
    return m


def _mock_http_error(status: int, body: dict) -> urllib.error.HTTPError:
    """Create a mock HTTPError with a JSON body."""
    err = urllib.error.HTTPError(
        url="https://sandbox.plaid.com",
        code=status,
        msg="Error",
        hdrs=mock.MagicMock(),  # type: ignore[arg-type]
        fp=None,
    )
    err.read = mock.Mock(return_value=json.dumps(body).encode("utf-8"))  # type: ignore[method-assign]
    return err


# ===========================================================================
# PlaidClient tests
# ===========================================================================

class TestPlaidClientLinkTokenCreate:
    def test_success(self) -> None:
        client = _make_client()
        response_data = {
            "link_token": "link-sandbox-abc123",
            "expiration": "2024-01-01T00:00:00Z",
            "request_id": "req_001",
        }
        with mock.patch("urllib.request.urlopen", return_value=_mock_http_response(response_data)):
            resp = client.link_token_create(
                user_client_user_id="user_123",
                client_name="TestApp",
            )
        assert resp.link_token == "link-sandbox-abc123"
        assert resp.expiration == "2024-01-01T00:00:00Z"
        assert resp.request_id == "req_001"
        assert isinstance(resp, LinkTokenCreateResponse)

    def test_returns_link_token_create_response(self) -> None:
        client = _make_client()
        data = {"link_token": "lt", "expiration": "exp", "request_id": "rid"}
        with mock.patch("urllib.request.urlopen", return_value=_mock_http_response(data)):
            resp = client.link_token_create("u", "app")
        assert isinstance(resp, LinkTokenCreateResponse)

    def test_http_401_raises_auth_error(self) -> None:
        client = _make_client()
        with mock.patch("urllib.request.urlopen", side_effect=_mock_http_error(401, {"error_code": "INVALID_API_KEYS", "error_message": "Bad keys"})):
            with pytest.raises(PlaidAuthError):
                client.link_token_create("u", "app")

    def test_http_400_raises_invalid_request(self) -> None:
        client = _make_client()
        with mock.patch("urllib.request.urlopen", side_effect=_mock_http_error(400, {"error_code": "MISSING_FIELDS", "error_message": "Missing"})):
            with pytest.raises(PlaidInvalidRequestError):
                client.link_token_create("u", "app")


class TestPlaidClientItemPublicTokenExchange:
    def test_success(self) -> None:
        client = _make_client()
        data = {
            "access_token": "access-sandbox-xyz",
            "item_id": "item_001",
            "request_id": "req_002",
        }
        with mock.patch("urllib.request.urlopen", return_value=_mock_http_response(data)):
            resp = client.item_public_token_exchange("public-token-abc")
        assert resp.access_token == "access-sandbox-xyz"
        assert resp.item_id == "item_001"
        assert isinstance(resp, ItemPublicTokenExchangeResponse)

    def test_http_400_raises(self) -> None:
        client = _make_client()
        with mock.patch("urllib.request.urlopen", side_effect=_mock_http_error(400, {"error_code": "INVALID_PUBLIC_TOKEN", "error_message": "Bad"})):
            with pytest.raises(PlaidInvalidRequestError):
                client.item_public_token_exchange("bad-token")


class TestPlaidClientTransactionsGet:
    def _make_txn(self, idx: int) -> dict:
        return {
            "transaction_id": f"txn_{idx}",
            "account_id": f"acc_{idx}",
            "amount": 42.50 + idx,
            "date": "2024-01-15",
            "name": f"Merchant {idx}",
            "merchant_name": f"Merchant {idx}",
            "pending": False,
            "category": ["Food", "Restaurants"],
            "iso_currency_code": "USD",
        }

    def test_success(self) -> None:
        client = _make_client()
        data = {
            "transactions": [self._make_txn(i) for i in range(3)],
            "total_transactions": 3,
            "request_id": "req_003",
        }
        with mock.patch("urllib.request.urlopen", return_value=_mock_http_response(data)):
            resp = client.transactions_get("access-token", "2024-01-01", "2024-01-31")
        assert len(resp.transactions) == 3
        assert resp.total_transactions == 3
        assert isinstance(resp.transactions[0], Transaction)

    def test_empty_transactions(self) -> None:
        client = _make_client()
        data = {"transactions": [], "total_transactions": 0, "request_id": "req_004"}
        with mock.patch("urllib.request.urlopen", return_value=_mock_http_response(data)):
            resp = client.transactions_get("at", "2024-01-01", "2024-01-31")
        assert resp.transactions == []
        assert resp.total_transactions == 0

    def test_cursor_in_response(self) -> None:
        client = _make_client()
        data = {
            "transactions": [self._make_txn(0)],
            "total_transactions": 100,
            "request_id": "r",
            "next_cursor": "cursor_abc",
        }
        with mock.patch("urllib.request.urlopen", return_value=_mock_http_response(data)):
            resp = client.transactions_get("at", "2024-01-01", "2024-01-31")
        assert resp.next_cursor == "cursor_abc"


class TestPlaidClientAccountsGet:
    def test_success(self) -> None:
        client = _make_client()
        data = {
            "accounts": [
                {
                    "account_id": "acc_001",
                    "name": "Checking",
                    "official_name": "Premium Checking",
                    "type": "depository",
                    "subtype": "checking",
                    "balances": {"available": 1000.0, "current": 1050.0, "iso_currency_code": "USD"},
                }
            ],
            "request_id": "req_005",
        }
        with mock.patch("urllib.request.urlopen", return_value=_mock_http_response(data)):
            resp = client.accounts_get("access-token")
        assert len(resp.accounts) == 1
        assert resp.accounts[0].name == "Checking"
        assert resp.accounts[0].balance_current == 1050.0
        assert isinstance(resp, AccountsGetResponse)

    def test_returns_accounts_get_response(self) -> None:
        client = _make_client()
        data = {"accounts": [], "request_id": "r"}
        with mock.patch("urllib.request.urlopen", return_value=_mock_http_response(data)):
            resp = client.accounts_get("at")
        assert isinstance(resp, AccountsGetResponse)


class TestPlaidClientIdentityGet:
    def test_success(self) -> None:
        client = _make_client()
        data = {
            "accounts": [
                {
                    "account_id": "acc_001",
                    "owners": [{"names": ["John Doe"], "emails": [], "addresses": []}],
                }
            ],
            "request_id": "req_006",
        }
        with mock.patch("urllib.request.urlopen", return_value=_mock_http_response(data)):
            resp = client.identity_get("access-token")
        assert len(resp.accounts) == 1
        assert resp.accounts[0].account_id == "acc_001"
        assert isinstance(resp, IdentityGetResponse)


class TestPlaidClientRetry:
    def test_rate_limit_raises_with_zero_retries(self) -> None:
        client = _make_client(RetryConfig(max_retries=0))
        with mock.patch("urllib.request.urlopen", side_effect=_mock_http_error(429, {"error_code": "RATE_LIMIT", "error_message": "Too many"})):
            with pytest.raises(PlaidRateLimitError):
                client.link_token_create("u", "app")

    def test_network_error_raises_plaid_network_error(self) -> None:
        client = _make_client(RetryConfig(max_retries=0))
        with mock.patch("urllib.request.urlopen", side_effect=urllib.error.URLError("DNS failure")):
            with pytest.raises(PlaidNetworkError):
                client.link_token_create("u", "app")

    def test_item_error_type(self) -> None:
        client = _make_client(RetryConfig(max_retries=0))
        with mock.patch("urllib.request.urlopen", side_effect=_mock_http_error(
            400, {"error_type": "ITEM_ERROR", "error_code": "ITEM_LOGIN_REQUIRED", "error_message": "Login required"}
        )):
            with pytest.raises((PlaidItemError, PlaidInvalidRequestError)):
                client.link_token_create("u", "app")


# ===========================================================================
# DeltaWriter tests
# ===========================================================================

class TestDeltaWriterWriteTransactions:
    def test_write_creates_file(self, tmp_path: pathlib.Path) -> None:
        writer = DeltaWriter(tmp_path)
        df = [{"id": "1", "amount": "42.50", "date": "2024-01-01"}]
        dest = tmp_path / "txns.csv"
        writer.write_transactions(df, dest)
        assert dest.exists()

    def test_write_overwrite(self, tmp_path: pathlib.Path) -> None:
        writer = DeltaWriter(tmp_path)
        dest = tmp_path / "txns.csv"
        writer.write_transactions([{"id": "1"}], dest)
        writer.write_transactions([{"id": "2"}], dest, mode="overwrite")
        # Read back
        import csv
        rows = list(csv.DictReader(dest.open()))
        assert len(rows) == 1
        assert rows[0]["id"] == "2"

    def test_write_append(self, tmp_path: pathlib.Path) -> None:
        writer = DeltaWriter(tmp_path)
        dest = tmp_path / "txns.csv"
        writer.write_transactions([{"id": "1", "v": "a"}], dest)
        writer.write_transactions([{"id": "2", "v": "b"}], dest, mode="append")
        import csv
        rows = list(csv.DictReader(dest.open()))
        assert len(rows) == 2

    def test_empty_dataframe(self, tmp_path: pathlib.Path) -> None:
        writer = DeltaWriter(tmp_path)
        dest = tmp_path / "empty.csv"
        writer.write_transactions([], dest)
        assert dest.exists()


class TestDeltaWriterMergeUpsert:
    def test_upsert_new_rows(self, tmp_path: pathlib.Path) -> None:
        writer = DeltaWriter(tmp_path)
        dest = tmp_path / "data.csv"
        existing = [{"id": "1", "amount": "100"}]
        writer.write_transactions(existing, dest)
        new_rows = [{"id": "2", "amount": "200"}]
        merged = writer.merge_upsert(new_rows, dest, key_cols=["id"])
        assert len(merged) == 2

    def test_upsert_updates_existing(self, tmp_path: pathlib.Path) -> None:
        writer = DeltaWriter(tmp_path)
        dest = tmp_path / "data.csv"
        existing = [{"id": "1", "amount": "100"}]
        writer.write_transactions(existing, dest)
        update = [{"id": "1", "amount": "999"}]
        merged = writer.merge_upsert(update, dest, key_cols=["id"])
        assert len(merged) == 1
        assert merged[0]["amount"] == "999"

    def test_upsert_mixed(self, tmp_path: pathlib.Path) -> None:
        writer = DeltaWriter(tmp_path)
        dest = tmp_path / "data.csv"
        existing = [{"id": "1", "v": "a"}, {"id": "2", "v": "b"}]
        writer.write_transactions(existing, dest)
        updates = [{"id": "2", "v": "B"}, {"id": "3", "v": "c"}]
        merged = writer.merge_upsert(updates, dest, key_cols=["id"])
        assert len(merged) == 3

    def test_upsert_on_nonexistent_file(self, tmp_path: pathlib.Path) -> None:
        writer = DeltaWriter(tmp_path)
        dest = tmp_path / "new.csv"
        new_rows = [{"id": "1", "v": "hello"}]
        merged = writer.merge_upsert(new_rows, dest, key_cols=["id"])
        assert len(merged) == 1


class TestDeltaWriterPartitionBy:
    def test_creates_partitions(self, tmp_path: pathlib.Path) -> None:
        writer = DeltaWriter(tmp_path)
        df = [
            {"date": "2024-01-01", "category": "Food", "amount": "10"},
            {"date": "2024-01-01", "category": "Travel", "amount": "50"},
            {"date": "2024-01-02", "category": "Food", "amount": "20"},
        ]
        partitions = writer.partition_by(df, cols=["date"], base_dir=tmp_path / "parts")
        assert len(partitions) == 2

    def test_partition_sums_correct(self, tmp_path: pathlib.Path) -> None:
        writer = DeltaWriter(tmp_path)
        df = [{"cat": "A"}, {"cat": "B"}, {"cat": "A"}]
        parts = writer.partition_by(df, cols=["cat"])
        assert len(parts[("A",)]) == 2
        assert len(parts[("B",)]) == 1

    def test_partition_files_written(self, tmp_path: pathlib.Path) -> None:
        writer = DeltaWriter(tmp_path)
        df = [{"x": "1", "v": "a"}, {"x": "2", "v": "b"}]
        base = tmp_path / "partitioned"
        writer.partition_by(df, cols=["x"], base_dir=base)
        assert (base / "x=1" / "data.csv").exists()
        assert (base / "x=2" / "data.csv").exists()


class TestDeltaWriterValidateSchema:
    def _schema(self) -> TableSchema:
        return TableSchema(columns=(
            ColumnSchema("id", str, nullable=False),
            ColumnSchema("amount", float, nullable=False),
            ColumnSchema("note", str, nullable=True),
        ))

    def test_valid_data_passes(self) -> None:
        writer = DeltaWriter()
        df = [{"id": "1", "amount": 42.5, "note": "test"}]
        writer.validate_schema(df, self._schema())  # should not raise

    def test_missing_column_raises(self) -> None:
        writer = DeltaWriter()
        df = [{"id": "1"}]  # missing amount
        with pytest.raises(SchemaValidationError, match="missing columns"):
            writer.validate_schema(df, self._schema())

    def test_non_nullable_none_raises(self) -> None:
        writer = DeltaWriter()
        df = [{"id": None, "amount": 10.0, "note": None}]
        with pytest.raises(SchemaValidationError, match="non-nullable"):
            writer.validate_schema(df, self._schema())

    def test_nullable_none_passes(self) -> None:
        writer = DeltaWriter()
        df = [{"id": "1", "amount": 10.0, "note": None}]
        writer.validate_schema(df, self._schema())  # should not raise

    def test_wrong_type_raises(self) -> None:
        writer = DeltaWriter()
        df = [{"id": "1", "amount": "not_a_float", "note": None}]
        with pytest.raises(SchemaValidationError, match="cannot be coerced"):
            writer.validate_schema(df, self._schema())

    def test_empty_dataframe_passes(self) -> None:
        writer = DeltaWriter()
        writer.validate_schema([], self._schema())  # should not raise


# ===========================================================================
# PiiVault tests
# ===========================================================================

class TestPiiVaultEncryptDecrypt:
    def test_roundtrip(self) -> None:
        vault = PiiVault()
        key = PiiVault.generate_key()
        plaintext = "Hello, World!"
        ciphertext = vault.encrypt_field(plaintext, key)
        recovered = vault.decrypt_field(ciphertext, key)
        assert recovered == plaintext

    def test_ciphertext_differs_from_plaintext(self) -> None:
        vault = PiiVault()
        key = PiiVault.generate_key()
        ct = vault.encrypt_field("secret", key)
        assert ct != b"secret"
        assert b"secret" not in ct

    def test_different_keys_produce_different_ciphertext(self) -> None:
        vault = PiiVault()
        k1 = PiiVault.generate_key()
        k2 = PiiVault.generate_key()
        ct1 = vault.encrypt_field("data", k1)
        ct2 = vault.encrypt_field("data", k2)
        assert ct1 != ct2

    def test_same_plaintext_different_ciphertext_each_call(self) -> None:
        # Due to random IV, encrypting the same value twice gives different output
        vault = PiiVault()
        key = PiiVault.generate_key()
        ct1 = vault.encrypt_field("same", key)
        ct2 = vault.encrypt_field("same", key)
        assert ct1 != ct2

    def test_wrong_key_raises_on_decrypt(self) -> None:
        vault = PiiVault()
        k1 = PiiVault.generate_key()
        k2 = PiiVault.generate_key()
        ct = vault.encrypt_field("secret data", k1)
        with pytest.raises(ValueError):
            vault.decrypt_field(ct, k2)

    def test_tampered_ciphertext_raises(self) -> None:
        vault = PiiVault()
        key = PiiVault.generate_key()
        ct = bytearray(vault.encrypt_field("hello", key))
        ct[50] ^= 0xFF  # flip a byte in the ciphertext
        with pytest.raises(ValueError):
            vault.decrypt_field(bytes(ct), key)

    def test_wrong_key_length_raises(self) -> None:
        vault = PiiVault()
        with pytest.raises(ValueError, match="Key must be"):
            vault.encrypt_field("test", b"short_key")

    def test_decrypt_wrong_key_length_raises(self) -> None:
        vault = PiiVault()
        with pytest.raises(ValueError, match="Key must be"):
            vault.decrypt_field(b"\x00" * 80, b"bad")

    def test_unicode_roundtrip(self) -> None:
        vault = PiiVault()
        key = PiiVault.generate_key()
        text = "Héllo Wörld — こんにちは"
        ct = vault.encrypt_field(text, key)
        assert vault.decrypt_field(ct, key) == text

    def test_generate_key_is_32_bytes(self) -> None:
        key = PiiVault.generate_key()
        assert len(key) == 32

    def test_generate_key_is_random(self) -> None:
        k1 = PiiVault.generate_key()
        k2 = PiiVault.generate_key()
        assert k1 != k2


class TestPiiVaultTokenize:
    def test_valid_ssn(self) -> None:
        vault = PiiVault()
        token = vault.tokenize_ssn("123-45-6789")
        assert isinstance(token, str)
        assert len(token) == 64

    def test_deterministic(self) -> None:
        vault = PiiVault()
        t1 = vault.tokenize_ssn("123-45-6789")
        t2 = vault.tokenize_ssn("123-45-6789")
        assert t1 == t2

    def test_different_ssns_different_tokens(self) -> None:
        vault = PiiVault()
        t1 = vault.tokenize_ssn("123-45-6789")
        t2 = vault.tokenize_ssn("987-65-4321")
        assert t1 != t2

    def test_custom_key(self) -> None:
        vault = PiiVault()
        key = PiiVault.generate_key()
        t1 = vault.tokenize_ssn("123-45-6789", secret_key=key)
        t2 = vault.tokenize_ssn("123-45-6789", secret_key=key)
        assert t1 == t2

    def test_different_keys_different_tokens(self) -> None:
        vault = PiiVault()
        k1 = PiiVault.generate_key()
        k2 = PiiVault.generate_key()
        t1 = vault.tokenize_ssn("123-45-6789", secret_key=k1)
        t2 = vault.tokenize_ssn("123-45-6789", secret_key=k2)
        assert t1 != t2

    def test_invalid_ssn_raises(self) -> None:
        vault = PiiVault()
        with pytest.raises(ValueError, match="Invalid SSN"):
            vault.tokenize_ssn("123456789")

    def test_invalid_ssn_wrong_format_raises(self) -> None:
        vault = PiiVault()
        with pytest.raises(ValueError):
            vault.tokenize_ssn("abc-de-fghi")


class TestPiiVaultAuditLog:
    def test_audit_log_records_entry(self) -> None:
        vault = PiiVault()
        vault.audit_log("encrypt", "ssn", "user1")
        entries = vault.get_audit_log()
        assert len(entries) == 1
        assert entries[0].operation == "encrypt"
        assert entries[0].field_name == "ssn"
        assert entries[0].user == "user1"

    def test_multiple_entries(self) -> None:
        vault = PiiVault()
        vault.audit_log("encrypt", "ssn", "user1")
        vault.audit_log("decrypt", "email", "user2")
        vault.audit_log("tokenize", "ssn", "user1")
        assert len(vault.get_audit_log()) == 3

    def test_audit_written_to_file(self, tmp_path: pathlib.Path) -> None:
        log_path = tmp_path / "audit.log"
        vault = PiiVault(audit_log_path=str(log_path))
        vault.audit_log("encrypt", "name", "svc_worker")
        assert log_path.exists()
        content = log_path.read_text()
        assert "encrypt" in content
        assert "name" in content

    def test_audit_entry_has_timestamp(self) -> None:
        vault = PiiVault()
        vault.audit_log("tokenize", "ssn", "u")
        entry = vault.get_audit_log()[0]
        assert "T" in entry.timestamp  # ISO format contains T

    def test_audit_success_flag(self) -> None:
        vault = PiiVault()
        vault.audit_log("decrypt", "email", "u", success=False, details="Key mismatch")
        entry = vault.get_audit_log()[0]
        assert not entry.success
        assert entry.details == "Key mismatch"

    def test_get_audit_log_returns_copy(self) -> None:
        vault = PiiVault()
        vault.audit_log("op", "f", "u")
        log1 = vault.get_audit_log()
        log2 = vault.get_audit_log()
        assert log1 is not log2
