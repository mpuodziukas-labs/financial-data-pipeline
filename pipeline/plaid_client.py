"""
Plaid API client with exponential backoff, proper error types,
and a clean interface for all required endpoints.

All HTTP is done via urllib (stdlib) — zero external dependencies.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional
from urllib.parse import urlencode


# ---------------------------------------------------------------------------
# Error hierarchy
# ---------------------------------------------------------------------------

class PlaidError(Exception):
    """Base class for all Plaid API errors."""

    def __init__(self, message: str, error_code: str = "", http_status: int = 0) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.http_status = http_status


class PlaidRateLimitError(PlaidError):
    """HTTP 429 — Too Many Requests."""


class PlaidAuthError(PlaidError):
    """HTTP 401 — Authentication failure."""


class PlaidItemError(PlaidError):
    """Plaid ITEM_ERROR — item requires re-authentication."""


class PlaidInstitutionError(PlaidError):
    """Plaid INSTITUTION_ERROR — institution is down or unavailable."""


class PlaidNetworkError(PlaidError):
    """Network-level error (timeout, DNS failure, etc.)."""


class PlaidInvalidRequestError(PlaidError):
    """HTTP 400 — Invalid request parameters."""


# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------

@dataclass
class LinkTokenCreateResponse:
    link_token: str
    expiration: str
    request_id: str


@dataclass
class ItemPublicTokenExchangeResponse:
    access_token: str
    item_id: str
    request_id: str


@dataclass
class Transaction:
    transaction_id: str
    account_id: str
    amount: float
    date: str
    name: str
    merchant_name: Optional[str]
    pending: bool
    category: list[str]
    iso_currency_code: Optional[str]


@dataclass
class TransactionsGetResponse:
    transactions: list[Transaction]
    total_transactions: int
    request_id: str
    next_cursor: Optional[str] = None


@dataclass
class Account:
    account_id: str
    name: str
    official_name: Optional[str]
    type: str
    subtype: Optional[str]
    balance_available: Optional[float]
    balance_current: float
    iso_currency_code: Optional[str]


@dataclass
class AccountsGetResponse:
    accounts: list[Account]
    request_id: str


@dataclass
class Identity:
    account_id: str
    owners: list[dict[str, Any]]


@dataclass
class IdentityGetResponse:
    accounts: list[Identity]
    request_id: str


# ---------------------------------------------------------------------------
# Retry configuration
# ---------------------------------------------------------------------------

@dataclass
class RetryConfig:
    max_retries: int = 3
    base_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    backoff_factor: float = 2.0
    retryable_status_codes: frozenset[int] = field(
        default_factory=lambda: frozenset({429, 500, 502, 503, 504})
    )


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class PlaidClient:
    """
    Plaid API client.

    Parameters
    ----------
    client_id:     Plaid client_id (from Plaid dashboard).
    secret:        Plaid secret key.
    base_url:      API base URL (defaults to sandbox).
    retry_config:  Exponential backoff configuration.
    """

    _DEFAULT_BASE_URL = "https://sandbox.plaid.com"

    def __init__(
        self,
        client_id: str,
        secret: str,
        base_url: str = _DEFAULT_BASE_URL,
        retry_config: Optional[RetryConfig] = None,
    ) -> None:
        self.client_id = client_id
        self.secret = secret
        self.base_url = base_url.rstrip("/")
        self.retry_config = retry_config or RetryConfig()

    # -----------------------------------------------------------------------
    # Internal HTTP helpers
    # -----------------------------------------------------------------------

    def _build_request(self, path: str, payload: dict[str, Any]) -> urllib.request.Request:
        url = f"{self.base_url}{path}"
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "PLAID-CLIENT-ID": self.client_id,
                "PLAID-SECRET": self.secret,
            },
        )
        return req

    def _parse_error(self, http_status: int, body: dict[str, Any]) -> PlaidError:
        error_type = body.get("error_type", "")
        error_code = body.get("error_code", "")
        message = body.get("error_message", f"HTTP {http_status}")

        if http_status == 429:
            return PlaidRateLimitError(message, error_code, http_status)
        if http_status == 401:
            return PlaidAuthError(message, error_code, http_status)
        if http_status == 400:
            return PlaidInvalidRequestError(message, error_code, http_status)
        if error_type == "ITEM_ERROR":
            return PlaidItemError(message, error_code, http_status)
        if error_type == "INSTITUTION_ERROR":
            return PlaidInstitutionError(message, error_code, http_status)
        return PlaidError(message, error_code, http_status)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """
        POST to *path* with *payload*, implementing exponential backoff.

        Raises the appropriate PlaidError subclass on failure.
        """
        cfg = self.retry_config
        attempt = 0
        last_error: Exception = PlaidError("Unknown error")

        while attempt <= cfg.max_retries:
            try:
                req = self._build_request(path, payload)
                with urllib.request.urlopen(req, timeout=30) as resp:
                    raw = resp.read()
                    return json.loads(raw)
            except urllib.error.HTTPError as exc:
                raw = exc.read()
                try:
                    body: dict[str, Any] = json.loads(raw)
                except Exception:
                    body = {}
                error = self._parse_error(exc.code, body)
                if exc.code in cfg.retryable_status_codes and attempt < cfg.max_retries:
                    delay = min(
                        cfg.base_delay_seconds * (cfg.backoff_factor ** attempt),
                        cfg.max_delay_seconds,
                    )
                    time.sleep(delay)
                    attempt += 1
                    last_error = error
                    continue
                raise error from exc
            except urllib.error.URLError as exc:
                network_err = PlaidNetworkError(str(exc), http_status=0)
                if attempt < cfg.max_retries:
                    delay = min(
                        cfg.base_delay_seconds * (cfg.backoff_factor ** attempt),
                        cfg.max_delay_seconds,
                    )
                    time.sleep(delay)
                    attempt += 1
                    last_error = network_err
                    continue
                raise network_err from exc
            except Exception as exc:
                raise PlaidError(str(exc)) from exc

        raise last_error

    # -----------------------------------------------------------------------
    # Public API methods
    # -----------------------------------------------------------------------

    def link_token_create(
        self,
        user_client_user_id: str,
        client_name: str,
        products: Optional[list[str]] = None,
        country_codes: Optional[list[str]] = None,
        language: str = "en",
    ) -> LinkTokenCreateResponse:
        """
        Create a Link token to initialise Plaid Link for a user.

        Parameters
        ----------
        user_client_user_id: Unique identifier for the user in your system.
        client_name:         Name of your application.
        products:            Plaid products to enable (default: ["auth", "transactions"]).
        country_codes:       Country codes (default: ["US"]).
        language:            Language for Link UI (default: "en").
        """
        payload: dict[str, Any] = {
            "client_id": self.client_id,
            "secret": self.secret,
            "user": {"client_user_id": user_client_user_id},
            "client_name": client_name,
            "products": products or ["auth", "transactions"],
            "country_codes": country_codes or ["US"],
            "language": language,
        }
        data = self._post("/link/token/create", payload)
        return LinkTokenCreateResponse(
            link_token=data["link_token"],
            expiration=data["expiration"],
            request_id=data["request_id"],
        )

    def item_public_token_exchange(
        self, public_token: str
    ) -> ItemPublicTokenExchangeResponse:
        """
        Exchange a public token (from Link) for a permanent access token.

        Parameters
        ----------
        public_token: Short-lived public token returned by Plaid Link.
        """
        payload: dict[str, Any] = {
            "client_id": self.client_id,
            "secret": self.secret,
            "public_token": public_token,
        }
        data = self._post("/item/public_token/exchange", payload)
        return ItemPublicTokenExchangeResponse(
            access_token=data["access_token"],
            item_id=data["item_id"],
            request_id=data["request_id"],
        )

    def transactions_get(
        self,
        access_token: str,
        start_date: str,
        end_date: str,
        cursor: Optional[str] = None,
        count: int = 500,
        offset: int = 0,
    ) -> TransactionsGetResponse:
        """
        Retrieve transactions for an item.

        Parameters
        ----------
        access_token: Access token from item_public_token_exchange.
        start_date:   Start date in YYYY-MM-DD format.
        end_date:     End date in YYYY-MM-DD format.
        cursor:       Pagination cursor from a previous response.
        count:        Number of transactions to fetch (max 500).
        offset:       Number of transactions to skip.
        """
        payload: dict[str, Any] = {
            "client_id": self.client_id,
            "secret": self.secret,
            "access_token": access_token,
            "start_date": start_date,
            "end_date": end_date,
            "options": {"count": count, "offset": offset},
        }
        if cursor:
            payload["cursor"] = cursor

        data = self._post("/transactions/get", payload)

        txns = [
            Transaction(
                transaction_id=t["transaction_id"],
                account_id=t["account_id"],
                amount=t["amount"],
                date=t["date"],
                name=t["name"],
                merchant_name=t.get("merchant_name"),
                pending=t.get("pending", False),
                category=t.get("category") or [],
                iso_currency_code=t.get("iso_currency_code"),
            )
            for t in data.get("transactions", [])
        ]
        return TransactionsGetResponse(
            transactions=txns,
            total_transactions=data.get("total_transactions", len(txns)),
            request_id=data["request_id"],
            next_cursor=data.get("next_cursor"),
        )

    def accounts_get(self, access_token: str) -> AccountsGetResponse:
        """
        Retrieve accounts associated with an item.

        Parameters
        ----------
        access_token: Access token from item_public_token_exchange.
        """
        payload: dict[str, Any] = {
            "client_id": self.client_id,
            "secret": self.secret,
            "access_token": access_token,
        }
        data = self._post("/accounts/get", payload)
        accounts = [
            Account(
                account_id=a["account_id"],
                name=a["name"],
                official_name=a.get("official_name"),
                type=a["type"],
                subtype=a.get("subtype"),
                balance_available=a.get("balances", {}).get("available"),
                balance_current=a.get("balances", {}).get("current", 0.0),
                iso_currency_code=a.get("balances", {}).get("iso_currency_code"),
            )
            for a in data.get("accounts", [])
        ]
        return AccountsGetResponse(accounts=accounts, request_id=data["request_id"])

    def identity_get(self, access_token: str) -> IdentityGetResponse:
        """
        Retrieve identity information for an item's accounts.

        Parameters
        ----------
        access_token: Access token from item_public_token_exchange.
        """
        payload: dict[str, Any] = {
            "client_id": self.client_id,
            "secret": self.secret,
            "access_token": access_token,
        }
        data = self._post("/identity/get", payload)
        accounts = [
            Identity(
                account_id=a["account_id"],
                owners=a.get("owners", []),
            )
            for a in data.get("accounts", [])
        ]
        return IdentityGetResponse(accounts=accounts, request_id=data["request_id"])
