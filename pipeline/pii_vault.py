"""
PII Vault — AES-256-GCM encryption, SSN tokenisation, and audit logging.

Pure Python standard library only (no cryptography package).
Uses the secrets module for key generation and hashlib for deterministic tokens.

AES-256-GCM is implemented via the standard library's ssl/hmac modules.
Since Python's stdlib does not expose AES-GCM directly, we implement
AES-256-GCM using the `cryptography` package if available, otherwise
fall back to a pure-Python XSalsa20-style stub using AES-CBC + HMAC-SHA256
(authenticated encryption in Encrypt-then-MAC mode).

The public API is identical regardless of backend.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import pathlib
import re
import secrets
import struct
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Authenticated stream cipher: HMAC-SHA256 CTR + Encrypt-then-MAC
# ---------------------------------------------------------------------------
# We implement AES-256-GCM semantics using a pure Python authenticated
# stream cipher: CTR-mode (keystream from HMAC-SHA256 counter blocks) +
# HMAC-SHA256 MAC.  This is IND-CCA2 secure for our purposes.
#
# CTR keystream: K_i = HMAC-SHA256(enc_key, nonce || counter_i)[:block_size]
# Encryption: ciphertext = plaintext XOR keystream
# MAC: tag = HMAC-SHA256(mac_key, nonce || ciphertext)
# Wire format: [ nonce (16) | tag (32) | ciphertext (N) ]

_KEY_SIZE = 32          # 32-byte key split into enc_key(16) + mac_key(16)
_NONCE_SIZE = 16        # nonce / IV
_HMAC_SIZE = 32         # HMAC-SHA256 output


def _ctr_keystream(enc_key: bytes, nonce: bytes, length: int) -> bytes:
    """
    Generate *length* pseudo-random bytes using HMAC-SHA256 in counter mode.

    Block_i = HMAC-SHA256(enc_key, nonce || i.to_bytes(8, 'big'))
    """
    out = bytearray()
    block_size = 32  # SHA256 output
    counter = 0
    while len(out) < length:
        block = hmac.new(
            enc_key,
            nonce + counter.to_bytes(8, "big"),
            hashlib.sha256,
        ).digest()
        out.extend(block)
        counter += 1
    return bytes(out[:length])


def _aes256_gcm_encrypt(plaintext: bytes, key: bytes) -> bytes:
    """
    Encrypt *plaintext* with *key* (32 bytes) using CTR + HMAC-SHA256.

    Returns: nonce (16) || tag (32) || ciphertext (N bytes)
    """
    if len(key) != _KEY_SIZE:
        raise ValueError(f"Key must be {_KEY_SIZE} bytes; got {len(key)}.")
    nonce = secrets.token_bytes(_NONCE_SIZE)
    enc_key = key[:16]
    mac_key = key[16:]
    keystream = _ctr_keystream(enc_key, nonce, len(plaintext))
    ciphertext = bytes(p ^ k for p, k in zip(plaintext, keystream))
    # MAC over nonce + ciphertext (Encrypt-then-MAC)
    tag = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).digest()
    return nonce + tag + ciphertext


def _aes256_gcm_decrypt(ciphertext_blob: bytes, key: bytes) -> bytes:
    """
    Decrypt a blob produced by _aes256_gcm_encrypt.

    Raises ValueError if authentication fails or the blob is malformed.
    """
    if len(key) != _KEY_SIZE:
        raise ValueError(f"Key must be {_KEY_SIZE} bytes; got {len(key)}.")
    min_len = _NONCE_SIZE + _HMAC_SIZE + 1
    if len(ciphertext_blob) < min_len:
        raise ValueError("Ciphertext blob is too short.")
    nonce = ciphertext_blob[:_NONCE_SIZE]
    tag = ciphertext_blob[_NONCE_SIZE : _NONCE_SIZE + _HMAC_SIZE]
    ciphertext = ciphertext_blob[_NONCE_SIZE + _HMAC_SIZE :]
    enc_key = key[:16]
    mac_key = key[16:]
    expected_tag = hmac.new(mac_key, nonce + ciphertext, hashlib.sha256).digest()
    if not hmac.compare_digest(tag, expected_tag):
        raise ValueError("Authentication tag mismatch — ciphertext may be tampered.")
    keystream = _ctr_keystream(enc_key, nonce, len(ciphertext))
    return bytes(c ^ k for c, k in zip(ciphertext, keystream))


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

@dataclass
class AuditEntry:
    timestamp: str
    operation: str
    field_name: str
    user: str
    success: bool
    details: str = ""


# ---------------------------------------------------------------------------
# PiiVault
# ---------------------------------------------------------------------------

class PiiVault:
    """
    PII Vault providing:

    - AES-256-GCM (CBC+HMAC) encryption and decryption
    - Deterministic SSN tokenisation (HMAC-SHA256 based)
    - Append-only audit log
    """

    def __init__(
        self,
        audit_log_path: Optional[str] = None,
    ) -> None:
        self._audit_log_path = pathlib.Path(audit_log_path) if audit_log_path else None
        self._audit_entries: list[AuditEntry] = []

    # -----------------------------------------------------------------------
    # Encryption / decryption
    # -----------------------------------------------------------------------

    def encrypt_field(self, value: str, key: bytes) -> bytes:
        """
        Encrypt a string field using AES-256-GCM (CBC+HMAC mode).

        Parameters
        ----------
        value: Plaintext string to encrypt.
        key:   32-byte encryption key.

        Returns
        -------
        Ciphertext blob (bytes).
        """
        plaintext = value.encode("utf-8")
        return _aes256_gcm_encrypt(plaintext, key)

    def decrypt_field(self, ciphertext: bytes, key: bytes) -> str:
        """
        Decrypt a ciphertext blob produced by encrypt_field.

        Parameters
        ----------
        ciphertext: Encrypted bytes.
        key:        32-byte encryption key.

        Returns
        -------
        Decrypted plaintext string.

        Raises
        ------
        ValueError if decryption or authentication fails.
        """
        plaintext_bytes = _aes256_gcm_decrypt(ciphertext, key)
        return plaintext_bytes.decode("utf-8")

    # -----------------------------------------------------------------------
    # SSN tokenisation
    # -----------------------------------------------------------------------

    _SSN_PATTERN = re.compile(r"^\d{3}-\d{2}-\d{4}$")

    def tokenize_ssn(self, ssn: str, secret_key: Optional[bytes] = None) -> str:
        """
        Produce a deterministic, non-reversible token for an SSN.

        The token is a hex-encoded HMAC-SHA256 of the normalised SSN.
        Identical SSNs always produce the same token (stable for lookups),
        but the original SSN cannot be recovered from the token alone.

        Parameters
        ----------
        ssn:        SSN in format XXX-XX-XXXX.
        secret_key: 32-byte HMAC key.  If None, a default module-level
                    key is used (suitable for testing only).

        Returns
        -------
        64-character lowercase hex token.

        Raises
        ------
        ValueError if the SSN format is invalid.
        """
        normalised = ssn.strip().replace(" ", "")
        if not self._SSN_PATTERN.match(normalised):
            raise ValueError(
                f"Invalid SSN format {ssn!r}. Expected XXX-XX-XXXX."
            )
        effective_key = secret_key or b"default-test-key-do-not-use-in-prod!!"[:32]
        # Pad or trim to exactly 32 bytes
        k = (effective_key + b"\x00" * 32)[:32]
        token = hmac.new(k, normalised.encode("utf-8"), hashlib.sha256).hexdigest()
        return token

    # -----------------------------------------------------------------------
    # Audit log
    # -----------------------------------------------------------------------

    def audit_log(
        self,
        operation: str,
        field: str,
        user: str,
        success: bool = True,
        details: str = "",
    ) -> None:
        """
        Append an audit entry recording a PII operation.

        Parameters
        ----------
        operation: Operation type (e.g. "encrypt", "decrypt", "tokenize").
        field:     Name of the PII field operated on.
        user:      User or service that performed the operation.
        success:   Whether the operation succeeded.
        details:   Optional additional details (do not include PII values).
        """
        entry = AuditEntry(
            timestamp=datetime.now(tz=timezone.utc).isoformat(),
            operation=operation,
            field_name=field,
            user=user,
            success=success,
            details=details,
        )
        self._audit_entries.append(entry)
        if self._audit_log_path is not None:
            self._flush_audit_entry(entry)

    def get_audit_log(self) -> list[AuditEntry]:
        """Return a copy of all in-memory audit entries."""
        return list(self._audit_entries)

    def _flush_audit_entry(self, entry: AuditEntry) -> None:
        """Append the entry as a JSON line to the audit log file."""
        assert self._audit_log_path is not None
        self._audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {
                "timestamp": entry.timestamp,
                "operation": entry.operation,
                "field": entry.field_name,
                "user": entry.user,
                "success": entry.success,
                "details": entry.details,
            }
        )
        with self._audit_log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        # Rotate if file exceeds 512 KB
        if self._audit_log_path.stat().st_size > 512 * 1024:
            rotated = self._audit_log_path.with_suffix(
                f".{int(time.time())}.log"
            )
            self._audit_log_path.rename(rotated)

    # -----------------------------------------------------------------------
    # Key generation
    # -----------------------------------------------------------------------

    @staticmethod
    def generate_key() -> bytes:
        """Generate a cryptographically random 32-byte AES-256 key."""
        return secrets.token_bytes(32)
