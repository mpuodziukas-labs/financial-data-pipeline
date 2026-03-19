"""
Delta Writer — write, merge, partition, and validate tabular data.

Uses only Python stdlib (csv, json, pathlib, typing).
No pandas, no pyarrow — pure Python dict/list tables.

A "DataFrame" here is a list[dict[str, Any]] — a list of row dicts.
Column names are strings; all values are native Python types.
"""

from __future__ import annotations

import csv
import json
import os
import pathlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

Row = dict[str, Any]
DataFrame = list[Row]

# ---------------------------------------------------------------------------
# Schema type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ColumnSchema:
    """Expected type for a single column."""

    name: str
    dtype: type       # e.g. str, int, float, bool
    nullable: bool = True


@dataclass(frozen=True)
class TableSchema:
    columns: tuple[ColumnSchema, ...]

    @property
    def column_names(self) -> list[str]:
        return [c.name for c in self.columns]


# ---------------------------------------------------------------------------
# Validation error
# ---------------------------------------------------------------------------

class SchemaValidationError(Exception):
    """Raised when a DataFrame does not conform to a TableSchema."""


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: str | pathlib.Path) -> None:
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)


def _read_csv(path: str | pathlib.Path) -> DataFrame:
    p = pathlib.Path(path)
    if not p.exists():
        return []
    with p.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def _write_csv(df: DataFrame, path: str | pathlib.Path) -> None:
    p = pathlib.Path(path)
    _ensure_dir(p)
    if not df:
        p.write_text("")
        return
    fieldnames = list(df[0].keys())
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(df)


# ---------------------------------------------------------------------------
# DeltaWriter
# ---------------------------------------------------------------------------

class DeltaWriter:
    """
    Handles writing and merging tabular data to CSV-based delta storage.

    All paths are treated as CSV files.  The "partitioning" feature
    creates subdirectories mirroring partition column values.
    """

    def __init__(self, base_path: str | pathlib.Path = "/tmp/delta") -> None:
        self.base_path = pathlib.Path(base_path)

    # -----------------------------------------------------------------------
    # Write
    # -----------------------------------------------------------------------

    def write_transactions(
        self,
        df: DataFrame,
        path: str | pathlib.Path,
        mode: str = "overwrite",
    ) -> None:
        """
        Write *df* to *path*.

        Parameters
        ----------
        df:   List of row dicts.
        path: Destination file path (CSV).
        mode: "overwrite" (replace) or "append" (add rows).
        """
        p = pathlib.Path(path)
        if mode == "append" and p.exists():
            existing = _read_csv(p)
            df = existing + df
        _write_csv(df, p)

    # -----------------------------------------------------------------------
    # Merge / upsert
    # -----------------------------------------------------------------------

    def merge_upsert(
        self,
        new_df: DataFrame,
        existing_path: str | pathlib.Path,
        key_cols: list[str],
    ) -> DataFrame:
        """
        Perform an upsert: insert new rows or update existing rows
        where all *key_cols* match.

        Parameters
        ----------
        new_df:        Incoming rows.
        existing_path: Path to the existing CSV file.
        key_cols:      Column names that form the composite primary key.

        Returns
        -------
        Merged DataFrame (also written back to *existing_path*).
        """
        existing = _read_csv(existing_path)

        def _key(row: Row) -> tuple[Any, ...]:
            return tuple(row.get(k) for k in key_cols)

        existing_index: dict[tuple[Any, ...], int] = {
            _key(row): i for i, row in enumerate(existing)
        }

        for new_row in new_df:
            k = _key(new_row)
            if k in existing_index:
                existing[existing_index[k]] = new_row
            else:
                existing.append(new_row)
                existing_index[k] = len(existing) - 1

        _write_csv(existing, existing_path)
        return existing

    # -----------------------------------------------------------------------
    # Partitioning
    # -----------------------------------------------------------------------

    def partition_by(
        self,
        df: DataFrame,
        cols: list[str],
        base_dir: Optional[str | pathlib.Path] = None,
    ) -> dict[tuple[Any, ...], DataFrame]:
        """
        Split *df* into partition buckets based on the values of *cols*.

        Writes each partition to a CSV file under *base_dir* in a
        Hive-style directory structure:
          base_dir/col1=val1/col2=val2/data.csv

        Parameters
        ----------
        df:       Input DataFrame.
        cols:     Columns to partition by.
        base_dir: Root directory for partitioned output.

        Returns
        -------
        Dict mapping partition key tuple → partition DataFrame.
        """
        partitions: dict[tuple[Any, ...], DataFrame] = {}
        for row in df:
            key = tuple(row.get(c) for c in cols)
            partitions.setdefault(key, []).append(row)

        if base_dir is not None:
            root = pathlib.Path(base_dir)
            for key, part_df in partitions.items():
                # Build Hive-style path
                parts = [f"{c}={v}" for c, v in zip(cols, key)]
                part_path = root.joinpath(*parts) / "data.csv"
                _write_csv(part_df, part_path)

        return partitions

    # -----------------------------------------------------------------------
    # Schema validation
    # -----------------------------------------------------------------------

    def validate_schema(self, df: DataFrame, expected_schema: TableSchema) -> None:
        """
        Validate that every row in *df* conforms to *expected_schema*.

        Checks:
        - All required columns are present.
        - Non-nullable columns have non-None values.
        - Values are coercible to the declared dtype.

        Raises
        ------
        SchemaValidationError on the first violation found.
        """
        required_cols = set(expected_schema.column_names)

        for row_idx, row in enumerate(df):
            present_cols = set(row.keys())
            missing = required_cols - present_cols
            if missing:
                raise SchemaValidationError(
                    f"Row {row_idx}: missing columns {sorted(missing)}."
                )
            for col_schema in expected_schema.columns:
                value = row.get(col_schema.name)
                if value is None:
                    if not col_schema.nullable:
                        raise SchemaValidationError(
                            f"Row {row_idx}: column {col_schema.name!r} is non-nullable but got None."
                        )
                    continue
                # Allow string representations that are coercible
                try:
                    col_schema.dtype(value)
                except (ValueError, TypeError):
                    raise SchemaValidationError(
                        f"Row {row_idx}: column {col_schema.name!r} value {value!r} "
                        f"cannot be coerced to {col_schema.dtype.__name__}."
                    )


# Make Optional available for type hints used inside method bodies
from typing import Optional  # noqa: E402 (import at bottom to avoid circular)
