"""按外键查涉及行的只读 SQL 辅助。

主键值按文本传入，使用 unnest(...::text[])::列类型 让 PostgreSQL 按列实际类型转换，
避免 bigint / uuid / numeric 等类型与 text 参数比较时不隐式转换的问题。

行的同一性以库内规范形态为准：canonicalize_values 把传入写法归一成
PostgreSQL 输出形态（与 列::text 一致），删除集 / 命中关系全程使用规范形态，
同一行的不同写法（前导零、加号、UUID 大小写等）因此只算一行。
"""
from __future__ import annotations

from typing import Any

import psycopg
from psycopg.rows import dict_row, tuple_row

# 单次 IN 查询上限，防止极大删除集把查询撑爆；分块循环处理
_BATCH = 1000
# 超范围边只抽样展示命中行
SAMPLE_LIMIT = 200


def qident(parts: list[str]) -> str:
    """拼接限定标识符，双引号转义防注入（表名/列名来自目录而非用户直拼）。"""
    return ".".join('"' + p.replace('"', '""') + '"' for p in parts)


def _chunks(items: list[str], size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def canonicalize_values(conn: psycopg.Connection, schema: str, table: str,
                        column: str, values: list[str]) -> dict[str, str]:
    """把主键文本写法按列类型归一成 PostgreSQL 输出形态，返回 {原始文本: 规范文本}。

    归一在库内完成（raw::列类型::text），与 ``列::text`` 的文本形态天然一致：
    '007' / '+7' / ' 7 ' 对 bigint 都归一为 '7'，UUID 大写归一为小写等。
    同一行的不同写法因此收敛到同一键；无法按列类型转换的值抛 22P02（-> 400）。
    """
    type_name = _type_name(conn, schema, table, column)
    mapping: dict[str, str] = {}
    unique = list(dict.fromkeys(values))
    with conn.cursor(row_factory=tuple_row) as cur:
        for batch in _chunks(unique, _BATCH):
            sql = (
                f"SELECT raw, (raw::{type_name})::text "
                f"FROM (SELECT unnest(%s::text[]) AS raw) AS v"
            )
            cur.execute(sql, (batch,))
            for raw, canon in cur.fetchall():
                mapping[str(raw)] = str(canon)
    return mapping


def rows_exist(conn: psycopg.Connection, schema: str, table: str,
               pk_column: str, pk_values: list[str]) -> set[str]:
    """返回实际存在的主键值集合（规范文本形态，与 canonicalize_values 对应）。"""
    full = qident([schema, table])
    col = qident([pk_column])
    type_name = _type_name(conn, schema, table, pk_column)
    found: set[str] = set()
    with conn.cursor(row_factory=tuple_row) as cur:
        for batch in _chunks(pk_values, _BATCH):
            sql = (
                f"SELECT {col}::text FROM {full} "
                f"WHERE {col} = ANY(SELECT unnest(%s::text[])::{type_name})"
            )
            cur.execute(sql, (batch,))
            found.update(str(r[0]) for r in cur.fetchall())
    return found


def _type_name(conn: psycopg.Connection, schema: str, table: str, column: str) -> str:
    with conn.cursor(row_factory=tuple_row) as cur:
        cur.execute(
            """
            SELECT format_type(att.atttypid, att.atttypmod)
            FROM pg_attribute att
            JOIN pg_class cl ON cl.oid = att.attrelid
            JOIN pg_namespace ns ON ns.oid = cl.relnamespace
            WHERE ns.nspname = %s AND cl.relname = %s AND att.attname = %s
            """,
            (schema, table, column),
        )
        row = cur.fetchone()
    return row[0] if row else "text"


def referencing_pk_values(conn: psycopg.Connection, fk, target_values: list[str],
                          from_pk_column: str) -> set[str]:
    """单列外键、单列主键场景：取引用了目标值的引用表主键值。"""
    from_full = qident([fk.from_schema, fk.from_table])
    fk_col = qident([fk.from_columns[0]])
    pk_col = qident([from_pk_column])
    type_name = _type_name(conn, fk.to_schema, fk.to_table, fk.to_columns[0])
    result: set[str] = set()
    with conn.cursor(row_factory=tuple_row) as cur:
        for batch in _chunks(target_values, _BATCH):
            sql = (
                f"SELECT DISTINCT {pk_col}::text FROM {from_full} "
                f"WHERE {fk_col} = ANY(SELECT unnest(%s::text[])::{type_name})"
            )
            cur.execute(sql, (batch,))
            result.update(str(r[0]) for r in cur.fetchall())
    return result


def referencing_pk_pairs(conn: psycopg.Connection, fk, target_values: list[str],
                         from_pk_column: str) -> list[tuple[str, str]]:
    """NO ACTION 边专用：返回 (引用行主键文本, 该行实际引用的目标主键文本) 对。

    通过 JOIN 让数据库按自身的键匹配关系认定被引用行：外键列与主键列
    类型 / 文本写法不同时也取库内规范形态（目标主键列 ::text）。
    结果行数 == 实际命中的引用行数（被引用主键唯一，每行至多一条），
    内存占用随实际命中关系数线性增长。
    """
    from_full = qident([fk.from_schema, fk.from_table])
    to_full = qident([fk.to_schema, fk.to_table])
    fk_col = qident([fk.from_columns[0]])
    to_col = qident([fk.to_columns[0]])
    pk_col = qident([from_pk_column])
    type_name = _type_name(conn, fk.to_schema, fk.to_table, fk.to_columns[0])
    pairs: list[tuple[str, str]] = []
    with conn.cursor(row_factory=tuple_row) as cur:
        for batch in _chunks(target_values, _BATCH):
            sql = (
                f"SELECT t.{pk_col}::text, p.{to_col}::text "
                f"FROM {from_full} t "
                f"JOIN {to_full} p ON t.{fk_col} = p.{to_col} "
                f"WHERE t.{fk_col} = ANY(SELECT unnest(%s::text[])::{type_name})"
            )
            cur.execute(sql, (batch,))
            pairs.extend((str(a), str(b)) for a, b in cur.fetchall())
    return pairs


def matching_rows(conn: psycopg.Connection, schema: str, table: str,
                  match_columns: list[str], match_values: list[Any],
                  target_schema: str, target_table: str, target_columns: list[str]
                  ) -> tuple[list[dict], bool]:
    """超范围边的通用命中查询：按外键列匹配，抽样返回整行。

    match_values 与 target_columns 对齐（单列外键时长度为 1）。
    返回 (行列表(主键/行内容字符串化), 是否截断)。
    """
    full = qident([schema, table])
    with conn.cursor(row_factory=dict_row) as cur:
        type_names = [
            _type_name(conn, target_schema, target_table, c) for c in target_columns
        ]
        raw_cols = [f"c{i}_raw" for i in range(len(type_names))]
        cast_select = ", ".join(
            f"{raw}::{tn} AS c{i}" for i, (raw, tn) in enumerate(zip(raw_cols, type_names))
        )
        if len(type_names) == 1:
            vals_from = "unnest(%s::text[]) AS v(c0_raw)"
            params = [match_values]
        else:
            placeholders = ", ".join("%s::text[]" for _ in type_names)
            col_aliases = ", ".join(raw_cols)
            vals_from = f"unnest({placeholders}) AS v({col_aliases})"
            params = [
                [row[i] for row in match_values] for i in range(len(type_names))
            ]
        where = " AND ".join(
            f"t.{qident([mc])} = vals.c{idx}"
            for idx, mc in enumerate(match_columns)
        )
        sql = (
            f"WITH vals AS (SELECT {cast_select} FROM {vals_from})"
            f" SELECT t.* FROM {full} t CROSS JOIN vals "
            f"WHERE {where} LIMIT {SAMPLE_LIMIT + 1}"
        )
        cur.execute(sql, params)
        rows = cur.fetchall()

    truncated = len(rows) > SAMPLE_LIMIT
    rows = rows[:SAMPLE_LIMIT]
    normalized = [{k: _stringify(v) for k, v in row.items()} for row in rows]
    return normalized, truncated


def _stringify(value) -> str | int | float | bool | None:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
