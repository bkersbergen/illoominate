from .illoominate import (
    data_loo_polars,
    data_shapley_polars,
    train_and_evaluate_polars,
    train_and_evaluate_sbr_file,
)

import pandas as pd
import polars as pl


def _normalize_params(params: dict) -> dict:
    return {key: float(value) for key, value in params.items()}


def _require_columns(df: pd.DataFrame, columns: list[str], frame_name: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(
            f"{frame_name} is missing required columns {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def _as_polars(df: pd.DataFrame, columns: list[str]) -> pl.DataFrame:
    return pl.DataFrame(df.loc[:, columns])


def _create_index(frame: pl.DataFrame, source_col: str, index_col: str) -> pl.DataFrame:
    return (
        frame.select(source_col)
        .unique()
        .with_row_count(name=index_col)
        .with_columns(pl.col(index_col).cast(pl.Int64))
    )


def _empty_items_frame() -> pl.DataFrame:
    return pl.DataFrame({"item_id": []}, schema={"item_id": pl.Int64})


def _index_sustainable_items(
    sustainable_df: pd.DataFrame | None,
    item_index: pl.DataFrame,
) -> pl.DataFrame:
    if sustainable_df is None:
        return _empty_items_frame()

    sustainable_pl = _as_polars(sustainable_df, ["item_id"])
    return (
        sustainable_pl.join(item_index, on="item_id", how="inner")
        .drop("item_id")
        .rename({"item_idx": "item_id"})
    )


def _apply_mappings(
    frame: pl.DataFrame,
    mappings: list[tuple[str, pl.DataFrame]],
    rename_map: dict[str, str],
) -> pl.DataFrame:
    result = frame
    original_columns = [source for source, _ in mappings]

    for source_col, mapping in mappings:
        result = result.join(mapping, on=source_col)

    return (
        result.drop(original_columns)
        .rename(rename_map)
        .with_columns(pl.all().cast(pl.Int64))
    )


def _prepare_sbr_frames(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    sustainable_df: pd.DataFrame | None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    _require_columns(train_df, ["session_id", "item_id", "timestamp"], "train_df")
    _require_columns(validation_df, ["session_id", "item_id", "timestamp"], "validation_df")
    train_pl = _as_polars(train_df, ["session_id", "item_id", "timestamp"])
    validation_pl = _as_polars(validation_df, ["session_id", "item_id", "timestamp"])

    train_session_index = _create_index(train_pl, "session_id", "session_idx")
    validation_session_index = _create_index(validation_pl, "session_id", "session_idx")
    item_index = _create_index(train_pl, "item_id", "item_idx")

    sustainable_pl = _index_sustainable_items(sustainable_df, item_index)

    train_pl = _apply_mappings(
        train_pl,
        [("session_id", train_session_index), ("item_id", item_index)],
        {"session_idx": "session_id", "item_idx": "item_id"},
    )
    validation_pl = _apply_mappings(
        validation_pl,
        [("session_id", validation_session_index), ("item_id", item_index)],
        {"session_idx": "session_id", "item_idx": "item_id"},
    )

    return train_pl, validation_pl, sustainable_pl, train_session_index


def _prepare_nbr_frames(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    sustainable_df: pd.DataFrame | None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    _require_columns(train_df, ["user_id", "basket_id", "item_id"], "train_df")
    _require_columns(validation_df, ["user_id", "basket_id", "item_id"], "validation_df")
    train_pl = _as_polars(train_df, ["user_id", "basket_id", "item_id"])
    validation_pl = _as_polars(validation_df, ["user_id", "basket_id", "item_id"])

    user_index = _create_index(train_pl, "user_id", "user_idx")
    item_index = _create_index(train_pl, "item_id", "item_idx")
    train_basket_index = _create_index(train_pl, "basket_id", "basket_idx")
    validation_basket_index = _create_index(validation_pl, "basket_id", "basket_idx")

    sustainable_pl = _index_sustainable_items(sustainable_df, item_index)

    train_pl = _apply_mappings(
        train_pl,
        [
            ("user_id", user_index),
            ("item_id", item_index),
            ("basket_id", train_basket_index),
        ],
        {"user_idx": "user_id", "item_idx": "item_id", "basket_idx": "basket_id"},
    )
    validation_pl = _apply_mappings(
        validation_pl,
        [
            ("user_id", user_index),
            ("item_id", item_index),
            ("basket_id", validation_basket_index),
        ],
        {"user_idx": "user_id", "item_idx": "item_id", "basket_idx": "basket_id"},
    )

    return train_pl, validation_pl, sustainable_pl, user_index


def _join_result_ids(
    result_df: pl.DataFrame,
    index_df: pl.DataFrame,
    result_id_column: str,
    original_id_column: str,
    ) -> pd.DataFrame:
    index_column = f"{result_id_column}_idx"
    source_index_column = f"{original_id_column}_idx"

    return (
        result_df.rename({result_id_column: index_column})
        .join(index_df.rename({source_index_column: index_column}), on=index_column)
        .drop(index_column)
        .to_pandas()
    )


def _compute_data_values(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    model: str,
    metric: str,
    params: dict,
    sustainable_df: pd.DataFrame | None,
    rust_fn,
) -> pd.DataFrame:
    params = _normalize_params(params)

    if model == "vmis":
        train_pl, validation_pl, sustainable_pl, train_index = _prepare_sbr_frames(
            train_df, validation_df, sustainable_df
        )
        result_df = rust_fn(
            data=train_pl,
            validation=validation_pl,
            model=model,
            metric=metric,
            params=params,
            sustainable=sustainable_pl,
        )
        return _join_result_ids(result_df, train_index, "session_id", "session")

    if model == "tifu":
        train_pl, validation_pl, sustainable_pl, user_index = _prepare_nbr_frames(
            train_df, validation_df, sustainable_df
        )
        result_df = rust_fn(
            data=train_pl,
            validation=validation_pl,
            model=model,
            metric=metric,
            params=params,
            sustainable=sustainable_pl,
        )
        return _join_result_ids(result_df, user_index, "user_id", "user")

    raise ValueError(f"Unexpected value for 'model': {model}")


def data_shapley_values(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    model: str,
    metric: str,
    params: dict,
    sustainable_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    return _compute_data_values(
        train_df,
        validation_df,
        model,
        metric,
        params,
        sustainable_df,
        data_shapley_polars,
    )


def data_loo_values(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    model: str,
    metric: str,
    params: dict,
    sustainable_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    return _compute_data_values(
        train_df,
        validation_df,
        model,
        metric,
        params,
        sustainable_df,
        data_loo_polars,
    )


def train_and_evaluate_for_sbr(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    model: str,
    metric: str,
    params: dict,
    sustainable_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    params = _normalize_params(params)
    train_pl = _as_polars(train_df, ["session_id", "item_id", "timestamp"]).with_columns(
        pl.all().cast(pl.Int64)
    )
    validation_pl = _as_polars(
        validation_df, ["session_id", "item_id", "timestamp"]
    ).with_columns(pl.all().cast(pl.Int64))
    sustainable_pl = (
        _as_polars(sustainable_df, ["item_id"]).with_columns(pl.col("item_id").cast(pl.Int64))
        if sustainable_df is not None
        else _empty_items_frame()
    )
    return train_and_evaluate_polars(
        data=train_pl,
        validation=validation_pl,
        sustainable=sustainable_pl,
        model=model,
        metric=metric,
        params=params,
    ).to_pandas()


def train_and_evaluate_for_sbr_files(
    train_path: str,
    validation_path: str,
    metric: str,
    params: dict,
    sep: str = "\t",
    sustainable_path: str | None = None,
) -> pd.DataFrame:
    return train_and_evaluate_sbr_file(
        train_path=train_path,
        validation_path=validation_path,
        metric=metric,
        params=_normalize_params(params),
        separator=sep,
        sustainable_path=sustainable_path,
    ).to_pandas()


__all__ = [
    "data_shapley_polars",
    "data_loo_polars",
    "train_and_evaluate_polars",
    "data_shapley_values",
    "data_loo_values",
    "train_and_evaluate_for_sbr",
    "train_and_evaluate_for_sbr_files",
]
