from .illoominate import *
import polars as pl
import pandas as pd
from abc import ABC, abstractmethod

class DataValueComputationForSBR(ABC):
    """
    Abstract base class for value computaion (Shapley, Leave-One-Out)
    This class handles the shared logic for indexing columns and processing of data.
    """
    def __init__(self, model: str, metric: str, params: dict):
        self.model = model
        self.metric = metric
        self.params = params

    def _index_columns(self, train_df: pd.DataFrame, validation_df: pd.DataFrame) -> tuple:
        """
        Index columns (session_id and item_id) in both train and validation datasets.
        Item indices are shared, but session indices are computed separately.
        """
        # Convert to polars
        train_pl = pl.DataFrame(train_df[['session_id', 'item_id', 'timestamp']])
        validation_pl = pl.DataFrame(validation_df[['session_id', 'item_id', 'timestamp']])

        # Create indices for session_id's and item_id
        session_id_index_train = train_pl.select("session_id").unique().with_row_count(name="session_idx").with_columns(
            pl.col("session_idx").cast(pl.Int64)
        )
        session_id_index_validation = validation_pl.select("session_id").unique().with_row_count(name="session_idx").with_columns(
            pl.col("session_idx").cast(pl.Int64)
        )
        item_id_index_train = train_pl.select("item_id").unique().with_row_count(name="item_idx").with_columns(
            pl.col("item_idx").cast(pl.Int64)
        )

        # Transform the train and validation data
        train_pl = (
            train_pl
            .join(session_id_index_train, on="session_id")
            .join(item_id_index_train, on="item_id")
            .drop(["session_id", "item_id"])
            .rename({"session_idx": "session_id", "item_idx": "item_id"})
            .with_columns([pl.col(column).cast(pl.Int64) for column in train_pl.columns])
        )

        validation_pl = (
            validation_pl
            .join(session_id_index_validation, on="session_id")
            .join(item_id_index_train, on="item_id")
            .drop(["session_id", "item_id"])
            .rename({"session_idx": "session_id", "item_idx": "item_id"})
            .with_columns([pl.col(column).cast(pl.Int64) for column in validation_pl.columns])
        )

        return train_pl, validation_pl, session_id_index_train

    def compute(self, train_df: pd.DataFrame, validation_df: pd.DataFrame) -> pd.DataFrame:
        """
        Template Method defining the overall workflow.
        This method is called by subclasses to compute specific values (Shapley or Leave-One-Out).
        """
        # Index the columns (session_id and item_id) in both datasets
        train_pl, validation_pl, session_id_index_train = self._index_columns(train_df, validation_df)

        # Delegate specific data value computation to subclass
        values_polars = self._compute_values(train_pl, validation_pl)

        # Map back session IDs
        result_df = (
            values_polars
            .rename({"session_id": "session_idx"})
            .join(session_id_index_train, on="session_idx")
            .drop(["session_idx"])
            .to_pandas()
        )

        return result_df

    @abstractmethod
    def _compute_values(self, train_pl: pl.DataFrame, validation_pl: pl.DataFrame) -> pl.DataFrame:
        """
        Abstract method to be implemented by subclasses for specific value computations.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class ShapleyComputationForSBR(DataValueComputationForSBR):
    """
    Data Shapley Value computation subclass.
    """
    def _compute_values(self, train_pl: pl.DataFrame, validation_pl: pl.DataFrame) -> pl.DataFrame:
        return illoominate.data_shapley_polars(
            data=train_pl,
            validation=validation_pl,
            model=self.model,
            metric=self.metric,
            params=self.params
        )

class LeaveOneOutComputationForSBR(DataValueComputationForSBR):
    """
    Data Leave One Out Value computation subclass.
    """
    def _compute_values(self, train_pl: pl.DataFrame, validation_pl: pl.DataFrame) -> pl.DataFrame:
        return illoominate.data_loo_polars(
            data=train_pl,
            validation=validation_pl,
            model=self.model,
            metric=self.metric,
            params=self.params
        )



def data_shapley_values(train_df: pd.DataFrame, validation_df: pd.DataFrame, model: str,
                 metric:str, params: dict):

    computation = ShapleyComputationForSBR(model, metric, params)
    return computation.compute(train_df, validation_df)

def data_loo_values(train_df: pd.DataFrame, validation_df: pd.DataFrame, model: str,
                 metric:str, params: dict):

    computation = LeaveOneOutComputationForSBR(model, metric, params)
    return computation.compute(train_df, validation_df)
