from .illoominate import *
import polars as pl
import pandas as pd
from abc import ABC, abstractmethod

class DataValueComputationForNBR(ABC):
    """
    Abstract base class for value computation (Shapley, Leave-One-Out)
    This class handles the shared logic for indexing columns and processing of data.
    """
    def __init__(self, model: str, metric: str, params: dict):
        self.model = model
        self.metric = metric
        # Convert all values in params to float (f64 in Rust)
        self.params = {key: float(value) for key, value in params.items()}

    def _index_columns(self, train_df: pd.DataFrame, validation_df: pd.DataFrame, sustainable_df: pd.DataFrame = None) -> tuple:
        """
        Index columns (user_id, basket_id, item_id) in both train and validation datasets.
        Item indices are shared, but session indices are computed separately.
        """
        # Convert to polars
        train_pl = pl.DataFrame(train_df[['user_id', 'basket_id', 'item_id']])
        validation_pl = pl.DataFrame(validation_df[['user_id', 'basket_id', 'item_id']])

        # Index Users and Items from training data. Basket_ids get simply replaced by a number for Rust type compatibility.
        user_idx_index_train = train_pl.select("user_id").unique().with_row_count(name="user_idx").with_columns(
            pl.col("user_idx").cast(pl.Int64)
        )
        item_id_index_train = train_pl.select("item_id").unique().with_row_count(name="item_idx").with_columns(
            pl.col("item_idx").cast(pl.Int64)
        )
        basket_id_index_train = train_pl.select("basket_id").unique().with_row_count(name="basket_idx").with_columns(
            pl.col("basket_idx").cast(pl.Int64)
        )
        basket_id_index_validation = validation_pl.select("basket_id").unique().with_row_count(name="basket_idx").with_columns(
            pl.col("basket_idx").cast(pl.Int64)
        )

        # Transform the sustainable items
        sustainable_pl = self._index_sustainable_items(sustainable_df, item_id_index_train)

        # Transform the train and validation data
        train_pl = (
            train_pl
            .join(user_idx_index_train, on="user_id")
            .join(item_id_index_train, on="item_id")
            .join(basket_id_index_train, on="basket_id")
            .drop(["user_id", "item_id", "basket_id"])
            .rename({"user_idx": "user_id", "item_idx": "item_id", "basket_idx": "basket_id"})
            .with_columns([pl.col(column).cast(pl.Int64) for column in train_pl.columns])
        )

        validation_pl = (
            validation_pl
            .join(user_idx_index_train, on="user_id")
            .join(item_id_index_train, on="item_id")
            .join(basket_id_index_validation, on="basket_id")
            .drop(["user_id", "item_id", "basket_id"])
            .rename({"user_idx": "user_id", "item_idx": "item_id", "basket_idx": "basket_id"})
            .with_columns([pl.col(column).cast(pl.Int64) for column in validation_pl.columns])
        )

        return train_pl, validation_pl, sustainable_pl, user_idx_index_train

    def compute(self, train_df: pd.DataFrame, validation_df: pd.DataFrame, sustainable_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Template Method defining the overall workflow.
        This method is called by subclasses to compute specific values (Shapley or Leave-One-Out).
        """

        # Index the columns (user_id and item_id) in both datasets
        train_pl, validation_pl, sustainable_pl, user_id_index_train = self._index_columns(train_df, validation_df, sustainable_df)

        # Delegate specific data value computation to subclass
        result_df = self._compute_values(train_pl, validation_pl, sustainable_pl)

        # Map back IDs
        result_df = (
            result_df
            .rename({"user_id": "user_idx"})
            .join(user_id_index_train, on="user_idx")
            .drop(["user_idx"])
        )
        return result_df.to_pandas()

    def _index_sustainable_items(self, sustainable_df: pd.DataFrame, item_id_index: pl.DataFrame) -> pl.DataFrame:
        """
        Map `sustainable_df` item IDs to their corresponding indices using `item_id_index`.
        """
        if sustainable_df is None:
            return pl.DataFrame({"item_id": []})

        sustainable_pl = pl.DataFrame(sustainable_df[['item_id']])
        result = (sustainable_pl.join(item_id_index, on="item_id", how="inner")
                  .drop(["item_id"]).rename({"item_idx": "item_id"}))
        return result

    @abstractmethod
    def _compute_values(self, train_pl: pl.DataFrame, validation_pl: pl.DataFrame, sustainable_pl: pl.DataFrame) -> pl.DataFrame:
        """
        Abstract method to be implemented by subclasses for specific value computations.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class DataValueComputationForSBR(ABC):
    """
    Abstract base class for value computation (Shapley, Leave-One-Out)
    This class handles the shared logic for indexing columns and processing of data.
    """
    def __init__(self, model: str, metric: str, params: dict):
        self.model = model
        self.metric = metric
        # Convert all values in params to float (f64 in Rust)
        self.params = {key: float(value) for key, value in params.items()}

    def _index_columns(self, train_df: pd.DataFrame, validation_df: pd.DataFrame, sustainable_df: pd.DataFrame = None) -> tuple:
        """
        Index columns (session_id and item_id) in both train and validation datasets.
        Item indices are shared, but session indices are computed separately.
        """
        # Convert to Polars
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

        # Transform the sustainable items
        sustainable_pl = self._index_sustainable_items(sustainable_df, item_id_index_train)

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

        return train_pl, validation_pl, sustainable_pl, session_id_index_train

    def _index_sustainable_items(self, sustainable_df: pd.DataFrame, item_id_index: pl.DataFrame) -> pl.DataFrame:
        """
        Map `sustainable_df` item IDs to their corresponding indices using `item_id_index`.
        """
        if sustainable_df is None:
            return pl.DataFrame({"item_id": []})

        sustainable_pl = pl.DataFrame(sustainable_df[['item_id']])
        result = (sustainable_pl.join(item_id_index, on="item_id", how="inner")
                  .drop(["item_id"]).rename({"item_idx": "item_id"}))
        return result

    def compute(self, train_df: pd.DataFrame, validation_df: pd.DataFrame, sustainable_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Template Method defining the overall workflow.
        This method is called by subclasses to compute specific values (Shapley or Leave-One-Out).
        """
        # Index the columns (session_id and item_id) in both datasets
        train_pl, validation_pl, sustainable_pl, session_id_index_train = self._index_columns(train_df, validation_df, sustainable_df)

        # Delegate specific data value computation to subclass
        values_polars = self._compute_values(train_pl, validation_pl, sustainable_pl)

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
    def _compute_values(self, train_pl: pl.DataFrame, validation_pl: pl.DataFrame, sustainable_pl: pl.DataFrame) -> pl.DataFrame:
        """
        Abstract method to be implemented by subclasses for specific value computations.
        """
        raise NotImplementedError("Subclasses must implement this method.")


class ShapleyComputationForSBR(DataValueComputationForSBR):
    """
    Data Shapley Value computation subclass.
    """
    def _compute_values(self, train_pl: pl.DataFrame, validation_pl: pl.DataFrame, sustainable_pl: pl.DataFrame) -> pl.DataFrame:
        return illoominate.data_shapley_polars(
            data=train_pl,
            validation=validation_pl,
            model=self.model,
            metric=self.metric,
            params=self.params,
            sustainable=sustainable_pl,
        )

class ShapleyComputationForNBR(DataValueComputationForNBR):
    """
    Data Shapley Value computation subclass.
    """
    def _compute_values(self, train_pl: pl.DataFrame, validation_pl: pl.DataFrame, sustainable_pl: pl.DataFrame) -> pl.DataFrame:
        return illoominate.data_shapley_polars(
            data=train_pl,
            validation=validation_pl,
            model=self.model,
            metric=self.metric,
            params=self.params,
            sustainable=sustainable_pl,
        )

class LeaveOneOutComputationForSBR(DataValueComputationForSBR):
    """
    Data Leave One Out Value computation subclass.
    """
    def _compute_values(self, train_pl: pl.DataFrame, validation_pl: pl.DataFrame, sustainable_pl: pl.DataFrame) -> pl.DataFrame:
        return illoominate.data_loo_polars(
            data=train_pl,
            validation=validation_pl,
            model=self.model,
            metric=self.metric,
            params=self.params,
            sustainable=sustainable_pl,
        )

class LeaveOneOutComputationForNBR(DataValueComputationForNBR):
    """
    Data Leave One Out Value computation subclass.
    """
    def _compute_values(self, train_pl: pl.DataFrame, validation_pl: pl.DataFrame, sustainable_pl: pl.DataFrame) -> pl.DataFrame:
        return illoominate.data_loo_polars(
            data=train_pl,
            validation=validation_pl,
            model=self.model,
            metric=self.metric,
            params=self.params,
            sustainable=sustainable_pl,
        )

def data_shapley_values(train_df: pd.DataFrame, validation_df: pd.DataFrame, model: str,
                        metric:str, params: dict, sustainable_df: pd.DataFrame = None):

    if model == 'vmis':
        computation = ShapleyComputationForSBR(model, metric, params)
    elif model == 'tifu':
        computation = ShapleyComputationForNBR(model, metric, params)
    else:
        raise ValueError(f"Unexpected value for 'model': {params['model']}")

    return computation.compute(train_df, validation_df, sustainable_df)

def data_loo_values(train_df: pd.DataFrame, validation_df: pd.DataFrame, model: str,
                 metric:str, params: dict, sustainable_df: pd.DataFrame = None):

    if model == 'vmis':
        computation = LeaveOneOutComputationForSBR(model, metric, params)
    elif model == 'tifu':
        computation = LeaveOneOutComputationForNBR(model, metric, params)
    else:
        raise ValueError(f"Unexpected value for 'model': {params['model']}")

    return computation.compute(train_df, validation_df, sustainable_df)


class TrainAndEvaluateForSBR:
    """
    Train and evaluate a session-based recommender model.
    This class preprocesses data, trains a model, and evaluates its performance.
    """

    def __init__(self, model: str, metric: str, params: dict):
        self.model = model
        self.metric = metric
        self.params = {key: float(value) for key, value in params.items()}

    def _index_sustainable_items(self, sustainable_df: pd.DataFrame, item_id_index: pl.DataFrame) -> pl.DataFrame:
        """
        Map `sustainable_df` item IDs to their corresponding indices using `item_id_index`.
        """
        if sustainable_df is None:
            return pl.DataFrame({"item_id": []})

        sustainable_pl = pl.DataFrame(sustainable_df[['item_id']])
        result = (sustainable_pl.join(item_id_index, on="item_id", how="inner")
                  .drop(["item_id"]).rename({"item_idx": "item_id"}))
        return result

    def _index_columns(self, train_df: pd.DataFrame, validation_df: pd.DataFrame, sustainable_df: pd.DataFrame = None) -> tuple:
        """
        Index session_id and item_id in both train and validation datasets.
        Ensures compatibility with Rust backend.
        """
        train_pl = pl.DataFrame(train_df[['session_id', 'item_id', 'timestamp']])
        validation_pl = pl.DataFrame(validation_df[['session_id', 'item_id', 'timestamp']])

        # Create indices for session_id and item_id
        session_id_index_train = train_pl.select("session_id").unique().with_row_count(name="session_idx").with_columns(
            pl.col("session_idx").cast(pl.Int64)
        )
        session_id_index_validation = validation_pl.select("session_id").unique().with_row_count(name="session_idx").with_columns(
            pl.col("session_idx").cast(pl.Int64)
        )
        item_id_index_train = train_pl.select("item_id").unique().with_row_count(name="item_idx").with_columns(
            pl.col("item_idx").cast(pl.Int64)
        )

        # Transform the sustainable items
        sustainable_pl = self._index_sustainable_items(sustainable_df, item_id_index_train)

        # Transform train and validation data
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

        return train_pl, validation_pl, sustainable_pl, session_id_index_train

    def train_and_evaluate(self, train_df: pd.DataFrame, validation_df: pd.DataFrame, sustainable_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Train and evaluate the model on session-based recommendation data.
        """
        # Preprocess: Index session_id and item_id
        train_pl, validation_pl, sustainable_pl, session_id_index_train = self._index_columns(train_df, validation_df, sustainable_df)

        # Train the SBR model and evaluate
        evaluation_result = illoominate.train_and_evaluate_polars(
            data=train_pl,
            validation=validation_pl,
            sustainable=sustainable_pl,
            model=self.model,
            metric=self.metric,
            params=self.params,
        )

        return evaluation_result.to_pandas()

# Exposed API function
def train_and_evaluate_for_sbr(train_df: pd.DataFrame, validation_df: pd.DataFrame, model: str, metric: str, params: dict):
    """
    Exposed function to train and evaluate a session-based recommendation model.
    """
    trainer = TrainAndEvaluateForSBR(model, metric, params)
    return trainer.train_and_evaluate(train_df, validation_df)