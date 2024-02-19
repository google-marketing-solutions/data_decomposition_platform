# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for the data loading and preprocessing operations."""

import abc
import dataclasses
import pathlib
from typing import Any

import jax.numpy as jnp
import numpy as np
import pandas as pd

_ALLOWED_DATAFRAME_SOURCE_EXTENSIONS = (".csv",)


class BaseDataInstanceCreator(metaclass=abc.ABCMeta):
  """Abstract parent class for all data instance creator classes."""

  def __init__(
      self, *, dataframe: str | pd.DataFrame, column_name_mapping: Any
  ) -> None:
    """Initializes the BaseDataInstanceCreator class.

    Args:
      dataframe: A dataframe or a path to a dataframe.
      column_name_mapping: A mapping of columns from the input dataset to the
        required columns. It is custom to any child class of
        BaseDataInstanceCreator.
    """
    self._dataframe = self._load_dataframe(dataframe=dataframe)
    self._column_name_mapping = column_name_mapping

  def _load_dataframe(self, *, dataframe: str | pd.DataFrame) -> pd.DataFrame:
    """Loads a Pandas DataFrame with an appropriate method.

    Args:
      dataframe: A dataframe or a path to a dataframe.

    Returns:
      A dataframe with the loaded dataset.

    Raises:
      ValueError: An error when the identified file extension is not supported
      and the file will not be able to be loaded.
    """
    if isinstance(dataframe, pd.DataFrame):
      return dataframe
    file_extension = pathlib.Path(dataframe).suffix
    if file_extension not in _ALLOWED_DATAFRAME_SOURCE_EXTENSIONS:
      raise ValueError(
          "File extension %s is not allowed. It must be one of %s."
          % (
              file_extension,
              ", ".join(_ALLOWED_DATAFRAME_SOURCE_EXTENSIONS),
          )
      )
    return pd.read_csv(dataframe)

  @abc.abstractmethod
  def create_data_instance(self) -> Any:
    """Creates a data instance from the provided data source."""


@dataclasses.dataclass(frozen=True)
class BayesianColumnNameMapping:
  """A mapping of columns from the input dataset to the required columns.

  Attributes:
    shared_id: A column in the input dataset representing "shared_id".
    x_aggregates: A column in the input dataset representing "x_aggregates".
    x_values: A column in the input dataset representing "x_values".
    y_aggregates: A column in the input dataset representing "y_aggregates".
  """

  shared_id: str
  x_aggregates: str
  x_values: str
  y_aggregates: str


@dataclasses.dataclass(frozen=True)
class BayesianDataInstance:
  """An instance of data with shared IDs for methods.BayesianDisaggregator.

  Attributes:
    shared_id: An array with an ID for each observation of shape (dataset_size,
      ).
    x_aggregates: An array with aggregated (on the assigned group level) X
      values for each observation of shape (dataset_size, ).
    x_values: An array with X values specific for each observation of shape
      (dataset_size, ).
    y_aggregates: An array with aggregated (on the assigned group level) Y
      values for each observation of shape (dataset_size, ). These are the
      values that later will be disaggregated.
  """

  shared_id: jnp.ndarray
  x_aggregates: jnp.ndarray
  x_values: jnp.ndarray
  y_aggregates: jnp.ndarray


class BayesianDataInstanceCreator(BaseDataInstanceCreator):
  """Loads a dataframe to later create a BayesianDataInstance.

  Example:

    mapping = BayesianColumnNameMapping(
        shared_id="column_a",
        x_aggregates="column_b",
        x_values="column_c",
        y_aggregates="column_d",
    )
    data_creator = BayesianDataInstanceCreator(
        dataframe="path/to/file.csv", column_name_mapping=mapping
    )
    bayesian_data_instance = data_creator.create_data_instance()
  """

  def __init__(
      self,
      dataframe: str,
      column_name_mapping: BayesianColumnNameMapping,
  ) -> None:
    """Initializes the BayesianDataInstanceCreator class.

    Args:
      dataframe: A dataframe or a path to a dataframe.
      column_name_mapping: A mapping of columns from the input dataset to the
        required columns.
    """
    super().__init__(
        dataframe=dataframe, column_name_mapping=column_name_mapping
    )
    self._shared_id = jnp.array(
        self._dataframe[self._column_name_mapping.shared_id], dtype=jnp.int32
    )
    self._x_aggregates = jnp.array(
        self._dataframe[self._column_name_mapping.x_aggregates],
        dtype=jnp.float32,
    )
    self._x_values = jnp.array(
        self._dataframe[self._column_name_mapping.x_values], dtype=jnp.float32
    )
    self._y_aggregates = jnp.array(
        self._dataframe[self._column_name_mapping.y_aggregates],
        dtype=jnp.float32,
    )

  def create_data_instance(self) -> BayesianDataInstance:
    """Returns a data instance from the provided data source."""
    return BayesianDataInstance(
        shared_id=self._shared_id,
        x_aggregates=self._x_aggregates,
        x_values=self._x_values,
        y_aggregates=self._y_aggregates,
    )


@dataclasses.dataclass(frozen=True)
class NeuralNetworksColumnNameMapping:
  """A mapping of columns from the input dataset to the required columns.

  Attributes:
    shared_id: A column in the input dataset representing "shared_id".
    features: Columns with features to be used in a Neural Networks model
      trainig and the following disaggregation process.
    y_aggregates: A column in the input dataset representing "y_aggregates".
  """

  shared_id: str
  features: tuple[str, ...]
  y_aggregates: str


@dataclasses.dataclass(frozen=True)
class NeuralNetworksDataInstance:
  """Instance of data with shared IDs for methods.NeuralNetworksDisaggregator.

  Attributes:
    shared_id: An array with an ID for each observation of shape (dataset_size,
      ).
    features: An array with features to be used in a Neural Networks model
      trainig and the following disaggregation process of shape (dataset_size,
      number_of_features).
    y_aggregates: An array with aggregated (on the assigned group level) Y
      values for each observation of shape (dataset_size, ). These are the
      values that later need to be disaggregated.
    y_aggregates_per_shared_id_aggregates: An array with the result of an
      element-wise divison between the y_aggregates and shared_id_aggregates
      arrays. It's of shape (dataset_size, ).
    shared_id_aggregates: An array with shared_id count for each observation.
      It's of shape (dataset_size, ).
    dataset_size: A number of observations in the dataset.
    number_of_features: A number of features to be used in a Neural Networks
      model trainig.
  """

  shared_id: np.ndarray
  features: np.ndarray
  y_aggregates: np.ndarray
  y_aggregates_per_shared_id_aggregates: np.ndarray
  shared_id_aggregates: np.ndarray
  dataset_size: np.ndarray
  number_of_features: np.ndarray


class NeuralNetworksDataInstanceCreator(BaseDataInstanceCreator):
  """Loads a dataframe to later create a NeuralNetworksDataInstance.

  Example:

    mapping = NeuralNetworksColumnNameMapping(
        shared_id="column_a",
        features=("column_b", "column_c"),
        y_aggregates="column_d",
    )
    data_creator = NeuralNetworksDataInstanceCreator(
        filepath="path/to/file.csv", column_name_mapping=mapping
    )
    neural_networks_data_instance = data_creator.create_data_instance()
  """

  def __init__(
      self,
      *,
      dataframe: str | pd.DataFrame,
      column_name_mapping: NeuralNetworksColumnNameMapping,
  ) -> None:
    """Initializes the NeuralNetworksDataInstanceCreator class.

    Args:
      dataframe: A dataframe or a path to a dataframe.
      column_name_mapping: A mapping of columns from the input dataset to the
        required columns.
    """
    super().__init__(
        dataframe=dataframe, column_name_mapping=column_name_mapping
    )
    self._shared_id = np.asarray(
        self._dataframe[self._column_name_mapping.shared_id], dtype=np.float32
    )
    self._features = np.asarray(
        self._dataframe[list(self._column_name_mapping.features)],
        dtype=np.float32,
    )
    self._y_aggregates = np.asarray(
        self._dataframe[self._column_name_mapping.y_aggregates],
        dtype=np.float32,
    )
    self._shared_id_aggregates = self._calculate_shared_id_aggregates(
        shared_id=self._shared_id
    )
    self._y_aggregates_per_shared_id_aggregates = (
        self._y_aggregates / self._shared_id_aggregates
    )
    self._dataset_size = np.asarray(np.size(self._shared_id))
    self._number_of_features = np.asarray(
        len(self._column_name_mapping.features)
    )

  def _calculate_shared_id_aggregates(
      self, *, shared_id: np.ndarray
  ) -> np.ndarray:
    """Calculates the shared_id_aggregates array.

    Args:
      shared_id: An array with an ID for each observation of shape
        (dataset_size, ).

    Returns:
      An array with a group size, grouped by shared_id, for each observation
      of shape (NeuralNetworksDataInstance.dataset_size, ).
    """
    unique_ids, unique_id_counts = np.unique(shared_id, return_counts=True)
    count_mapping = dict(zip(unique_ids, unique_id_counts))
    assign_count = np.vectorize(lambda x: count_mapping[x])
    return assign_count(shared_id)

  def create_data_instance(self) -> NeuralNetworksDataInstance:
    """Returns a data instance from the provided data source."""
    return NeuralNetworksDataInstance(
        shared_id=self._shared_id,
        features=self._features,
        y_aggregates=self._y_aggregates,
        y_aggregates_per_shared_id_aggregates=self._y_aggregates_per_shared_id_aggregates,
        shared_id_aggregates=self._shared_id_aggregates,
        dataset_size=self._dataset_size,
        number_of_features=self._number_of_features,
    )
