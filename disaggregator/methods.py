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

"""Utility functions for the disaggregation operations."""

import functools
import textwrap
from typing import Any, Final, Literal, Mapping

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer import hmc
from numpyro.infer import mcmc
import pandas as pd
import tensorflow as tf

from disaggregator import preprocessors

_REGRESSION: Final[str] = "regression"
_CLASSIFICATION: Final[str] = "classification"
_ALLOWED_TASK_TYPES = (_REGRESSION, _CLASSIFICATION)
_BCE: Final[str] = "bce"
_ALLOWED_LOSS_CLASSIFICATION = (_BCE,)
_MEAN_ABSOLUTE_ERROR: Final[str] = "mean_absolute_error"
_ALLOWED_LOSS_REGRESSION = (_MEAN_ABSOLUTE_ERROR,)


class BayesianModel:
  """Bayesian model to disaggregate values in the dataframe.

  Example:

    mapping = preprocessors.BayesianColumnNameMapping(
        shared_id="column_a",
        x_aggregates="column_b",
        x_values="column_c",
        y_aggregates="column_d",
    )

    y_aggregates are going to be disaggregated.

    bayesian_dataset_creator = preprocessors.BayesianDataInstanceCreator(
        dataframe="path/to/file.csv", column_name_mapping=mapping
    )
    bayesian_dataset = bayesian_dataset_creator.create_data_instance()
    bayesian_model = BayesianModel()
    disaggregated_y_aggregates = bayesian_model.fit_transform(
        dataset=bayesian_dataset
    )
  """

  def __init__(
      self,
      *,
      concentration: float = 0.1,
      rate: float = 1.0,
      warmup_steps: int = 1000,
      samples: int = 500,
      number_of_chains: int = 8,
      chain_method: str = "vectorized",
      seed: int = -1,
  ) -> None:
    """Initializes the BayesianModel class.

    Args:
      concentration: A number used to describe a concentration in the Gamma
        distribution.
      rate: A number used to describe a rate in the Gamma distribution.
      warmup_steps: A number of warm up step in Monte Carlo simulations.
      samples: A number of sample to produce in Monte Carlo simulations.
      number_of_chains: A number of chains to use in Monte Carlo simulations.
      chain_method: A chain method to use in Monte Carlo simulations.
      seed: A seed to use in simulations for repeatability.
    """
    self.trace = None
    self._concentration = concentration
    self._rate = rate
    self._warmup_steps = warmup_steps
    self._samples = samples
    self._number_of_chains = number_of_chains
    self._chain_method = chain_method
    self._seed = seed

  def _define_model(
      self,
      *,
      dataset: preprocessors.BayesianDataInstance,
      unique_shared_ids_count: int,
  ) -> Any:
    """Defines a NumPyro model to use in the simulation.

    Args:
      dataset: A data instance with the shared_id, x_aggregates, x_values and
        y_aggregates arrays. All arrays are of shape (input_dataframe_length, ).
      unique_shared_ids_count: A count of unique shared IDs in the input data
        later used to describe the concentration in the Dirichlet distribution.

    Returns:
      Predictions from the model.
    """
    weights = numpyro.sample(
        name="weights",
        fn=dist.Dirichlet(concentration=jnp.ones((unique_shared_ids_count,))),
    )
    mean = dataset.x_aggregates * weights[dataset.shared_id]
    standard_deviation = numpyro.sample(
        name="standard_deviation",
        fn=dist.Gamma(concentration=self._concentration, rate=self._rate),
    )
    return numpyro.sample(
        "predictions",
        dist.Normal(loc=mean, scale=standard_deviation),
        obs=dataset.x_values,
    )

  def _run_monte_carlo_markov_chain(
      self,
      *,
      prng_key: jax.Array,
      dataset: preprocessors.BayesianDataInstance,
      unique_shared_ids_count: int,
  ) -> Mapping[str, jnp.ndarray]:
    """Generates data samples with the Markov Chain Monte Carlo method.

    Args:
      prng_key: A PRNG key to use in the simulation.
      dataset: A data instance with the shared_id, x_aggregates, x_values and
        y_aggregates arrays. All arrays are of shape (input_dataframe_length, ).
      unique_shared_ids_count: A count of unique shared IDs in the input data
        later used to describe the concentration in the Dirichlet distribution.

    Returns:
      A data sample from the simulation.
    """
    partial_define_model = functools.partial(
        self._define_model,
        dataset=dataset,
        unique_shared_ids_count=unique_shared_ids_count,
    )
    nuts_kernel = hmc.NUTS(partial_define_model)
    monte_carlo_markov_chain = mcmc.MCMC(
        nuts_kernel,
        num_warmup=self._warmup_steps,
        num_samples=self._samples,
        num_chains=self._number_of_chains,
        chain_method=self._chain_method,
        progress_bar=False,
    )
    monte_carlo_markov_chain.run(prng_key)
    return {
        **monte_carlo_markov_chain.get_samples(),
        **monte_carlo_markov_chain.get_extra_fields(),
    }

  def fit(
      self,
      *,
      dataset: preprocessors.BayesianDataInstance,
  ) -> None:
    """Fits a model for the disaggregation process.

    After fitting the model will be assigned a "trace" property. It is a mapping
    between "standard_deviation", "weights" and "diverging" and their
    corresponding values. This mapping will be used in the disaggregation
    process.

    Args:
      dataset: A data instance with the shared_id, x_aggregates, x_values and
        y_aggregates arrays. All arrays are of shape (input_dataframe_length, ).
    """
    prng_key = jax.random.PRNGKey(self._seed)
    number_of_parallel_processes = jax.local_device_count()
    prng_keys = jax.random.split(prng_key, number_of_parallel_processes)
    unique_shared_ids = jnp.unique(dataset.shared_id).size
    pmapped_run_monte_carlo_markov_chain = jax.pmap(
        functools.partial(
            self._run_monte_carlo_markov_chain,
            dataset=dataset,
            unique_shared_ids_count=unique_shared_ids,
        )
    )
    self.trace = pmapped_run_monte_carlo_markov_chain(prng_key=prng_keys)

  def _is_fit(self) -> None:
    """Verifies if the model has been fit yet.

    Raises:
      AttributeError: An error if the model has not been fit yet.
    """
    if not self.trace:
      raise AttributeError("Model must be fit before making any predictions.")

  def predict(
      self,
      *,
      dataset: preprocessors.BayesianDataInstance,
  ) -> jnp.ndarray:
    """Disaggregates y_aggregates.

    Args:
      dataset: A data instance with the shared_id, x_aggregates, x_values and
        y_aggregates arrays. All arrays are of shape (input_dataframe_length, ).

    Returns:
      An array with disaggregated y_aggregates.

    Raises:
      AttributeError: An error if the model has not been fit yet.
    """
    self._is_fit()
    weights_hat = self.trace["weights"].mean(axis=(0, 1))
    return weights_hat[dataset.shared_id] * dataset.y_aggregates


def load_bayesian_dataset(
    *,
    dataframe: str | pd.DataFrame,
    column_name_mapping: preprocessors.BayesianColumnNameMapping,
) -> preprocessors.BayesianDataInstance:
  """Loads and creates a data instance from the provided data file.

  Args:
    dataframe: A dataframe or a path to a dataframe.
    column_name_mapping: A mapping of columns from the input dataframe to the
      required columns ("shared_id", "x_aggregates", "x_values",
      "y_aggregates").

  Returns:
    A data instance with the shared_id, x_aggregates, x_values and
    y_aggregates arrays. All arrays are of shape (input_dataframe_length, ).
  """
  shared_id_data_creator = preprocessors.BayesianDataInstanceCreator(
      dataframe=dataframe,
      column_name_mapping=column_name_mapping,
  )
  return shared_id_data_creator.create_data_instance()


class BayesianDisaggregator:
  """Uses a Bayesian model to disaggregate y_aggregates.

  Example:

    mapping = preprocessors.BayesianColumnNameMapping(
        shared_id="column_a",
        x_aggregates="column_b",
        x_values="column_c",
        y_aggregates="column_d",
    )
    bayesian_model = BayesianModel()

    y_aggregates are going to be disaggregated.

    bayesian_disaggregator = BayesianDisaggregator(model=bayesian_model)
    disaggregated_dataframe = bayesian_disaggregator.fit_transform(
        dataframe="path/to/file.csv", column_name_mapping=mapping
    )
  """

  def __init__(self, *, model: BayesianModel) -> None:
    """Initializes the BayesianDisaggregator class.

    Args:
      model: A class with an initialized Bayesian model to be used in fitting
        and disaggregation.
    """
    self.model = model

  def fit(
      self,
      *,
      dataframe: str | pd.DataFrame,
      column_name_mapping: preprocessors.BayesianColumnNameMapping,
      **kwargs,
  ) -> None:
    """Fits the model for the disaggregation process.

    After fitting the model will be assigned a "trace" property. It is a mapping
    between "standard_deviation", "weights" and "diverging" and their
    corresponding values. This mapping will be used in the disaggregation
    process.

    Args:
      dataframe: A dataframe or a path to a dataframe to be used for model
        fitting.
      column_name_mapping: A mapping of columns from the input dataframe to the
        required columns ("shared_id", "x_aggregates", "x_values",
        "y_aggregates").
      **kwargs: Keyword arguments necessary for the provided model fitting.
    """
    dataset = load_bayesian_dataset(
        dataframe=dataframe, column_name_mapping=column_name_mapping
    )
    self.model.fit(dataset=dataset, **kwargs)

  def transform(
      self,
      *,
      dataframe: str | pd.DataFrame,
      column_name_mapping: preprocessors.BayesianColumnNameMapping,
  ) -> pd.DataFrame:
    """Disaggregates y_aggregates.

    Args:
      dataframe: A dataframe or a path to a dataframe to be disaggregated.
      column_name_mapping: A mapping of columns from the input dataframe to the
        required columns ("shared_id", "x_aggregates", "x_values",
        "y_aggregates").

    Returns:
      A dataframe with shared_id, x_aggregates, x_values, y_aggregates
      and disaggregated_y_values.
    """
    dataset = load_bayesian_dataset(
        dataframe=dataframe, column_name_mapping=column_name_mapping
    )
    disaggregated_y_values = self.model.predict(dataset=dataset)
    return pd.DataFrame.from_dict(
        dict(
            shared_id=dataset.shared_id,
            x_aggregates=dataset.x_aggregates,
            x_values=dataset.x_values,
            y_aggregates=dataset.y_aggregates,
            disaggregated_y_values=disaggregated_y_values,
        )
    )

  def fit_transform(
      self,
      *,
      dataframe: str | pd.DataFrame,
      column_name_mapping: preprocessors.BayesianColumnNameMapping,
      **kwargs,
  ) -> pd.DataFrame:
    """Fits a model and disaggregates y_aggregates.

    Args:
      dataframe: A dataframe or a path to a dataframe to be disaggregated.
      column_name_mapping: A mapping of columns from the input dataframe to the
        required columns ("shared_id", "x_aggregates", "x_values",
        "y_aggregates").
      **kwargs: Keyword arguments necessary for the provided model fitting.

    Returns:
      A dataframe with shared_id, x_aggregates, x_values, y_aggregates
      and disaggregated_y_values.
    """
    self.fit(
        dataframe=dataframe,
        column_name_mapping=column_name_mapping,
        **kwargs,
    )
    return self.transform(
        dataframe=dataframe,
        column_name_mapping=column_name_mapping,
    )


class NeuralNetworksModel(tf.keras.Model):
  """Class with a Keras model.

  Attributes:
    dense1: The first out of two Dense layers in the model.
    dense2: The second out of two Dense layers in the model.
  """

  def __init__(
      self,
      *,
      task_type: Literal["classification", "regression"],
      units: tuple[int, int] = (5, 1),
      use_bias: bool = False,
      kernel_initializer: str = "glorot_normal",
      bias_initializer: str = "zeros",
      dense1_activation: str = "relu",
  ) -> None:
    """Initializes the Model class.

    Args:
      task_type: What type of task is the model going to be used for. Allowed:
        'classification' or 'regression'.
      units: A sequence with a number of units in each Dense layer.
      use_bias: Whether to use bias in Dense layers.
      kernel_initializer: A kernel initializer to use in Dense layers.
      bias_initializer: A bias initializer to use in Dense layers.
      dense1_activation: An activation function to use in the first Dense layer.
    """
    super().__init__()

    self.dense1 = tf.keras.layers.Dense(
        units=units[0],
        activation=dense1_activation,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        use_bias=use_bias,
        name="dense1",
    )

    self.dense2 = tf.keras.layers.Dense(
        units=units[1],
        activation="sigmoid" if task_type == _CLASSIFICATION else "linear",
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        use_bias=use_bias,
        name="dense2",
    )

  def call(self, input_x: tf.Tensor) -> tf.Tensor:
    """Passes an input through Model.

    Args:
      input_x: A data tensor of shape (batch_size, number_of_features).

    Returns:
      A tensor with predictions of shape (batch_size, ).
    """
    dense1_output = self.dense1(input_x)
    return self.dense2(dense1_output)


def is_valid_task_type(*, task_type: str) -> None:
  """Verifies if the provided task type is valid.

  Args:
    task_type: What type of task is the model going to be used for. Allowed:
      'classification' or 'regression'.

  Raises:
    ValueError: An error when task_type is neither 'regression' or
    'classification'.
  """
  if task_type not in _ALLOWED_TASK_TYPES:
    raise ValueError(
        f"Task type {task_type!r} is not allowed. "
        f"It must be one of {_ALLOWED_TASK_TYPES}."
    )


def is_valid_data_instance(
    *, task_type: str, dataset: preprocessors.NeuralNetworksDataInstance
) -> None:
  """Verifies if the loaded data instance is valid.

  Args:
    task_type: What type of task is the model going to be used for. Allowed:
      'classification' or 'regression'.
    dataset: A data instance with the shared_id, features, y_aggregates,
      y_aggregates_per_shared_id_aggregates, shared_id_aggregates, dataset_size
      and number_of_features arrays.

  Raises:
    ValueError: An error  when Y labels are not between 0 and 1 in a
    classification task.
  """
  if task_type == _CLASSIFICATION:
    min_value = np.min(dataset.y_aggregates_per_shared_id_aggregates)
    max_value = np.max(dataset.y_aggregates_per_shared_id_aggregates)

    if not 0 <= min_value <= 1 or not 0 <= max_value <= 1:
      raise ValueError(
          "The minimum and maximum values of aggregated Y labels per"
          " aggregated group array must be more or equal to 0 and less or"
          f" equal to 1. Received: min_value={min_value} and"
          f" max_value={max_value}."
      )


def load_neural_networks_dataset(
    *,
    task_type: str,
    dataframe: str | pd.DataFrame,
    column_name_mapping: preprocessors.NeuralNetworksColumnNameMapping,
) -> preprocessors.NeuralNetworksDataInstance:
  """Loads and creates a data instance from the provided data file.

  Args:
    task_type: What type of task is the model going to be used for. Allowed:
      'classification' or 'regression'.
    dataframe: A dataframe or a path to a dataframe.
    column_name_mapping: A mapping of columns from the input dataframe to the
      required columns ("shared_id", "features", "y_aggregates").

  Returns:
    Returns a data instance from the provided data source.
  """
  shared_id_data_creator = preprocessors.NeuralNetworksDataInstanceCreator(
      dataframe=dataframe,
      column_name_mapping=column_name_mapping,
  )
  dataset = shared_id_data_creator.create_data_instance()
  is_valid_data_instance(task_type=task_type, dataset=dataset)
  return dataset


def assign_regression_value(*, shared_id_array: np.ndarray) -> np.ndarray:
  """Assigns values to each observation within a unique shared ID group.

  Args:
    shared_id_array: An array with stacked and sorted values for each
      observation within that unique shared ID for the index, shared ID,
      y_aggregate and prediction.  The expected shape is: (4,
      number_of_observations_within_the_shared_id_group).

  Returns:
    The prediction index and value. The expected shape is:
      (number_of_observations_within_the_shared_id_group, 2).
  """
  assert shared_id_array.shape[0] == 4, textwrap.dedent(
      f"""
        The specified `shared_id_array` has an incompatible shape of
        {shared_id_array.shape[0]}"""
  ).strip()
  prediction_sum = np.full(
      (shared_id_array[3].shape[0], 1), shared_id_array[3].sum()
  )
  prediction_sum_by_y_aggregates = prediction_sum / np.expand_dims(
      shared_id_array[2], axis=1
  )
  predictions = (
      np.expand_dims(shared_id_array[3], axis=1)
      / prediction_sum_by_y_aggregates
  )
  return np.concatenate(
      [np.expand_dims(shared_id_array[0], axis=1), predictions], axis=1
  )


def process_regression_predictions(
    *,
    dataset: preprocessors.NeuralNetworksDataInstance,
    model_predictions: np.ndarray,
) -> np.ndarray:
  """Processes predictions in the regression task.

  Args:
    dataset: A data instance with the shared_id, features, y_aggregates,
      y_aggregates_per_shared_id_aggregates, shared_id_aggregates, dataset_size
      and number_of_features arrays.
    model_predictions: An array with the predicted disaggregated Y values of
      shape (NeuralNetworksDataInstance.dataset_size, ).

  Returns:
    An array with the final disaggregated Y values for regression of shape
    (NeuralNetworksDataInstance.dataset_size, ).
  """
  stacked_arrays = np.stack([
      np.arange(dataset.dataset_size),
      dataset.shared_id,
      dataset.y_aggregates,
      model_predictions,
  ])
  sorted_arrays_indices = stacked_arrays[1].argsort()
  sorted_arrays = np.take(stacked_arrays, sorted_arrays_indices, axis=1)
  _, unique_ids_index = np.unique(sorted_arrays[1], return_index=True)
  processed_unique_ids_index = np.sort(unique_ids_index[unique_ids_index != 0])
  split_arrays = np.split(sorted_arrays, processed_unique_ids_index, axis=1)
  regression_outcome = np.vstack(
      [
          assign_regression_value(shared_id_array=unique_id_group)
          for unique_id_group in split_arrays
      ],
  )
  sorted_prediction_array = regression_outcome[
      regression_outcome[:, 0].argsort()
  ]
  return np.expand_dims(sorted_prediction_array[:, 1], axis=1).astype(
      np.float32
  )


def assign_class(*, shared_id_array: np.ndarray) -> np.ndarray:
  """Assigns classes to each observation within a unique shared ID group.

  Args:
    shared_id_array: An array with stacked and sorted values for each
      observation the within that unique shared ID for the index, shared ID,
      y_aggregate and prediction.  The expected shape is: (4,
      number_of_observations_within_the_shared_id_group).

  Returns:
    The specified `shared_id_array` stacked with a row of classification
    results. The expected shape is:
      (5, number_of_observations_within_the_shared_id_group).
  """
  assert shared_id_array.shape[0] == 4, textwrap.dedent(
      f"""
        The specified `shared_id_array` has an incompatible shape of
        {shared_id_array.shape[0]}"""
  ).strip()
  shared_id_size = shared_id_array.shape[1]
  max_y_aggragates = np.unique(shared_id_array[2]).astype(np.int32).item()
  classification_predictions = np.zeros(shared_id_size)
  classification_predictions[0:max_y_aggragates] = 1
  classification_predictions = classification_predictions.reshape(
      1, shared_id_size
  )
  return np.concatenate([shared_id_array, classification_predictions])


def process_classification_predictions(
    *,
    dataset: preprocessors.NeuralNetworksDataInstance,
    model_predictions: np.ndarray,
) -> np.ndarray:
  """Processes predictions in the classification task.

  Args:
    dataset: A data instance with the shared_id, features, y_aggregates,
      y_aggregates_per_shared_id_aggregates, shared_id_aggregates, dataset_size
      and number_of_features arrays.
    model_predictions: An array with the predicted disaggregated Y values of
      shape (NeuralNetworksDataInstance.dataset_size, ).

  Returns:
    An array with the final disaggregated Y values for classification of shape
    (NeuralNetworksDataInstance.dataset_size, ).
  """
  stacked_arrays = np.stack([
      np.arange(dataset.dataset_size),
      dataset.shared_id,
      dataset.y_aggregates,
      model_predictions,
  ])
  sorted_arrays_indices = np.lexsort((stacked_arrays[3], stacked_arrays[1]))[
      ::-1
  ]
  sorted_arrays = np.take(stacked_arrays, sorted_arrays_indices, axis=1)
  _, unique_ids_index = np.unique(sorted_arrays[1], return_index=True)
  processed_unique_ids_index = np.sort(unique_ids_index[unique_ids_index != 0])
  split_arrays = np.split(sorted_arrays, processed_unique_ids_index, axis=1)
  classification_outcome = np.concatenate(
      [
          assign_class(shared_id_array=unique_id_group)
          for unique_id_group in split_arrays
      ],
      axis=1,
  )
  sorted_prediction_indices = np.lexsort(
      (classification_outcome[4], classification_outcome[0])
  )
  return np.expand_dims(
      np.take(classification_outcome[4], sorted_prediction_indices), axis=1
  ).astype(np.int32)


def is_valid_loss(*, task_type: str, loss: str) -> None:
  """Verifies if the provided loss and task_type are compatible.

  Args:
    task_type: What type of task is the model going to be used for. Allowed:
      'classification' or 'regression'.
    loss: A string with a loss name.

  Raises:
    ValueError: An error when the provided loss cannot be used with the
      specified task_type.
  """
  if task_type is _CLASSIFICATION and loss not in _ALLOWED_LOSS_CLASSIFICATION:
    raise ValueError(
        f"{loss!r} loss type is not compatible with {task_type!r}. "
        f"Compatible loss types are: {_ALLOWED_LOSS_CLASSIFICATION}."
    )
  if task_type is _REGRESSION and loss not in _ALLOWED_LOSS_REGRESSION:
    raise ValueError(
        f"{loss!r} loss type is not compatible with {task_type!r}. "
        f"Compatible loss types are: {_ALLOWED_LOSS_REGRESSION}."
    )


def construct_output_dataframe(
    *,
    dataset: preprocessors.NeuralNetworksDataInstance,
    disaggregated_y_values: np.ndarray,
) -> pd.DataFrame:
  """Constructs an output dataframe out of the source and disaggregated data.

  Args:
    dataset: A data instance with the shared_id, features, y_aggregates,
      y_aggregates_per_shared_id_aggregates, shared_id_aggregates, dataset_size
      and number_of_features arrays.
    disaggregated_y_values: An array with disaggregated and preprocessed values
      of shape (NeuralNetworksDataInstance.dataset_size, ).

  Returns:
    A dataframe with "shared_id", "features", "y_aggregates" and
    "disaggregated_y_values".
  """
  shared_id_dataframe = pd.DataFrame(dataset.shared_id, columns=["shared_id"])
  features_dataframe = pd.DataFrame(
      dataset.features,
      columns=["feature_" + str(i) for i in range(dataset.number_of_features)],
  )
  y_aggregates_dataframe = pd.DataFrame(
      dataset.y_aggregates, columns=["y_aggregates"]
  )
  disaggregated_y_values_dataframe = pd.DataFrame(
      disaggregated_y_values, columns=["disaggregated_y_values"]
  )
  return pd.concat(
      [
          shared_id_dataframe,
          features_dataframe,
          y_aggregates_dataframe,
          disaggregated_y_values_dataframe,
      ],
      axis=1,
  )


class NeuralNetworksDisaggregator:
  """Uses a Neural Networks model to disaggregate y_aggregates.

  Example:

    task_type = "classification"
    model = NeuralNetworksModel(task_type=task_type)
    mapping = preprocessors.NeuralNetworksColumnNameMapping(
        shared_id="column_a",
        features=("column_b", "column_c"),
        y_aggregates="column_d",
    )

    y_aggregates are going to be disaggregated.

    nn_disaggregator = NeuralNetworksDisaggregator(
        model=model,
        task_type=task_type,
    )
    disaggregated_dataframe = nn_disaggregator.fit_transform(
        dataframe="path/to/file.csv",
        column_name_mapping=mapping,
        compile_kwargs=dict(loss="bce", optimizer="adam"),
        fit_kwargs=dict(epochs=1),
    )

  Attributes:
    model: A Keras model used for fitting and disaggregation.
    task_type: What type of task is the model going to be used for, one of
      "classification" or "regression".
  """

  def __init__(
      self,
      *,
      model: tf.keras.Model,
      task_type: Literal["classification", "regression"],
      seed: int = 1,
  ) -> None:
    """Initializes the NeuralNetworksDisaggregator class.

    Args:
      model: A class with an initialized Keras model for training and
        disaggregation.
      task_type: What type of task is the model going to be used for. Allowed:
        'classification' or 'regression'.
      seed: A seed to use in Keras model training for repeatability.
    """
    self.model = model
    is_valid_task_type(task_type=task_type)
    self.task_type = task_type
    self._seed = seed

  def fit(
      self,
      *,
      dataframe: str | pd.DataFrame,
      column_name_mapping: preprocessors.NeuralNetworksColumnNameMapping,
      compile_kwargs: Mapping[str, Any],
      fit_kwargs: Mapping[str, Any],
  ) -> None:
    """Fits a Neural Networks model for the disaggregation process.

    Args:
      dataframe: A dataframe or a path to a dataframe to be used for model
        fitting.
      column_name_mapping: A mapping of columns from the input dataframe to the
        required columns ("shared_id", "features", "y_aggregates").
      compile_kwargs: Keyword arguments necessary for the provided model
        compilation.
      fit_kwargs: Keyword arguments necessary for the provided model fitting.
    """
    tf.keras.utils.set_random_seed(self._seed)
    is_valid_loss(task_type=self.task_type, loss=compile_kwargs["loss"])
    dataset = load_neural_networks_dataset(
        task_type=self.task_type,
        dataframe=dataframe,
        column_name_mapping=column_name_mapping,
    )
    self.model.compile(**compile_kwargs)
    self.model.fit(
        x=dataset.features,
        y=dataset.y_aggregates_per_shared_id_aggregates,
        **fit_kwargs,
    )

  def transform(
      self,
      *,
      dataframe: str | pd.DataFrame,
      column_name_mapping: preprocessors.NeuralNetworksColumnNameMapping,
  ) -> pd.DataFrame:
    """Disaggregates y_aggregates.

    Args:
      dataframe: A dataframe or a path to a dataframe to be disaggregated.
      column_name_mapping: A mapping of columns from the input dataframe to the
        required columns ("shared_id", "features", "y_aggregates").

    Returns:
      A dataframe with "shared_id", "features", "y_aggregates" and
      "disaggregated_y_values".
    """
    dataset = load_neural_networks_dataset(
        task_type=self.task_type,
        dataframe=dataframe,
        column_name_mapping=column_name_mapping,
    )
    raw_model_predictions = np.squeeze(self.model.predict(x=dataset.features))
    if self.task_type == _CLASSIFICATION:
      disaggregated_y_values = process_classification_predictions(
          dataset=dataset,
          model_predictions=raw_model_predictions,
      )
    else:
      disaggregated_y_values = process_regression_predictions(
          dataset=dataset, model_predictions=raw_model_predictions
      )
    return construct_output_dataframe(
        dataset=dataset, disaggregated_y_values=disaggregated_y_values
    )

  def fit_transform(
      self,
      *,
      dataframe: str | pd.DataFrame,
      column_name_mapping: preprocessors.NeuralNetworksColumnNameMapping,
      compile_kwargs: Mapping[str, Any],
      fit_kwargs: Mapping[str, Any],
  ) -> pd.DataFrame:
    """Fits a model and disaggregates y_aggregates.

    If dataframe and column_name_mapping aren't provided the model will
    disaggregate the dataset used for fitting the model.

    Args:
      dataframe: A dataframe or a path to a dataframe to be disaggregated.
      column_name_mapping: A mapping of columns from the input dataframe to the
        required columns ("shared_id", "features", "y_aggregates").
      compile_kwargs: Keyword arguments necessary for the provided model
        compilation.
      fit_kwargs: Keyword arguments necessary for the provided model fitting.

    Returns:
      A dataframe with "shared_id", "features", "y_aggregates" and
      "disaggregated_y_values".
    """
    self.fit(
        dataframe=dataframe,
        column_name_mapping=column_name_mapping,
        compile_kwargs=compile_kwargs,
        fit_kwargs=fit_kwargs,
    )
    return self.transform(
        dataframe=dataframe,
        column_name_mapping=column_name_mapping,
    )
