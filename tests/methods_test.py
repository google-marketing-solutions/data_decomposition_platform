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

"""Tests for utility functions in methods.py."""

import dataclasses

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import tensorflow as tf

from disaggregator import methods
from disaggregator import preprocessors

jax.config.update("jax_threefry_partitionable", False)


_TEST_WARMUP_STEPS = 1
_TEST_SAMPLES = 1
_TEST_NUMBER_OF_CHAINS = 1
_TEST_BAYESIAN_DATA_INPUT = pd.DataFrame.from_dict(
    dict(
        shared_id=np.asarray([0, 1, 0, 1], dtype=np.int32),
        x_aggregates=np.asarray([30.0, 40.0, 30.0, 40.0], dtype=np.float32),
        x_values=np.asarray([25.0, 20.0, 10.0, 20.0], dtype=np.float32),
        y_aggregates=np.asarray([5.0, 10.0, 5.0, 10.0], dtype=np.float32),
    )
)
_TEST_BAYESIAN_MAPPING = preprocessors.BayesianColumnNameMapping(
    shared_id="shared_id",
    x_aggregates="x_aggregates",
    x_values="x_values",
    y_aggregates="y_aggregates",
)
_TEST_BAYESIAN_DATA_INSTANCE = preprocessors.BayesianDataInstance(
    shared_id=np.asarray([0, 1, 0, 1], dtype=np.int32),
    x_aggregates=np.asarray([30.0, 40.0, 30.0, 40.0], dtype=np.float32),
    x_values=np.asarray([25.0, 20.0, 10.0, 20.0], dtype=np.float32),
    y_aggregates=np.asarray([5.0, 10.0, 5.0, 10.0], dtype=np.float32),
)
_TEST_BAYESIAN_EXPECTED_OUTPUT = pd.DataFrame.from_dict(
    dict(
        shared_id=_TEST_BAYESIAN_DATA_INPUT.shared_id,
        x_aggregates=_TEST_BAYESIAN_DATA_INPUT.x_aggregates,
        x_values=_TEST_BAYESIAN_DATA_INPUT.x_values,
        y_aggregates=_TEST_BAYESIAN_DATA_INPUT.y_aggregates,
        disaggregated_y_values=np.asarray(
            [0.696105, 8.607791, 0.696105, 8.607791], dtype=np.float32
        ),
    )
)
_TEST_NEURAL_NETWORKS_DATA_INPUT = pd.DataFrame.from_dict(
    dict(
        shared_id=np.asarray([1, 2], dtype=np.int32),
        feature_1=np.asarray([30.0, 40.0], dtype=np.float32),
        feature_2=np.asarray([18.0, 16.0], dtype=np.float32),
        feature_3=np.asarray([10.0, 20.0], dtype=np.float32),
        y_aggregates=np.asarray([1.0, 1.0], dtype=np.float32),
    )
)
_TEST_NEURAL_NETWORKS_MAPPING = preprocessors.NeuralNetworksColumnNameMapping(
    shared_id="shared_id",
    features=("feature_1", "feature_2", "feature_3"),
    y_aggregates="y_aggregates",
)
_TEST_NEURAL_NETWORKS_DATA_INSTANCE = preprocessors.NeuralNetworksDataInstance(
    shared_id=np.asarray([1.0, 2.0], dtype=np.float32),
    features=np.asarray(
        [[30.0, 18.0, 10.0], [40.0, 16.0, 20.0]], dtype=np.float32
    ),
    y_aggregates=np.asarray([1.0, 1.0], dtype=np.float32),
    y_aggregates_per_shared_id_aggregates=np.asarray(
        [1.0, 1.0], dtype=np.float32
    ),
    shared_id_aggregates=np.asarray([1.0, 1.0], dtype=np.float32),
    dataset_size=np.asarray(2, dtype=np.int32),
    number_of_features=np.asarray(3, dtype=np.int32),
)
_TEST_NEURAL_NETWORK_PREDICTION_OUTCOME = dict(
    shared_id=_TEST_NEURAL_NETWORKS_DATA_INPUT.shared_id.astype(np.float32),
    feature_0=_TEST_NEURAL_NETWORKS_DATA_INPUT.feature_1,
    feature_1=_TEST_NEURAL_NETWORKS_DATA_INPUT.feature_2,
    feature_2=_TEST_NEURAL_NETWORKS_DATA_INPUT.feature_3,
    y_aggregates=_TEST_NEURAL_NETWORKS_DATA_INPUT.y_aggregates,
)


class BayesianModelTests(absltest.TestCase):

  def test_fit(self):
    model = methods.BayesianModel(
        warmup_steps=_TEST_WARMUP_STEPS,
        samples=_TEST_SAMPLES,
        number_of_chains=_TEST_NUMBER_OF_CHAINS,
    )
    dataset = preprocessors.BayesianDataInstanceCreator(
        dataframe=_TEST_BAYESIAN_DATA_INPUT,
        column_name_mapping=_TEST_BAYESIAN_MAPPING,
    ).create_data_instance()
    model.fit(dataset=dataset)
    actual_trace = {k: v.tolist() for k, v in model.trace.items()}
    expected_trace = dict(
        standard_deviation=[[1.5953984260559082]],
        weights=[[[0.13922090828418732, 0.8607791066169739]]],
        diverging=[[True]],
    )
    self.assertEqual(actual_trace, expected_trace)

  def test_predict(self):
    model = methods.BayesianModel(
        warmup_steps=_TEST_WARMUP_STEPS,
        samples=_TEST_SAMPLES,
        number_of_chains=_TEST_NUMBER_OF_CHAINS,
    )
    dataset = preprocessors.BayesianDataInstanceCreator(
        dataframe=_TEST_BAYESIAN_DATA_INPUT,
        column_name_mapping=_TEST_BAYESIAN_MAPPING,
    ).create_data_instance()
    model.fit(dataset=dataset)
    actual_output = model.predict(dataset=dataset)
    expected_output = jnp.array(
        [0.6961045, 8.607791, 0.6961045, 8.607791], dtype=jnp.float32
    )
    self.assertTrue(jnp.array_equal(actual_output, expected_output))

  def test_predict_attribute_error(self):
    model = methods.BayesianModel()
    dataset = preprocessors.BayesianDataInstanceCreator(
        dataframe=_TEST_BAYESIAN_DATA_INPUT,
        column_name_mapping=_TEST_BAYESIAN_MAPPING,
    ).create_data_instance()
    with self.assertRaisesRegex(AttributeError, "must be fit before"):
      model.predict(dataset=dataset)


class BayesianDisaggregatorUtilsFunctionsTests(absltest.TestCase):

  def test_load_neural_networks_dataset(self):
    data_instance = methods.load_bayesian_dataset(
        dataframe=_TEST_BAYESIAN_DATA_INPUT,
        column_name_mapping=_TEST_BAYESIAN_MAPPING,
    )
    actual_output = {
        k: v.tolist() for k, v in dataclasses.asdict(data_instance).items()
    }
    processed_data_instance = dataclasses.asdict(_TEST_BAYESIAN_DATA_INSTANCE)
    expected_output = {
        k: v.tolist() for k, v in processed_data_instance.items()
    }
    self.assertEqual(actual_output, expected_output)


class BayesianDisaggregatorTests(absltest.TestCase):

  def test_fit(self):
    model = methods.BayesianModel(
        warmup_steps=_TEST_WARMUP_STEPS,
        samples=_TEST_SAMPLES,
        number_of_chains=_TEST_NUMBER_OF_CHAINS,
    )
    bayesian_disaggregator = methods.BayesianDisaggregator(
        model=model,
    )
    bayesian_disaggregator.fit(
        dataframe=_TEST_BAYESIAN_DATA_INPUT,
        column_name_mapping=_TEST_BAYESIAN_MAPPING,
    )
    actual_trace = {
        k: v.tolist() for k, v in bayesian_disaggregator.model.trace.items()
    }
    expected_trace = dict(
        standard_deviation=[[1.5953984260559082]],
        weights=[[[0.13922090828418732, 0.8607791066169739]]],
        diverging=[[True]],
    )
    self.assertEqual(actual_trace, expected_trace)

  def test_transform(self):
    model = methods.BayesianModel(
        warmup_steps=_TEST_WARMUP_STEPS,
        samples=_TEST_SAMPLES,
        number_of_chains=_TEST_NUMBER_OF_CHAINS,
    )
    dataset = preprocessors.BayesianDataInstanceCreator(
        dataframe=_TEST_BAYESIAN_DATA_INPUT,
        column_name_mapping=_TEST_BAYESIAN_MAPPING,
    ).create_data_instance()
    model.fit(
        dataset=dataset,
    )
    bayesian_disaggregator = methods.BayesianDisaggregator(
        model=model,
    )
    actual_output = bayesian_disaggregator.transform(
        dataframe=_TEST_BAYESIAN_DATA_INPUT,
        column_name_mapping=_TEST_BAYESIAN_MAPPING,
    )
    pd.testing.assert_frame_equal(actual_output, _TEST_BAYESIAN_EXPECTED_OUTPUT)

  def test_fit_transform(self):
    model = methods.BayesianModel(
        warmup_steps=_TEST_WARMUP_STEPS,
        samples=_TEST_SAMPLES,
        number_of_chains=_TEST_NUMBER_OF_CHAINS,
    )
    bayesian_disaggregator = methods.BayesianDisaggregator(
        model=model,
    )
    actual_output = bayesian_disaggregator.fit_transform(
        dataframe=_TEST_BAYESIAN_DATA_INPUT,
        column_name_mapping=_TEST_BAYESIAN_MAPPING,
    )
    pd.testing.assert_frame_equal(actual_output, _TEST_BAYESIAN_EXPECTED_OUTPUT)


class NeuralNetworksModelTests(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="classification",
          task_type="classification",
          final_activation_function="sigmoid",
      ),
      dict(
          testcase_name="regression",
          task_type="regression",
          final_activation_function="linear",
      ),
  )
  def test_model(self, task_type, final_activation_function):
    model = methods.NeuralNetworksModel(task_type=task_type)
    actual_dense1_configuration = model.get_layer(index=0).get_config()
    actual_dense2_configuration = model.get_layer(index=1).get_config()
    expected_dense1_configuration = {
        "name": "dense1",
        "trainable": True,
        "dtype": "float32",
        "units": 5,
        "activation": "relu",
        "use_bias": False,
        "kernel_initializer": {
            "module": "keras.initializers",
            "class_name": "GlorotNormal",
            "config": {"seed": None},
            "registered_name": None,
        },
        "bias_initializer": {
            "module": "keras.initializers",
            "class_name": "Zeros",
            "config": {},
            "registered_name": None,
        },
        "kernel_regularizer": None,
        "bias_regularizer": None,
        "activity_regularizer": None,
        "kernel_constraint": None,
        "bias_constraint": None,
    }
    expected_dense2_configuration = {
        "name": "dense2",
        "trainable": True,
        "dtype": "float32",
        "units": 1,
        "activation": "",
        "use_bias": False,
        "kernel_initializer": {
            "module": "keras.initializers",
            "class_name": "GlorotNormal",
            "config": {"seed": None},
            "registered_name": None,
        },
        "bias_initializer": {
            "module": "keras.initializers",
            "class_name": "Zeros",
            "config": {},
            "registered_name": None,
        },
        "kernel_regularizer": None,
        "bias_regularizer": None,
        "activity_regularizer": None,
        "kernel_constraint": None,
        "bias_constraint": None,
    }
    expected_dense2_configuration["activation"] = final_activation_function
    self.assertEqual(
        [actual_dense1_configuration, actual_dense2_configuration],
        [expected_dense1_configuration, expected_dense2_configuration],
    )


class NeuralNetworksDisaggregatorUtilsFunctionsTests(parameterized.TestCase):

  def test_is_valid_task_type(self):
    with self.assertRaisesRegex(ValueError, "is not allowed"):
      methods.is_valid_task_type(task_type="clustering")

  def test_is_valid_data_instance(self):
    data_instance = preprocessors.NeuralNetworksDataInstance(
        shared_id=np.asarray([0, 0], dtype=np.int32),
        features=np.asarray([0.0, 0.0], dtype=np.float32),
        y_aggregates=np.asarray([0.0, 0.0], dtype=np.float32),
        y_aggregates_per_shared_id_aggregates=np.asarray(
            [-1.0, 2.0], dtype=np.float32
        ),
        shared_id_aggregates=np.asarray([0.0, 0.0], dtype=np.float32),
        dataset_size=np.asarray(0.0, dtype=np.float32),
        number_of_features=np.asarray(0.0, dtype=np.float32),
    )
    with self.assertRaisesRegex(ValueError, "The minimum and maximum values"):
      methods.is_valid_data_instance(
          task_type="classification", dataset=data_instance
      )

  def test_load_neural_networks_dataset(self):
    data_instance = methods.load_neural_networks_dataset(
        task_type="classification",
        dataframe=_TEST_NEURAL_NETWORKS_DATA_INPUT,
        column_name_mapping=_TEST_NEURAL_NETWORKS_MAPPING,
    )
    actual_output = {
        k: v.tolist() for k, v in dataclasses.asdict(data_instance).items()
    }
    processed_data_instance = dataclasses.asdict(
        _TEST_NEURAL_NETWORKS_DATA_INSTANCE
    )
    expected_output = {
        k: v.tolist() for k, v in processed_data_instance.items()
    }
    self.assertEqual(actual_output, expected_output)

  def test_assign_regression_value(self):
    shared_id_array = np.asarray(
        [[0.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.8, 0.2]], dtype=np.float32
    )
    actual_output = methods.assign_regression_value(
        shared_id_array=shared_id_array
    )
    expected_output = [[0.0, 0.800000011920929], [1.0, 0.20000000298023224]]
    self.assertEqual(actual_output.tolist(), expected_output)

  def test_assign_regression_value_assertion_error(self):
    shared_id_array = np.asarray(
        [[0.0, 1.0], [1.0, 1.0], [1.0, 1.0]], dtype=np.float32
    )
    with self.assertRaisesRegex(
        AssertionError, "The specified `shared_id_array` has an incompatible"
    ):
      methods.assign_regression_value(shared_id_array=shared_id_array)

  def test_process_regression_predictions(self):
    actual_output = methods.process_regression_predictions(
        dataset=_TEST_NEURAL_NETWORKS_DATA_INSTANCE,
        model_predictions=np.asarray([0.2, 0.8], dtype=np.float32),
    )
    np.testing.assert_almost_equal(
        actual_output, np.asarray([[1.0], [1.0]], dtype=np.float32)
    )

  def test_assign_class(self):
    shared_id_array = np.asarray(
        [[0.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.8, 0.2]], dtype=np.float32
    )
    actual_output = methods.assign_class(shared_id_array=shared_id_array)
    expected_output = np.asarray(
        [[0.0, 1.0], [1.0, 1.0], [1.0, 1.0], [0.8, 0.2], [1.0, 0.0]],
        dtype=np.float32,
    )
    np.testing.assert_almost_equal(actual_output, expected_output)

  def test_assign_class_assertion_error(self):
    shared_id_array = np.asarray(
        [[0.0, 1.0], [1.0, 1.0], [1.0, 1.0]], dtype=np.float32
    )
    with self.assertRaisesRegex(
        AssertionError, "The specified `shared_id_array` has an incompatible"
    ):
      methods.assign_class(shared_id_array=shared_id_array)

  def test_process_classification_predictions(self):
    actual_output = methods.process_classification_predictions(
        dataset=_TEST_NEURAL_NETWORKS_DATA_INSTANCE,
        model_predictions=np.asarray([0.2, 0.8], dtype=np.float32),
    )
    np.testing.assert_equal(
        actual_output, np.asarray([[1.0], [1.0]], dtype=np.float32)
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="classification",
          loss="mean_absolute_error",
          task_type="classification",
      ),
      dict(
          testcase_name="regression",
          loss="bce",
          task_type="regression",
      ),
  )
  def test_is_valid_loss(self, task_type, loss):
    with self.assertRaisesRegex(
        ValueError, f"loss type is not compatible with '{task_type}'"
    ):
      methods.is_valid_loss(task_type=task_type, loss=loss)

  def test_construct_output_dataframe(self):
    disaggregated_y_values = np.asarray([1, 1], dtype=np.int32)
    actual_output = methods.construct_output_dataframe(
        dataset=_TEST_NEURAL_NETWORKS_DATA_INSTANCE,
        disaggregated_y_values=disaggregated_y_values,
    )
    expected_output = _TEST_NEURAL_NETWORK_PREDICTION_OUTCOME.copy()
    expected_output["disaggregated_y_values"] = disaggregated_y_values
    pd.testing.assert_frame_equal(
        actual_output, pd.DataFrame.from_dict(expected_output)
    )


class NeuralNetworksDisaggregatorTests(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="classification",
          test_task_type="classification",
          test_loss="bce",
          expected_loss=[0.0],
      ),
      dict(
          testcase_name="regression",
          test_task_type="regression",
          test_loss="mean_absolute_error",
          expected_loss=[0.0],
      ),
  )
  def test_fit(self, test_task_type, test_loss, expected_loss):
    class TestModel(tf.keras.Model):

      def __init__(self, **kwargs):
        del kwargs
        super().__init__()

      def call(self, input_x):
        return tf.constant([1.0], dtype=tf.float32)

    nn_disaggregator = methods.NeuralNetworksDisaggregator(
        model=TestModel(),
        task_type=test_task_type,
    )
    nn_disaggregator.fit(
        dataframe=_TEST_NEURAL_NETWORKS_DATA_INPUT,
        column_name_mapping=_TEST_NEURAL_NETWORKS_MAPPING,
        compile_kwargs=dict(loss=test_loss, optimizer="adam"),
        fit_kwargs=dict(epochs=1),
    )
    np.testing.assert_almost_equal(
        nn_disaggregator.model.history.history["loss"], expected_loss
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="classification",
          test_task_type="classification",
          test_loss="bce",
          expected_predictions=np.asarray([1, 1], dtype=np.int32),
      ),
      dict(
          testcase_name="regression",
          test_task_type="regression",
          test_loss="mean_absolute_error",
          expected_predictions=np.asarray([1.0, 1.0], dtype=np.float32),
      ),
  )
  def test_transform(self, test_task_type, test_loss, expected_predictions):
    nn_disaggregator = methods.NeuralNetworksDisaggregator(
        model=methods.NeuralNetworksModel(task_type=test_task_type),
        task_type=test_task_type,
    )
    nn_disaggregator.fit(
        dataframe=_TEST_NEURAL_NETWORKS_DATA_INPUT,
        column_name_mapping=_TEST_NEURAL_NETWORKS_MAPPING,
        compile_kwargs=dict(loss=test_loss, optimizer="adam"),
        fit_kwargs=dict(epochs=1),
    )
    actual_output = nn_disaggregator.transform(
        dataframe=_TEST_NEURAL_NETWORKS_DATA_INPUT,
        column_name_mapping=_TEST_NEURAL_NETWORKS_MAPPING,
    )
    expected_output = _TEST_NEURAL_NETWORK_PREDICTION_OUTCOME
    expected_output["disaggregated_y_values"] = expected_predictions
    pd.testing.assert_frame_equal(
        actual_output, pd.DataFrame.from_dict(expected_output)
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="classification",
          test_task_type="classification",
          test_loss="bce",
          expected_predictions=np.asarray([1, 1], dtype=np.int32),
      ),
      dict(
          testcase_name="regression",
          test_task_type="regression",
          test_loss="mean_absolute_error",
          expected_predictions=np.asarray([1.0, 1.0], dtype=np.float32),
      ),
  )
  def test_fit_transform(self, test_task_type, test_loss, expected_predictions):
    nn_disaggregator = methods.NeuralNetworksDisaggregator(
        model=methods.NeuralNetworksModel(task_type=test_task_type),
        task_type=test_task_type,
    )
    actual_output = nn_disaggregator.fit_transform(
        dataframe=_TEST_NEURAL_NETWORKS_DATA_INPUT,
        column_name_mapping=_TEST_NEURAL_NETWORKS_MAPPING,
        compile_kwargs=dict(loss=test_loss, optimizer="adam"),
        fit_kwargs=dict(epochs=1),
    )
    expected_output = _TEST_NEURAL_NETWORK_PREDICTION_OUTCOME
    expected_output["disaggregated_y_values"] = expected_predictions
    pd.testing.assert_frame_equal(
        actual_output, pd.DataFrame.from_dict(expected_output)
    )


if __name__ == "__main__":
  absltest.main()
