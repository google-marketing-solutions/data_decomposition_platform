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

"""Tests for utility functions in preprocessors.py."""

import dataclasses
import pathlib

from absl.testing import absltest
import numpy as np
import pandas as pd

from disaggregator import preprocessors

_TEST_BAYESIAN_DATA_INPUT = pd.DataFrame.from_dict(
    dict(
        shared_id=np.asarray([1, 2], dtype=np.int32),
        x_aggregates=np.asarray([30.0, 40.0], dtype=np.float32),
        x_values=np.asarray([18.0, 16.0], dtype=np.float32),
        y_aggregates=np.asarray([10.0, 20.0], dtype=np.float32),
    )
)
_TEST_BAYESIAN_MAPPING = preprocessors.BayesianColumnNameMapping(
    shared_id="shared_id",
    x_aggregates="x_aggregates",
    x_values="x_values",
    y_aggregates="y_aggregates",
)
_TEST_NEURAL_NETWORKS_DATA_INPUT = pd.DataFrame.from_dict(
    dict(
        shared_id=np.asarray([1, 2], dtype=np.int32),
        feature_1=np.asarray([30.0, 40.0], dtype=np.float32),
        feature_2=np.asarray([18.0, 16.0], dtype=np.float32),
        feature_3=np.asarray([10.0, 20.0], dtype=np.float32),
        y_aggregates=np.asarray([100.0, 200.0], dtype=np.float32),
    )
)
_TEST_NEURAL_NETWORKS_MAPPING = preprocessors.NeuralNetworksColumnNameMapping(
    shared_id="shared_id",
    features=("feature_1", "feature_2", "feature_3"),
    y_aggregates="y_aggregates",
)


class BaseDataInstanceCreatorTests(absltest.TestCase):

  def test_dataframe_dataframe_input(self):
    class MockChildClass(preprocessors.BaseDataInstanceCreator):

      def create_data_instance(self):
        return self._dataframe

    dataframe = pd.DataFrame.from_dict(dict(column_a=[1]))
    mock_class = MockChildClass(dataframe=dataframe, column_name_mapping="test")
    actual_output = mock_class.create_data_instance()
    pd.testing.assert_frame_equal(actual_output, dataframe)

  def test_dataframe_filepath_input(self):
    class MockChildClass(preprocessors.BaseDataInstanceCreator):

      def create_data_instance(self):
        return self._dataframe

    filepath = pathlib.Path(
        self.create_tempdir().full_path, "test_dataframe.csv"
    )
    dataframe = pd.DataFrame.from_dict(dict(column_a=[1]))
    dataframe.to_csv(str(filepath), index=False)
    mock_class = MockChildClass(dataframe=filepath, column_name_mapping="test")
    actual_output = mock_class.create_data_instance()
    pd.testing.assert_frame_equal(actual_output, dataframe)

  def test_dataframe_value_error(self):
    class MockChildClass(preprocessors.BaseDataInstanceCreator):

      def create_data_instance(self):
        pass

    with self.assertRaisesRegex(ValueError, "is not allowed. It must be"):
      MockChildClass(dataframe="path/to/file.json", column_name_mapping="test")


class BayesianColumnMappingTests(absltest.TestCase):

  def test_class_initialization(self):
    actual_output = preprocessors.BayesianColumnNameMapping(
        shared_id="shared_id",
        x_aggregates="x_aggregates",
        x_values="x_values",
        y_aggregates="y_aggregates",
    )
    expected_output = dict(
        shared_id="shared_id",
        x_aggregates="x_aggregates",
        x_values="x_values",
        y_aggregates="y_aggregates",
    )
    self.assertEqual(dataclasses.asdict(actual_output), expected_output)


class BayesianDataInstanceTests(absltest.TestCase):

  def test_class_initialization(self):
    data_instance = preprocessors.BayesianDataInstance(
        shared_id=_TEST_BAYESIAN_DATA_INPUT.shared_id,
        x_aggregates=_TEST_BAYESIAN_DATA_INPUT.x_aggregates,
        x_values=_TEST_BAYESIAN_DATA_INPUT.x_values,
        y_aggregates=_TEST_BAYESIAN_DATA_INPUT.y_aggregates,
    )
    actual_output = {
        k: v.tolist() for k, v in dataclasses.asdict(data_instance).items()
    }
    self.assertEqual(
        actual_output, _TEST_BAYESIAN_DATA_INPUT.to_dict(orient="list")
    )


class BayesianDataInstanceCreatorTests(absltest.TestCase):

  def test_create_data_instance(self):
    data_creator = preprocessors.BayesianDataInstanceCreator(
        dataframe=_TEST_BAYESIAN_DATA_INPUT,
        column_name_mapping=_TEST_BAYESIAN_MAPPING,
    )
    data_instance = data_creator.create_data_instance()
    actual_output = {
        k: v.tolist() for k, v in dataclasses.asdict(data_instance).items()
    }
    self.assertEqual(
        actual_output, _TEST_BAYESIAN_DATA_INPUT.to_dict(orient="list")
    )


class NeuralNetworksColumnNameMappingTests(absltest.TestCase):

  def test_class_initialization(self):
    actual_output = preprocessors.NeuralNetworksColumnNameMapping(
        shared_id="shared_id",
        features=("feature_1", "feature_2", "feature_3"),
        y_aggregates="y_aggregates",
    )
    expected_output = dict(
        shared_id="shared_id",
        features=("feature_1", "feature_2", "feature_3"),
        y_aggregates="y_aggregates",
    )
    self.assertEqual(dataclasses.asdict(actual_output), expected_output)


class NeuralNetworksDataInstanceTests(absltest.TestCase):

  def test_class_initialization(self):
    dataset_size = np.asarray(2)
    number_of_features = np.asarray(3)
    shared_id_aggregates = np.asarray([1.0, 1.0], dtype=np.float32)
    shared_id = _TEST_NEURAL_NETWORKS_DATA_INPUT.shared_id.values.astype(
        np.float32
    )
    features = _TEST_NEURAL_NETWORKS_DATA_INPUT[
        ["feature_1", "feature_2", "feature_3"]
    ].values
    y_aggregates = _TEST_NEURAL_NETWORKS_DATA_INPUT.y_aggregates.values
    y_aggregates_per_shared_id_aggregates = y_aggregates / shared_id_aggregates
    data_instance = preprocessors.NeuralNetworksDataInstance(
        shared_id=shared_id,
        features=features,
        y_aggregates=y_aggregates,
        y_aggregates_per_shared_id_aggregates=y_aggregates_per_shared_id_aggregates,
        shared_id_aggregates=shared_id_aggregates,
        dataset_size=dataset_size,
        number_of_features=number_of_features,
    )
    actual_output = {
        k: v.tolist() for k, v in dataclasses.asdict(data_instance).items()
    }
    expected_output = dict(
        shared_id=shared_id.tolist(),
        features=features.tolist(),
        y_aggregates=y_aggregates.tolist(),
        y_aggregates_per_shared_id_aggregates=y_aggregates_per_shared_id_aggregates.tolist(),
        shared_id_aggregates=shared_id_aggregates.tolist(),
        dataset_size=dataset_size.tolist(),
        number_of_features=number_of_features.tolist(),
    )
    self.assertEqual(actual_output, expected_output)


class NeuralNetworksDataInstanceCreatorTests(absltest.TestCase):

  def test_create_data_instance(self):
    data_creator = preprocessors.NeuralNetworksDataInstanceCreator(
        dataframe=_TEST_NEURAL_NETWORKS_DATA_INPUT,
        column_name_mapping=_TEST_NEURAL_NETWORKS_MAPPING,
    )
    data_instance = data_creator.create_data_instance()
    actual_output = {
        k: v.tolist() for k, v in dataclasses.asdict(data_instance).items()
    }
    shared_id_aggregates = np.asarray([1.0, 1.0], dtype=np.float32)
    y_aggregates = _TEST_NEURAL_NETWORKS_DATA_INPUT.y_aggregates.values
    y_aggregates_per_shared_id_aggregates = y_aggregates / shared_id_aggregates
    expected_output = dict(
        shared_id=_TEST_NEURAL_NETWORKS_DATA_INPUT.shared_id.values.astype(
            np.float32
        ).tolist(),
        features=_TEST_NEURAL_NETWORKS_DATA_INPUT[
            ["feature_1", "feature_2", "feature_3"]
        ].values.tolist(),
        y_aggregates=y_aggregates.tolist(),
        y_aggregates_per_shared_id_aggregates=y_aggregates_per_shared_id_aggregates.tolist(),
        shared_id_aggregates=shared_id_aggregates.tolist(),
        dataset_size=np.asarray(2),
        number_of_features=np.asarray(3),
    )
    self.assertEqual(actual_output, expected_output)


if __name__ == "__main__":
  absltest.main()
