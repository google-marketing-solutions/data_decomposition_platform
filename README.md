# gTech Ads - Data Decomposition Package
### Data Decomposition Package is an open-source Python package to decompose aggregated observations down to a single observation level.
##### This is not an official Google product.

[Introduction](#introduction) •
[Decomposition process](#predictions)

# Introduction

In a privacy-first world, an increasing volume of data is be collected on an aggregated level which means that certain data points will not contain information on a single observational level. Instead these values are aggregated by a shared id where shared id could be a purchase date, a purchase time window, and others.

This level of data aggregation can be a major challenge when working with data pipelines and projects expecting data structures oriented on an individual-level observation. The single observation levels are representative of the aggregated whole, allowing for privacy-first methods to be used with existing data pipelines and methods which otherwise may break with new privacy-first, aggregated upstream methods.

gTech Ads Data Decomposition Platform is an open-source Python package to decompose aggregated observations down to a statistically representative single observation level.

The package implements 2 methods: 1) Bayesian Disaggregator and 2) Neural Networks Disaggregator. The former allows for decomposing continuous values, while the latter can handle both the continuous and categorical values. Both methods use a similar API.

# Decomposition process with Disaggregator methods

First, the input data needs to be either a comma-separated CSV file or a Pandas DataFrame.

```
dataframe = pd.read_csv(“path/to/your/file.csv”)
```

## Decomposition with Bayesian Disaggregator method

### Numerical aggregated data

The default Bayesian Disaggregator is compatible with only one input feature. It requires the dataframe to have a column with shared id (e.g. date, but represented as an integer), a single feature X (e.g. items in a basket), a feature X aggregated by shared id and a numerical aggregated value Y to be later decomposed in the process.

This method simply estimates a weight to distribute aggregated value Y into value Y, given learned weights from aggregated feature X to feature X. It assumes the weighting is the same between X and Y and that each group always has the same number of items.

*Example 1. Sample dataframe compatible with the Bayesian Disaggregator (numerical aggregated data).*

|**shared_id**| **feature_X** | **aggregated_feature_X**| **aggregated_value_Y** |
|:-----------------:|:-----------:|:--------:|:-------------------------:|
| 0                 | 1      | 3     | 10.5                       |
| 0                 | 2      | 3    | 10.5                      |
| 1                 | 5      | 9    | 16.8                      |
| 1                 | 4      | 9    | 16.8                      |

The following step is to provide a mapping between `shared_id`, `x_aggregates`, `x_values` and `y_aggregates` and the respective column names in the input dataframe.

```
column_name_mapping = preprocessors.BayesianColumnNameMapping(
    shared_id="shared_id",
    x_aggregates="aggregated_feature_X",
    x_values="feature_X",
    y_aggregates="aggregated_value_Y",
)
```

Next, you initialize the Bayesian model that you wish to use in the decomposition process. You can replace the default model with a customized one as long as it’s compatible with the `BayesianDisaggregator` class.

```
bayesian_model = methods.BayesianModel()
```

The final steps are to initialize the `BayesianDisaggregator` class and decompose the input dataframe.

```
bayesian_disaggregator = methods.BayesianDisaggregator(model=bayesian_model)
disaggregated_dataframe = bayesian_disaggregator.fit_transform(
    dataframe=dataframe, column_name_mapping=column_name_mapping
)
```

*Example 2. Sample output from the Bayesian Disaggregator  (numerical aggregated data).*

|**shared_id**| **x_aggregates** | **x_values**| **y_aggregates** | **disaggregated_y_values** |
|:-----------------:|:-----------:|:--------:|:-------------------------:|:--------:|
| 0                 | 3      | 1     | 10.5                       | 5.0      |
| 0                 | 3      | 2    | 10.5                      |5.5      |
| 1                 | 9      | 5    | 16.8                      |6.8      |
| 1                 | 9      | 4    | 16.8                      |10.0  |


## Decomposition with Neural Networks Disaggregator method

### Numerical aggregated data

The default Neural Networks Disaggregator is compatible with multiple input features.
It requires the dataframe to have a column with shared id (e.g. date, but represented as an integer), one or more features (e.g. items in a basket, browser type, device type etc.) and numerical aggregated value Y to be later decomposed in the process.

*Example 3. Sample dataframe compatible with the Neural Networks Disaggregator (numerical aggregated data).*

|**shared_id**| **feature_1** | **feature_2**| **aggregated_value_Y** |
|:-----------------:|:-----------:|:--------:|:-------------------------:|
| 0                 | 1.5      | 1.3     | 10.5                       |
| 0                 | 3.4      | 5.6    | 10.5                      |
| 1                 | 10.1      | 0.0    | 16.8                      |
| 1                 | 4.5      | 9.9    | 16.8                      |

The following step is to provide a mapping between `shared_id`, `features` and `y_aggregates` and the respective column names in the input dataframe.

```
column_name_mapping = preprocessors.NeuralNetworksColumnNameMapping(
    shared_id="shared_id",
    features=["feature_1", “feature_2”],
    y_aggregates="aggregated_value_Y",
)
```

Next, you initialize the Neural Networks `regression` model that you wish to use in the decomposition process. You can replace the default model with a customized TensorFlow Model [TensorFlow Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model).

```
task_type = "regression"
model = methods.NeuralNetworksModel(task_type=task_type)
```

The final steps are to initialize the `NeuralNetworksDisaggreagtor` class and decompose the input dataframe. The `compile` and `fit` arguments accept the exact same kwargs as the `compile` and `fit` methods of a [TensorFlow Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model). Make sure that you use a loss type compatible with the regression problems when decomposing numerical data.

```
disaggregated_dataframe = neural_network_disaggregator.fit_transform(
    dataframe=input_dataframe,
    column_name_mapping=column_name_mapping,
    compile_kwargs=dict(loss="mean_absolute_error", optimizer="adam"),
    fit_kwargs=dict(epochs=30, verbose=False),
)
```

*Example 4. Sample output from the Neural Networks Disaggregator (numerical aggregated data).*

|**shared_id**| **feature_1** | **feature_2**| **y_aggregates** | **disaggregated_y_values** |
|:-----------------:|:-----------:|:--------:|:-------------------------:|:-------------:|
| 0                 | 1.5      | 1.3     | 10.5                       | 5.0 |
| 0                 | 3.4      | 5.6    | 10.5                      | 5.5 |
| 1                 | 10.1      | 0.0    | 16.8                      | 6.8 |
| 1                 | 4.5      | 9.9    | 16.8                      | 10.0 |

### Categorical aggregated data

The statistical windowing process of categorical aggregated data with the Neural Networks Disaggregator is almost identical. At this point the Neural Networks Disaggregator can work only with binary problems. E.g. there is an aggregated number of conversions per given day and the model will only be able to assign 0 to users who did not convert and 1 otherwise.

*Example 5. Sample dataframe compatible with the Neural Networks Disaggregator (categorical aggregated data).*

|**shared_id**| **feature_1** | **feature_2**| **aggregated_value_Y** |
|:-----------------:|:-----------:|:--------:|:-------------------------:|
| 0                 | 1.5      | 1.3     | 2                       |
| 0                 | 3.4      | 5.6    | 2                      |
| 1                 | 10.1      | 0.0    | 1                      |
| 1                 | 4.5      | 9.9    | 1                      |

The following step is also to provide a mapping between `shared_id`, `features` and `y_aggregates` and the respective column names in the input dataframe.

```
column_name_mapping = preprocessors.NeuralNetworksColumnNameMapping(
    shared_id="shared_id",
    features=["feature_1", “feature_2”],
    y_aggregates="aggregated_value_Y",
)
```
Next, you initialize the Neural Networks `classification` model that you wish to use in the decomposition process.

```
task_type = "classification"
model = methods.NeuralNetworksModel(task_type=task_type)
```

The final steps are to initialize the `NeuralNetworksDisaggreagtor` class and decompose the input dataframe. Make sure that you use a loss type compatible with the binary classification problems when decomposing categorical data.

```
neural_network_disaggregator = methods.NeuralNetworksDisaggregator(
    model=model,
    task_type=task_type,
)
disaggregated_dataframe = neural_network_disaggregator.fit_transform(
    dataframe=input_dataframe,
    column_name_mapping=column_name_mapping,
    compile_kwargs=dict(loss="bce", optimizer="adam"),
    fit_kwargs=dict(epochs=30, verbose=False),
)
```

*Example 5. Sample output from the Neural Networks Disaggregator (categorical aggregated data).*

|**shared_id**| **feature_1** | **feature_2**| **y_aggregates** | **disaggregated_y_values** |
|:-----------------:|:-----------:|:--------:|:-------------------------:|:-------------:|
| 0                 | 1.5      | 1.3     | 10.5                       | 1 |
| 0                 | 3.4      | 5.6    | 10.5                      | 1 |
| 1                 | 10.1      | 0.0    | 16.8                      | 0 |
| 1                 | 4.5      | 9.9    | 16.8                      | 1 |
