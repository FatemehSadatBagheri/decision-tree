# AI Project: Decision Tree for Online Fraud Detection

-----

## description

This project is a from-scratch Python implementation of a **Decision Tree classifier**. It was created for the "Artificial Intelligence and Expert Systems" course at Iran University of Science and Technology.

The primary goal of this project is to **detect fraudulent online payment transactions**. The model is trained and evaluated using the `onlinefraud.csv` dataset. The implementation is built from the ground up, without relying on pre-built decision tree libraries like `scikit-learn` for the core algorithm, as per the project requirements.

The classifier supports two different splitting criteria:

  * **Entropy** (Information Gain)
  * **Gini Index**

-----

## 💻 Technology Stack

  * **Python 3.x**
  * **NumPy:** For numerical computing and array manipulation.
  * **Pandas:** For data loading (`onlinefraud.csv`) and preprocessing.

-----

## 🧠 Algorithm & Features

  * **Custom `Node` Class:** Implements the basic structure of a tree node, storing the feature, threshold, child nodes (`left`, `right`), and the predicted `value` if it's a leaf.
  * **Custom `DecisionTree` Class:** Contains the full logic for the classifier.
  * **Dual Splitting Criteria:** Can be initialized to use either `entropy` or `gini` index to find the best split.
  * **Configurable Hyperparameters:** Supports `min_samples_split`, `max_depth`, and `n_features` to control tree growth and prevent overfitting.
  * **Core Methods Implemented:**
      * `fit(X, y)`: Trains the tree on data.
      * `predict(X)`: Predicts labels for new data.
      * `_grow_tree()`: A private recursive function to build the tree.
      * `_best_split()`: Finds the optimal feature and threshold by calculating information gain.
      * `_information_gain()`: Calculates the gain using the selected mode (entropy or gini).
      * `_entropy()` and `_gini_index()`: Calculate impurity.
  * **Data Type Handling:** The `_split` method can handle both numerical (continuous) and categorical (string) features.
  * **Console Visualization:** Includes a `print_tree()` method to output a text-based representation of the trained tree.

-----

## 📦 Installation

1.  Ensure you have Python 3.x installed.
2.  Clone this repository to your local machine.
3.  Install the required Python libraries.
    ```bash
    pip install numpy pandas
    ```
4.  Download the `onlinefraud.csv` dataset and place it in the root directory of the project.

-----

## 🚀 Usage

The main script is designed to be run directly from the command line. It will automatically load the data, train the model, and evaluate its accuracy.

1.  Run the script from your terminal:

2.  The script will perform the following steps as defined in its `if __name__== "__main__":` block:

      * Load `onlinefraud.csv` using pandas.
      * Preprocess the data by dropping the `nameOrig` column and any rows with missing values.
      * Split the data:
          * **Training Set:** The first 2000 rows.
          * **Test Set:** The remaining rows.
      * Initialize the `DecisionTree` (defaulting to `max_depth=10`, `n_features=8`, and `mode="entropy"`).
      * Train the model using the `fit()` method.
      * Generate predictions on the test set using the `predict()` method.
      * Calculate and print the final accuracy of the model.

### Customizing the Split Criterion

To switch from the default **Entropy** to the **Gini Index**, modify the `DecisionTree` initialization in the main script:

```python
# Default (Entropy)
clf = DecisionTree(max_depth=10, n_features=8, mode="entropy")

# To use Gini Index
clf_gini = DecisionTree(max_depth=10, n_features=8, mode="gini")

# Then fit the new classifier
clf_gini.fit(X_train, y_train)
predictions = clf_gini.predict(X_test)
```

-----

## 📄 License

This project is licensed under the [NAME OF LICENSE] License.
