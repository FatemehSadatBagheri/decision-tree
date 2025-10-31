# AI Project: Decision Tree for Online Fraud Detection

---

## Description

[cite_start]This project is a from-scratch Python implementation of a **Decision Tree classifier**[cite: 26, 89, 174]. [cite_start]It was created for the "Artificial Intelligence and Expert Systems" course at Iran University of Science and Technology[cite: 1, 139, 141].

[cite_start]The primary goal of this project is to **detect fraudulent online payment transactions**[cite: 167]. [cite_start]The model is trained and evaluated using the `onlinefraud.csv` dataset[cite: 178]. [cite_start]The implementation is built from the ground up, without relying on pre-built decision tree libraries like `scikit-learn` for the core algorithm, as per the project requirements[cite: 203].

The classifier supports two different splitting criteria:
* [cite_start]**Entropy** (Information Gain) [cite: 62, 99, 133, 176]
* [cite_start]**Gini Index** [cite: 136, 176]

## Features

* [cite_start]**Custom `Node` Class:** Implements the basic structure of a tree node, storing the feature, threshold, child nodes (`left`, `right`), and the predicted `value` if it's a leaf [cite: 6-16].
* [cite_start]**Custom `DecisionTree` Class:** Contains the full logic for the classifier[cite: 26, 89].
* **Dual Splitting Criteria:** Can be initialized to use either `entropy` or `gini` index to find the best split [Code, cite: 176].
* [cite_start]**Configurable Hyperparameters:** Supports `min_samples_split` [cite: 28, 92][cite_start], `max_depth` [cite: 29, 93][cite_start], and `n_features` [cite: 30, 94] [cite_start]to control tree growth and prevent overfitting[cite: 123].
* **Core Methods Implemented:**
    * [cite_start]`fit(X, y)`: Trains the tree on data[cite: 32].
    * [cite_start]`predict(X)`: Predicts labels for new data[cite: 81, 103].
    * [cite_start]`_grow_tree()`: A private recursive function to build the tree[cite: 36, 96].
    * [cite_start]`_best_split()`: Finds the optimal feature and threshold by calculating information gain[cite: 48, 97].
    * [cite_start]`_information_gain()`: Calculates the gain using the selected mode (entropy or gini)[cite: 62, 99].
    * [cite_start]`_entropy()` [cite: 73, 101, 133] and `_gini_index()` [Code, cite: 136]: Calculate impurity.
* **Data Type Handling:** The `_split` method can handle both numerical (continuous) and categorical (string) features [Code].
* **Console Visualization:** Includes a `print_tree()` method to output a text-based representation of the trained tree [Code].

## 📦 Installation

1.  Ensure you have Python 3.x installed.
2.  Clone this repository to your local machine.
    ```bash
    git clone [YOUR_REPOSITORY_URL]
    cd [REPOSITORY_DIRECTORY]
    ```
3.  Install the required Python libraries. This project uses `numpy` and `pandas` for data manipulation [Code].
    ```bash
    pip install numpy pandas
    ```
4.  [cite_start]Download the `onlinefraud.csv` dataset [cite: 178] and place it in the root directory of the project.

## 🚀 Usage

The main script is designed to be run directly from the command line. It will automatically load the data, train the model, and evaluate its accuracy.

1.  Run the script from your terminal:
    ```bash
    python your_script_name.py
    ```
    *(Note: Replace `your_script_name.py` with the actual name of the Python file.)*

2.  The script will perform the following steps as defined in its `if __name__== "__main__":` block [Code]:
    * Load `onlinefraud.csv` using pandas.
    * Preprocess the data by dropping the `nameOrig` column and any rows with missing values.
    * Split the data:
        * [cite_start]**Training Set:** The first 2000 rows[cite: 179].
        * **Test Set:** The remaining rows.
    * Initialize the `DecisionTree` (defaulting to `max_depth=10`, `n_features=8`, and `mode="entropy"`).
    * Train the model using the `fit()` method.
    * Generate predictions on the test set using the `predict()` method.
    * [cite_start]Calculate and print the final accuracy of the model[cite: 111].

### Customizing the Split Criterion

To switch from the default **Entropy** to the **Gini Index**, modify the `DecisionTree` initialization in the main script [Code]:

```python
# Default (Entropy)
clf = DecisionTree(max_depth=10, n_features=8, mode="entropy")

# To use Gini Index
clf_gini = DecisionTree(max_depth=10, n_features=8, mode="gini")

# Then fit the new classifier
clf_gini.fit(X_train, y_train)
predictions = clf_gini.predict(X_test)