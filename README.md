# Heart Failure Prediction ANN

This project demonstrates the use of an Artificial Neural Network (ANN) to predict the occurrence of heart failure using clinical data. The model is built using TensorFlow and Keras, processing both numerical and categorical features to classify patients.

## Project Structure

- `project/main.ipynb`: The main Jupyter Notebook containing the data preprocessing, model architecture, training, and evaluation.
- `project/heart.csv`: The dataset used for training and testing the model.
- `requirements.txt`: List of Python dependencies required to run the project.

## Dependencies

The project requires the following Python libraries:
- `numpy`
- `pandas`
- `scikit-learn`
- `tensorflow`
- `matplotlib`
- `seaborn`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Mateosz9656/Heart_failure_ANN.git
   ```

2. Navigate to the project directory:
   ```bash
   cd Heart_failure_ANN
   ```

3. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

4. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Open the Jupyter Notebook:
   ```bash
   jupyter notebook project/main.ipynb
   ```

2. Run the cells in the notebook to load the data, train the model, and view the results.

## Model Overview

The model uses a combination of data handling techniques:
- **Preprocessing**: Handling missing values with KNNImputer, scaling numerical features with StandardScaler, and encoding categorical features with LabelEncoder.
- **Architecture**: A multi-branch neural network that handles numerical and categorical inputs separately (using Embeddings), then concatenates them into several Dense layers with Dropout and BatchNormalization for robust learning.
