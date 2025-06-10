# Heart Disease Prediction using Deep Learning

This project aims to predict the presence of heart disease in patients based on a set of medical attributes. A Deep Neural Network (DNN) was built using TensorFlow and Keras to classify whether a patient has heart disease.

## Dataset

The dataset used in this project is the "Heart Failure Prediction" dataset, which contains 11 clinical features for predicting heart disease events. The dataset was loaded from a `heart.csv` file.

Key features include:
- Age
- Sex
- ChestPainType
- RestingBP (Resting Blood Pressure)
- Cholesterol
- FastingBS (Fasting Blood Sugar)
- MaxHR (Maximum Heart Rate)
- ExerciseAngina
- ST_Slope

The target variable is `HeartDisease`, which is a binary outcome (1 for presence, 0 for absence of heart disease).

## Project Workflow

The project follows standard machine learning practices:

1.  **Data Loading**: The dataset is loaded using the `pandas` library.
2.  **Data Preprocessing**:
    * Categorical features were converted into numerical format using `LabelEncoder`.
    * The data was split into training (80%) and testing (20%) sets.
    * Features were normalized using `MinMaxScaler` to scale them between 0 and 1, which helps with model convergence.
3.  **Model Building**:
    * A `Sequential` model was created with Keras.
    * The architecture consists of an input layer, four hidden `Dense` layers with `relu` activation, and `Dropout` layers to prevent overfitting.
    * The output layer is a single `Dense` neuron with a `sigmoid` activation function, suitable for binary classification.
4.  **Model Compilation and Training**:
    * The model was compiled with the `Adam` optimizer and `binary_crossentropy` loss function.
    * It was trained for 200 epochs with a batch size of 128.

## Results

-   **Training Accuracy**: Reached approximately 95%.
-   **Validation Accuracy**: Achieved a peak of around 91%.

The model demonstrates good performance in identifying patients with heart disease. The training and validation accuracy plots show that the model learns effectively, although there are signs of overfitting in later epochs, which could be addressed with techniques like Early Stopping.

## How to Run the Project

1.  Clone the repository:
    ```bash
    git clone [https://github.com/Erfanhajiesmaeili/Deep-Learning/blob/main/Heart_diseasess/Heart_diseases.ipynb]
    ```
2.  Open the `Heart_diseases.ipynb` notebook in Google Colab or a local Jupyter environment.
3.  Make sure you have the `heart.csv` dataset available and update the file path in the notebook.
4.  Run the cells sequentially to preprocess the data, build the model, and start the training process.
