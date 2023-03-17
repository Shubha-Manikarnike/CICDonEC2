from fastapi import FastAPI
import yaml
# Import Union since our Item object will have tags that can be strings or a list.
from typing import Union
# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel
from typing_extensions import Literal
from joblib import load
import pandas as pd
from fastapi import  Request
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder


# Declare the data object with its components and their type.

def inference(model, x):
    """ Run model inferences and return the predictions.
    Inputs
    ------
    model : RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    preds = model.predict(x)
    return preds

def process_data(
        x,
        categorical_features=[],
        label=None,
        training=True,
        encoder=None,
        lb=None):
    """ Process the data used in the machine learning pipeline.
    Processes the data using one hot encoding for the categorical features
    and amlabel binarizer for the labels. This can be used in either
    training or inference/validation.
    Note: depending on the type of model used, you may want to add in
    functionality that scales the continuous data.
    Inputs
    ------
    X : pd.DataFrame
        Dataframe containing the features and label. Columns in
        `categorical_features`
    categorical_features: list[str]
        List containing the names of the categorical features (default=[])
    label : str
        Name of the label column in `X`. If None, then an empty array will
        be returned for y (default=None)
    training : bool
        Indicator if training mode or inference/validation mode.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder, only used if training=False.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer, only used if training=False.
    Returns
    -------
    X : np.array
        Processed data.
    y : np.array
        Processed labels if labeled=True, otherwise empty np.array.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained OneHotEncoder if training is True, otherwise returns the
        encoder passed in.
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained LabelBinarizer if training is True, otherwise returns the
        binarizer passed in.
    """

    if categorical_features is None:
        categorical_features = list()
    if label is not None:
        y = x[label]
        x = x.drop([label], axis=1)
    else:
        y = np.array([])

    x_categorical = x[categorical_features].values
    x_continuous = x.drop(*[categorical_features], axis=1)

    if training is True:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        lb = LabelBinarizer()
        x_categorical = encoder.fit_transform(x_categorical)
        y = lb.fit_transform(y.values).ravel()
    else:
        x_categorical = encoder.transform(x_categorical)
        try:
            y = lb.transform(y.values).ravel()
        # Catch the case where y is None because we're doing inference.
        except AttributeError:
            pass

    x = np.concatenate([x_continuous, x_categorical], axis=1)
    return x, y, encoder,lb 


def run_inference(data, cat_features):
    """
    Load model and run inference
    Parameters
    ----------
    root_path
    data
    cat_features
    Returns
    -------
    prediction
    """
    model = load("model.joblib")
    encoder = load("encoder.joblib")
    lb = load("lb.joblib")

    X, _, _, _ = process_data(
        data,
        categorical_features=cat_features,
        encoder=encoder, lb=lb, training=False)

    pred = inference(model, X)
    prediction = lb.inverse_transform(pred)[0]

    return prediction


class ModelInput(BaseModel):
    age: int
    workclass: Literal[
        'State-gov', 'Self-emp-not-inc', 'Private', 'Federal-gov',
        'Local-gov', 'Self-emp-inc', 'Without-pay']
    fnlgt: int
    education: Literal[
        'Bachelors', 'HS-grad', '11th', 'Masters', '9th',
        'Some-college',
        'Assoc-acdm', '7th-8th', 'Doctorate', 'Assoc-voc', 'Prof-school',
        '5th-6th', '10th', 'Preschool', '12th', '1st-4th']
    marital_status: Literal[
        'Never-married', 'Married-civ-spouse', 'Divorced',
        'Married-spouse-absent', 'Separated', 'Married-AF-spouse',
        'Widowed']
    occupation: Literal[
        'Adm-clerical', 'Exec-managerial', 'Handlers-cleaners',
        'Prof-specialty', 'Other-service', 'Sales', 'Transport-moving',
        'Farming-fishing', 'Machine-op-inspct', 'Tech-support',
        'Craft-repair', 'Protective-serv', 'Armed-Forces',
        'Priv-house-serv']
    relationship: Literal[
        'Not-in-family', 'Husband', 'Wife', 'Own-child',
        'Unmarried', 'Other-relative']
    race: Literal[
        'White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo',
        'Other']
    sex: Literal['Male', 'Female']
    hours_per_week: int
    native_country: Literal[
        'United-States', 'Cuba', 'Jamaica', 'India', 'Mexico',
        'Puerto-Rico', 'Honduras', 'England', 'Canada', 'Germany', 'Iran',
        'Philippines', 'Poland', 'Columbia', 'Cambodia', 'Thailand',
        'Ecuador', 'Laos', 'Taiwan', 'Haiti', 'Portugal',
        'Dominican-Republic', 'El-Salvador', 'France', 'Guatemala',
        'Italy', 'China', 'South', 'Japan', 'Yugoslavia', 'Peru',
        'Outlying-US(Guam-USVI-etc)', 'Scotland', 'Trinadad&Tobago',
        'Greece', 'Nicaragua', 'Vietnam', 'Hong', 'Ireland', 'Hungary',
        'Holand-Netherlands']

# Save items from POST method in the memory
items = {}
with open('config.yml') as f:
    config = yaml.load(f,Loader=yaml.Loader)
# Initialize FastAPI instance
app = FastAPI()

@app.get("/")
async def say_hello():
    return {"greeting": "Hello World!"}

# This allows sending of data (our TaggedItem) via POST to the API.
@app.post("/items/")
async def infer(input_data: ModelInput):

    input_data = input_data.dict()

    change_keys = config['infer']['update_keys']
    columns = config['infer']['columns']
    cat_features = config['data']['cat_features']

    for new_key, old_key in change_keys:
        input_data[new_key] = input_data.pop(old_key)

    input_df = pd.DataFrame(data=input_data.values(), index=input_data.keys()).T
    input_df = input_df[columns]

    prediction = run_inference(input_df, cat_features)

    return {"prediction": prediction}




@app.exception_handler(ValueError)
async def value_error_exception_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"message": str(exc)},
    )
# A GET that in this case just returns the item_id we pass,
# but a future iteration may link the item_id here to the one we defined in our TaggedItem.
