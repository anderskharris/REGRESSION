# %%

# import packages
import yaml
import pandas as pd
import os
from snowflake.snowpark.session import Session
from snowflake.snowpark.types import IntegerType, Variant, VariantType, DecimalType
import snowflake.connector as snowflake
from snowflake.snowpark.functions import udf
from dotenv import load_dotenv
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from joblib import dump

# %%

# connect to snowflake
with open("config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

with open("secrets.yaml", "r") as f:
    secrets = yaml.load(f, Loader=yaml.FullLoader)

connection_parameters = {
    "account": config["account"],
    "user": secrets["user"],
    "password": secrets["password"],
    "role": config["role"],
    "warehouse": config["warehouse"],
    "database": config["database"],
    "schema": config["schema"],
}

session = Session.builder.configs(connection_parameters).create()
session.add_packages("snowflake-snowpark-python", "numpy", "scikit-learn", "pandas")

# %%


# create function
def car_saleprice_prediction(session: Session):
    df = session.sql(
        'SELECT "YEAR", "MILES", "PRICE", "NAME" FROM CARVANA_DATA'
    ).to_pandas()

    X = df[["YEAR", "MILES", "NAME"]]
    y = df["PRICE"]

    numeric_features = ["YEAR", "MILES"]
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_features = ["NAME"]
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    lmod = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", LinearRegression())]
    )

    lmod.fit(X, y)

    model_output_dir = "/tmp"
    model_file = os.path.join(model_output_dir, "model.joblib")
    dump(lmod, model_file)
    session.file.put(model_file, "@carvana_model", overwrite=True)

    return lmod.score(X, y)


# %%

# register procedure to snowflake
session.sproc.register(
    func=car_saleprice_prediction,
    name="car_saleprice_prediction",
    return_type=DecimalType(38, 2),
    stage_location="@CARVANA_MODEL",
    replace=True,
)

# %%

# call stored procedure to write model file to stage

print(session.call("car_saleprice_prediction"))

# %%

# create udf

session.clear_imports()
session.clear_packages()
session.add_import("@carvana_model/model.joblib.gz")


@udf(
    name="predict_price",
    session=session,
    packages=["pandas", "joblib", "scikit-learn", "numpy"],
    replace=True,
    stage_location="@CARVANA_MODEL",
)
def predict_price(cars: list) -> float:
    import sys
    import pandas as pd
    from joblib import load

    IMPORT_DIRECTORY_NAME = "snowflake_import_directory"
    import_dir = sys._xoptions[IMPORT_DIRECTORY_NAME]

    model_file = import_dir + "model.joblib.gz"

    model = load(model_file)

    features = ["NAME", "YEAR", "MILES"]
    df = pd.DataFrame([cars], columns=features)

    price = model.predict(df)[0]

    return price


# %%

# close session

session.close()

# %%
