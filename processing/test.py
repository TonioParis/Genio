import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer



X = pd.DataFrame({
    "Age": [25, np.nan, 40, 35],
    "Credit_amount": [1000, 2000, np.nan, 3000],
    "Sex": ["male", "female", "female", "male"]
})

print("Données initiales :")
print(X)

numeric_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),  # remplace les NaN
    ("scaler", StandardScaler())                    # normalise
])
