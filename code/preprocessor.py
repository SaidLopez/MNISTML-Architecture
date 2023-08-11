import os
import numpy as np
import json
import numpy as np
import tempfile
import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from pickle import dump


DATA_FILEPATH = Path("dataset") / "mnist_train.csv"
BASE_DIRECTORY = Path("dataset")


def _save_splits(base_directory, train, validation, test):
    """
    One of the goals of this script is to output the three
    dataset splits. This function will save each of these
    splits to disk.
    """

    train_path = Path(base_directory) / "train"
    validation_path = Path(base_directory) / "validation"
    test_path = Path(base_directory) / "test"

    train_path.mkdir(parents=True, exist_ok=True)
    validation_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(train).to_csv(train_path / "train.csv", header=False, index=False)
    pd.DataFrame(validation).to_csv(
        validation_path / "validation.csv", header=False, index=False
    )
    pd.DataFrame(test).to_csv(test_path / "test.csv", header=False, index=False)

    
def _save_pipeline(base_directory, pipeline):
    """
    Saves the Scikit-Learn pipeline that we used to
    preprocess the data.
    """
    pipeline_path = Path(base_directory) / "pipeline"
    pipeline_path.mkdir(parents=True, exist_ok=True)
    dump(pipeline, open(pipeline_path / "pipeline.pkl", "wb"))
    
# def _save_baseline(base_directory, df_train, df_test):
#     """
#     During the data and quality monitoring steps, we will need a baseline
#     to compute constraints and statistics. This function will save that
#     baseline to the disk.
#     """

#     for split, data in [("train", df_train), ("test", df_test)]:
#         print(split)
#         baseline_path = Path(base_directory) / f"{split}-baseline"
#         baseline_path.mkdir(parents=True, exist_ok=True)

#         df = data.copy()
        
#         df.to_json(
#             baseline_path / f"{split}-baseline.json", orient="records", lines=True
#         )


class PixelTransformer(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        print('init called')
    
    def fit(self):
        print('fit called')
        return self
        
    def transform(self,pixels_df):
        """Removes from the images the pixels that have a constant intensity value,
        either always black (0) or white (255)
        Returns the cleared dataset & the list of the removed pixels (columns)"""

        #Remove the pixels that are always black to compute faster
        changing_pixels_df = pixels_df.loc[:]

        #Pixels with max value =0 are pixels that never change
        for col in pixels_df:
            if changing_pixels_df[col].max() == 0:
                changing_pixels_df.drop(columns=[col], inplace=True)

        #Same with pixels with min=255 (white pixels)
        for col in changing_pixels_df:
            if changing_pixels_df[col].min() == 255:
                changing_pixels_df.drop(columns=[col], inplace=True)

        return changing_pixels_df

def preprocess(base_directory, data_filepath):
    """
    Preprocesses the supplied raw dataset and splits it into a train,
    validation, and a test set.
    """

    """Read Data"""
    pixels_df = pd.read_csv(data_filepath)


    pipeline = Pipeline(
        steps=[
            ("pixel_transformer", PixelTransformer())
        ]
    )
    

    df = pixels_df.sample(frac=1, random_state=42)
    df_train, temp = train_test_split(df, test_size=0.3)
    df_validation, df_test = train_test_split(temp, test_size=0.5)

    y_train = df_train['label']
    y_validation = df_validation['label']
    y_test = df_test['label']


    #_save_baseline(base_directory, df_train, df_test)

    df_train = df_train.drop(["label"], axis=1)
    df_validation = df_validation.drop(["label"], axis=1)
    df_test = df_test.drop(["label"], axis=1)
    

    X_train = pipeline.transform(df_train)
    X_validation = pipeline.transform(df_validation)
    X_test = pipeline.transform(df_test)

    train = np.concatenate((X_train, np.expand_dims(y_train, axis=1)), axis=1)
    validation = np.concatenate((X_validation, np.expand_dims(y_validation, axis=1)), axis=1)
    test = np.concatenate((X_test, np.expand_dims(y_test, axis=1)), axis=1)

    _save_splits(base_directory, train, validation, test)
    _save_pipeline(base_directory, pipeline=pipeline)



if __name__ == "__main__":
    preprocess(BASE_DIRECTORY, DATA_FILEPATH)
