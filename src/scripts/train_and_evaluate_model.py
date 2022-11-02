import argparse
import json
from typing import Any, Dict, List, Union
from src.process.preprocessor import Preprocessor
from src.process.regressor import Regressor


def initialize_cli(sys_args: Union[List, None] = None) -> Dict[str, Any]:
    """Initialise a CLI which parses command line arguments.
    Args:
        sys_args (Union[List, None]): The list of command line arguments passed
            to a Python script.
    Returns:
        dict: A dictionary of initialise_cli arguments parsed by argparse.
    """
    # Initialize Parser
    parser = argparse.ArgumentParser(description="Train regressor model.")

    # Add arguments
    parser.add_argument(
        "-t",
        "--test-size",
        help="Share of datapoints to use for testing.",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "-n",
        "--normalize-data",
        help="Whether to do a min-max normalization of the data or not.",
        type=bool,
        default=True,
    )

    parser.add_argument(
        "-r",
        "--model-save-path",
        help="Path to which the trained regressor model will be saved. Model won't be saved an empty string '' is given. Defaults to assets/models/model.joblib.",
        type=str,
        default="assets/models/model.joblib",
    )

    parser.add_argument(
        "-p",
        "--preprocessor-save-path",
        help="Path to which the preprocessor object will be saved. Object won't be saved an empty string '' is given. Defaults to assets/preprocessors/preprocessor.joblib.",
        type=str,
        default="assets/preprocessors/preprocessor.joblib",
    )

    parser.add_argument(
        "-m",
        "--metrics-save-path",
        help="Folder path to which the features and model metrics object will be saved. Metrics won't be saved an empty string '' is given. Defaults to assets/metrics/.",
        type=str,
        default="assets/metrics",
    )

    parser.add_argument(
        "-f",
        "--family-composition-path",
        help="Path to the family composition Excel file.",
        type=str,
        default="assets/data/2021_family_composition_amsterdam.xlsx",
    )

    parser.add_argument(
        "-w",
        "--woz-prices-path",
        help="Path to the woz prices Excel file.",
        type=str,
        default="assets/data/woz_prices_2021_amsterdam.xlsx",
    )

    return vars(parser.parse_args(sys_args))


if __name__ == "__main__":
    kwargs = initialize_cli()

    # create a preprocessor object
    preprocessor = Preprocessor(normalize_data=kwargs["normalize_data"])

    # load data
    X, y = preprocessor.load_data(
        family_composition_file_path=kwargs["family_composition_path"],
        woz_file_path=kwargs["woz_prices_path"],
    )

    # generate report about data quality
    data_report = preprocessor.get_report(X, y)

    # split data into train and test sets
    X_train, X_test, y_train, y_test = preprocessor.train_test_split(
        X, y, test_size=kwargs["test_size"]
    )

    # create a regressor object
    regressor = Regressor()

    # fit regressor to data
    regressor.fit(X_train, y_train)

    # evaluate the regressor
    model_report = regressor.evaluate(X_test, y_test, preprocessor)

    # save the preprocessor and regressor objects
    if kwargs["model_save_path"]:
        regressor.save_model(kwargs["model_save_path"])
        print(f"Regressor model saved to {kwargs['model_save_path']}.")

    if kwargs["preprocessor_save_path"]:
        preprocessor.save_object(kwargs["preprocessor_save_path"])
        print(f"Preprocessor object saved to {kwargs['preprocessor_save_path']}.")

    # saving the metrics
    if kwargs["metrics_save_path"]:
        data_report_save_path = kwargs["metrics_save_path"] + "/features_score.json"
        model_report_save_path = kwargs["metrics_save_path"] + "/model_score.json"

        with open(data_report_save_path, "w") as f:
            f.write(json.dumps(data_report))

        with open(model_report_save_path, "w") as f:
            f.write(json.dumps(model_report))

        print(f"Features scores saved to {data_report_save_path}.")
        print(f"Model scores saved to {model_report_save_path}.")
