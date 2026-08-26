#!/usr/bin/env python3
"""
Leaf Wetness Prediction CLI Tool
Inference script using ONNX Runtime for agrometeorological models.
"""

import argparse
import sys
import numpy as np
import pandas as pd
import onnxruntime as ort

from pathlib import Path


# Expected input feature names in exact order
EXPECTED_FEATURES = [
    "soil_vwc",
    "soil_temperature",
    "temperature",
    "rh",
    "rain",
    "wind_speed",
    "doy",
    "hour"
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Leaf Wetness prediction (Dry/Wet) using pre-trained models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=True,
        help="Path to the trained .onnx model file.",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Path to the input CSV file containing weather observations.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="predictions.csv",
        help="Path where the output CSV file with predictions will be saved.",
    )
    parser.add_argument(
        "--prob",
        action="store_true",
        help="Include prediction probabilities for the 'Wet' class (if supported by model).",
    )

    return parser.parse_args()


def load_and_validate_csv(input_path: Path) -> pd.DataFrame:
    """Reads input CSV and verifies required weather columns are present."""
    if not input_path.exists():
        sys.exit(f"Error: Input file '{input_path}' does not exist.")

    df = pd.read_csv(input_path)

    # Convert column names to lowercase for case-insensitive matching
    df.columns = df.columns.str.strip().str.lower()

    missing_cols = [col for col in EXPECTED_FEATURES if col not in df.columns]
    if missing_cols:
        sys.exit(
            f"Error: Missing required feature column(s): {missing_cols}\n"
            f"CSV must contain the following headers: {EXPECTED_FEATURES}"
        )

    return df


def run_inference(model_path: Path, features_array: np.ndarray):
    """Executes ONNX Runtime session and returns raw outputs."""
    if not model_path.exists():
        sys.exit(f"Error: Model file '{model_path}' does not exist.")

    try:
        session = ort.InferenceSession(str(model_path))
    except Exception as e:
        sys.exit(f"Error loading ONNX model: {e}")

    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: features_array})

    # ONNX classification outputs: [labels, probabilities_map_list]
    predictions = outputs[0]
    probabilities = outputs[1] if len(outputs) > 1 else None

    return predictions, probabilities


def main():
    args = parse_args()

    model_path = Path(args.model)
    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"--> Loading input data from: {input_path}")
    df = load_and_validate_csv(input_path)

    # Extract required features in exact expected sequence
    feature_data = df[EXPECTED_FEATURES].values.astype(np.float32)

    print(f"--> Running inference with model: {model_path.name}")
    predictions, probabilities = run_inference(model_path, feature_data)

    # Append results to original DataFrame
    df["predicted_status_code"] = predictions
    df["predicted_status_label"] = np.where(predictions == 1, "Wet", "Dry")

    # Extract class 1 (Wet) probability if requested and available
    if args.prob:
        if probabilities is not None and isinstance(probabilities, list):
            # ONNX Scikit-learn probability output is a list of dicts: [{0: p_dry, 1: p_wet}, ...]
            wet_probs = [prob_dict.get(1, np.nan) for prob_dict in probabilities]
            df["prob_wet"] = wet_probs
        else:
            print(
                "Warning: Probability flag (--prob) passed, but model output does not contain probability maps."
            )

    # Save results
    df.to_csv(output_path, index=False)
    print(f"--> Success! Predictions saved to: {output_path}")


if __name__ == "__main__":
    main()
