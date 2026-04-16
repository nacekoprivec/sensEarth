import pandas as pd
import sys
from pathlib import Path

def convert_csv(input_file: str, output_file: str = None):
    # Read semicolon-separated CSV
    df = pd.read_csv(input_file, sep=";")

    # Rename required columns
    df = df.rename(columns={
        "Datum": "ds",
        "vodostaj (cm)": "y"
    })

    # Keep only required columns
    df = df[["ds", "y"]]

    # Convert date format (optional but recommended)
    # from dd.mm.yyyy → datetime
    df["ds"] = pd.to_datetime(df["ds"], format="%d.%m.%Y")

    # Ensure numeric values
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    # Drop invalid rows
    df = df.dropna()

    # Output file
    if output_file is None:
        path = Path(input_file)
        output_file = str(path.with_name(path.stem + "_clean.csv"))

    # Save with comma separator
    df.to_csv(output_file, index=False, sep=",")

    print(f"Converted file saved to: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert.py input.csv [output.csv]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None

    convert_csv(input_file, output_file)