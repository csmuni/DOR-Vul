import pandas as pd
import re
import argparse


def normalize_c_code_soft(code: str) -> str:
    """
    Syntax-Normalization for C/C++ Code:
    Removes block comments, line comments, newlines, tabs, and extra spaces.
    Flattens the code into a single continuous string to enable robust exact-matching
    by neutralizing stylistic and formatting differences.
    """
    if not isinstance(code, str):
        return ""

    # 1. Remove Block Comments (/* ... */)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)

    # 2. Remove Line Comments (// ...)
    code = re.sub(r"//.*", "", code)

    # 3. Flatten whitespaces, newlines, and tabs
    return " ".join(code.split())


def apply_asymmetric_decontamination(train_path, test_path, output_path):
    """
    Applies Asymmetric Train-Set Decontamination to prevent data leakage.
    Ensures that no code snippet in the Train set exists in the Test set
    after syntax-normalization. The Test set remains completely untouched.
    """
    print(f"Loading datasets...\n  Train: {train_path}\n  Test: {test_path}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    initial_train_size = len(train_df)

    # Step 1: Normalize the Test Set to create a verification Hash Set
    print("Normalizing Test Set...")
    test_normalized_set = set(test_df["text"].apply(normalize_c_code_soft))

    # Step 2: Define leakage checker
    def is_leaked(code_snippet):
        return normalize_c_code_soft(code_snippet) in test_normalized_set

    # Step 3: Filter the Train Set
    print("Scanning Train Set for leaked samples...")
    mask_leaked = train_df["text"].apply(is_leaked)

    clean_train_df = train_df[~mask_leaked]
    dropped_count = initial_train_size - len(clean_train_df)

    print("\n" + "=" * 30)
    print("📊 DECONTAMINATION STATISTICS")
    print("=" * 30)
    print(f"Original Train Size : {initial_train_size}")
    print(f"Leaked Dropped      : {dropped_count}")
    print(f"Cleaned Train Size  : {len(clean_train_df)}")
    print("=" * 30 + "\n")

    # Step 4: Save the purified train set
    clean_train_df.to_csv(output_path, index=False)
    print(f"✅ Purified training set saved to: {output_path}")


if __name__ == "__main__":
    # Example usage for the reproducibility repository
    # python decontaminate_train_set.py --train train.csv --test val.csv --out clean_train.csv

    parser = argparse.ArgumentParser(
        description="Asymmetric Train-Test Decontamination Script"
    )
    parser.add_argument(
        "--train", type=str, required=True, help="Path to original train.csv"
    )
    parser.add_argument(
        "--test",
        type=str,
        required=True,
        help="Path to test.csv (e.g., official val.csv)",
    )
    parser.add_argument(
        "--out", type=str, required=True, help="Output path for purified train.csv"
    )

    args = parser.parse_args()

    apply_asymmetric_decontamination(args.train, args.test, args.out)
