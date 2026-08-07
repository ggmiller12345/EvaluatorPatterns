"""
Classify claims in individual response files.

Project structure:

    project/
    ├── classify_claims.py
    ├── Jacques_Hnizdovsky_Prompts.csv
    ├── catalog/
    │   └── Criterion_Micro_Pattern_Catalog.txt
    ├── responses/
    │   ├── JH-001.txt
    │   ├── JH-002.txt
    │   └── ...
    └── claims/
        ├── JH-001.txt
        ├── JH-002.txt
        └── ...

For each row in the CSV, the program:

1. Reads the Prompt ID and Prompt.
2. Opens the matching file in responses/.
3. Reads catalog/Criterion_Micro_Pattern_Catalog.txt.
4. Sends the prompt, response, and catalog to the OpenAI API.
5. Saves the claim analysis under the same filename in claims/.

Required CSV columns:

    Prompt ID
    Prompt

Install:

    pip install --upgrade openai pandas

Set the API key in Windows PowerShell:

    $env:OPENAI_API_KEY="your-api-key"

Run:

    python classify_claims.py "Jacques_Hnizdovsky_Prompts.csv"
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import time
from pathlib import Path

import pandas as pd
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)


# ---------------------------------------------------------------------------
# File and directory configuration
# ---------------------------------------------------------------------------

SCRIPT_DIRECTORY = Path(__file__).resolve().parent

CATALOG_DIRECTORY = SCRIPT_DIRECTORY / "..\\catalog"
CATALOG_FILE = (
    CATALOG_DIRECTORY / "Criterion_Micro_Pattern_Catalog.txt"
)

RESPONSES_DIRECTORY = SCRIPT_DIRECTORY / "..\\responses"
CLAIMS_DIRECTORY = SCRIPT_DIRECTORY / "..\\claims"

ERRORS_DIRECTORY = CLAIMS_DIRECTORY / "errors"
AUDIT_DIRECTORY = CLAIMS_DIRECTORY / "input_prompts"

PROMPT_ID_COLUMN = "Prompt ID"
PROMPT_COLUMN = "Prompt"

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6")


# ---------------------------------------------------------------------------
# Model instructions
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTIONS = """
You are an evaluator-design and claim-analysis specialist.

Divide the supplied response into discrete, testable claims and classify each
claim using the supplied Criterion Micro-Pattern Catalog.

Requirements:

1. Preserve the meaning and scope of the original response.
2. Divide compound statements into atomic claims when the components could be
   independently true or false.
3. Do not add claims that are absent from the response.
4. Classify every substantive claim.
5. Use only the canonical criterion micro-pattern names in the catalog.
6. Distinguish the claim's knowledge dimension from its criterion
   micro-pattern.
7. Use secondary classifications when one claim properly instantiates more
   than one pattern.
8. Preserve epistemic qualifications such as may, might, likely, suggests,
   establishes, and does not establish.
9. Preserve deontic qualifications such as must, should, may, optional, and
   prohibited.
10. Do not determine whether a claim is correct unless explicitly requested.
11. Do not rewrite the response as a general summary.
12. Return the result in the requested Markdown format.
""".strip()


OUTPUT_FORMAT = """
Return the analysis in this Markdown format:

# Claim Analysis

## Source Information

- Prompt ID: [prompt ID]
- Number of atomic claims: [number]

## Claims

### Claim 1

- Claim: [complete atomic claim]
- Knowledge dimension: [Factual, Conceptual, Procedural, Metacognitive, or Cross-Cutting]
- Criterion micro-pattern: [canonical pattern name]
- Secondary pattern(s): [canonical pattern names or None]
- Rationale: [brief classification rationale]
- Response evidence: [exact or minimally edited supporting passage]
- Epistemic status: [definite, qualified, uncertain, inferred, or not applicable]
- Deontic status: [mandatory, advisory, permitted, prohibited, or not applicable]

Continue until every substantive claim has been classified.

## Criterion Micro-Pattern Summary

| Criterion micro-pattern | Count |
|---|---:|
| [pattern] | [count] |

## Knowledge-Dimension Summary

| Knowledge dimension | Count |
|---|---:|
| [dimension] | [count] |

## Excluded Material

Identify headings, transitions, questions, repetitions, disclaimers, or other
text that was not treated as a substantive claim.
""".strip()


# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read prompts from a CSV, match them with files in responses/, "
            "classify the response claims using the catalog, and save the "
            "results in claims/."
        )
    )

    parser.add_argument(
        "csv_file",
        type=Path,
        help="CSV containing the Prompt ID and Prompt columns.",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use. Default: {DEFAULT_MODEL}",
    )

    parser.add_argument(
        "--response-extension",
        default=".txt",
        help="Extension of files in responses/. Default: .txt",
    )

    parser.add_argument(
        "--output-extension",
        default=None,
        help=(
            "Extension for files written to claims/. By default, the "
            "response-file extension is retained."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing files in claims/.",
    )

    parser.add_argument(
        "--start-row",
        type=int,
        default=1,
        help="First CSV data row to process, using 1-based numbering.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of CSV records to process.",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=6,
        help="Maximum API attempts for each item. Default: 6",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to pause after each successful request. Default: 0.5",
    )

    parser.add_argument(
        "--save-input-prompts",
        action="store_true",
        help=(
            "Save each complete API input under "
            "claims/input_prompts/ for auditing."
        ),
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Validation and file handling
# ---------------------------------------------------------------------------

def normalize_extension(extension: str) -> str:
    """Return a normalized file extension beginning with a period."""
    extension = extension.strip()

    if not extension:
        raise ValueError("A file extension cannot be empty.")

    if not extension.startswith("."):
        extension = "." + extension

    return extension


def sanitize_filename(value: object) -> str:
    """Convert a Prompt ID into a safe filename."""
    text = str(value).strip()

    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)
    text = text.strip(" ._")

    return text or "unnamed_prompt"


def resolve_csv_path(csv_path: Path) -> Path:
    """
    Resolve the CSV path.

    Relative paths are first interpreted relative to the current working
    directory. If not found there, they are interpreted relative to the
    script's directory.
    """
    if csv_path.is_absolute():
        return csv_path

    working_directory_path = Path.cwd() / csv_path

    if working_directory_path.exists():
        return working_directory_path.resolve()

    return (SCRIPT_DIRECTORY / csv_path).resolve()


def validate_directories_and_files(csv_path: Path) -> None:
    """Validate required inputs and create output directories."""
    print(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV file not found: {csv_path}"
        )

    if not CATALOG_DIRECTORY.exists():
        raise FileNotFoundError(
            "Catalog directory not found. Expected location: "
            f"{CATALOG_DIRECTORY}"
        )

    if not CATALOG_FILE.exists():
        raise FileNotFoundError(
            "Criterion Micro-Pattern Catalog not found. "
            f"Expected location: {CATALOG_FILE}"
        )

    if not RESPONSES_DIRECTORY.exists():
        raise FileNotFoundError(
            "Responses directory not found. Expected location: "
            f"{RESPONSES_DIRECTORY}"
        )

    CLAIMS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_csv(csv_path: Path) -> pd.DataFrame:
    """Read and validate the prompt CSV."""
    dataframe = pd.read_csv(
        csv_path,
        encoding="utf-8-sig",
        dtype=str,
        keep_default_na=False,
    )

    dataframe.columns = [
        str(column).strip()
        for column in dataframe.columns
    ]

    required_columns = {
        PROMPT_ID_COLUMN,
        PROMPT_COLUMN,
    }

    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        available_text = ", ".join(dataframe.columns)

        raise ValueError(
            f"Missing required CSV column(s): {missing_text}. "
            f"Available columns: {available_text}"
        )

    dataframe[PROMPT_ID_COLUMN] = (
        dataframe[PROMPT_ID_COLUMN]
        .astype(str)
        .str.strip()
    )

    dataframe[PROMPT_COLUMN] = (
        dataframe[PROMPT_COLUMN]
        .astype(str)
        .str.strip()
    )

    dataframe = dataframe[
        (dataframe[PROMPT_ID_COLUMN] != "")
        & (dataframe[PROMPT_COLUMN] != "")
    ].copy()

    duplicate_ids = dataframe[
        dataframe[PROMPT_ID_COLUMN].duplicated(keep=False)
    ][PROMPT_ID_COLUMN].tolist()

    if duplicate_ids:
        examples = ", ".join(
            sorted(set(duplicate_ids))[:10]
        )

        raise ValueError(
            "The CSV contains duplicate Prompt ID values. "
            f"Examples: {examples}"
        )

    return dataframe.reset_index(drop=True)


def load_catalog() -> str:
    """Load the catalog from the fixed catalog/ directory."""
    catalog_text = CATALOG_FILE.read_text(
        encoding="utf-8-sig"
    ).strip()

    if not catalog_text:
        raise ValueError(
            f"The catalog file is empty: {CATALOG_FILE}"
        )

    return catalog_text


def read_response(response_path: Path) -> str:
    """Read and validate an individual response file."""
    response_text = response_path.read_text(
        encoding="utf-8-sig"
    ).strip()

    if not response_text:
        raise ValueError(
            f"The response file is empty: {response_path}"
        )

    return response_text


# ---------------------------------------------------------------------------
# Claim-analysis prompt
# ---------------------------------------------------------------------------

def build_analysis_prompt(
    prompt_id: str,
    original_prompt: str,
    response_text: str,
    catalog_text: str,
) -> str:
    """
    Construct the claim-analysis request.

    Delimiters prevent the original prompt and response from being confused
    with the analysis instructions.
    """
    return f"""
Using the prompt and response below, divide the response into atomic claims
and type the claims according to the Criterion Micro-Pattern Catalog.

Prompt ID: {prompt_id}

<original_prompt>
{original_prompt}
</original_prompt>

<response_to_analyze>
{response_text}
</response_to_analyze>

Use the following catalog as the authoritative classification reference:

<criterion_micro_pattern_catalog>
{catalog_text}
</criterion_micro_pattern_catalog>

{OUTPUT_FORMAT}
""".strip()


# ---------------------------------------------------------------------------
# OpenAI API
# ---------------------------------------------------------------------------

def call_openai_with_retry(
    client: OpenAI,
    model: str,
    analysis_prompt: str,
    prompt_id: str,
    max_retries: int,
) -> str:
    """Submit one claim-analysis request with retry handling."""
    for attempt in range(1, max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                instructions=SYSTEM_INSTRUCTIONS,
                input=analysis_prompt,
            )

            output_text = response.output_text

            if not output_text or not output_text.strip():
                raise RuntimeError(
                    f"The API returned no text for {prompt_id}."
                )

            return output_text.strip()

        except (
            RateLimitError,
            APIConnectionError,
            APITimeoutError,
        ) as error:
            if attempt == max_retries:
                raise

            wait_seconds = min(2 ** attempt, 60)

            logging.warning(
                "Temporary API error for %s on attempt %d/%d: %s. "
                "Retrying in %d seconds.",
                prompt_id,
                attempt,
                max_retries,
                error,
                wait_seconds,
            )

            time.sleep(wait_seconds)

        except APIStatusError as error:
            if error.status_code >= 500 and attempt < max_retries:
                wait_seconds = min(2 ** attempt, 60)

                logging.warning(
                    "OpenAI server error %s for %s. "
                    "Retrying in %d seconds.",
                    error.status_code,
                    prompt_id,
                    wait_seconds,
                )

                time.sleep(wait_seconds)
                continue

            raise

    raise RuntimeError(
        f"Unable to obtain a claim analysis for {prompt_id}."
    )


# ---------------------------------------------------------------------------
# Output handling
# ---------------------------------------------------------------------------

def save_text_atomically(
    output_path: Path,
    text: str,
) -> None:
    """Write a text file atomically to avoid incomplete output files."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = output_path.with_suffix(
        output_path.suffix + ".temporary"
    )

    temporary_path.write_text(
        text.rstrip() + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(output_path)


def save_error_report(
    prompt_id: str,
    response_path: Path,
    error: Exception,
) -> None:
    """Write an error report under claims/errors/."""
    ERRORS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    error_path = (
        ERRORS_DIRECTORY
        / f"{sanitize_filename(prompt_id)}.error.txt"
    )

    error_path.write_text(
        (
            f"Prompt ID: {prompt_id}\n"
            f"Response file: {response_path}\n"
            f"Error type: {type(error).__name__}\n"
            f"Error message: {error}\n"
        ),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main processing loop
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_arguments()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Set the environment variable "
            "before running this program."
        )

    if args.start_row < 1:
        raise ValueError(
            "--start-row must be at least 1."
        )

    if args.limit is not None and args.limit < 1:
        raise ValueError(
            "--limit must be at least 1."
        )

    if args.max_retries < 1:
        raise ValueError(
            "--max-retries must be at least 1."
        )

    response_extension = normalize_extension(
        args.response_extension
    )

    if args.output_extension:
        output_extension = normalize_extension(
            args.output_extension
        )
    else:
        output_extension = response_extension

    csv_path = resolve_csv_path(args.csv_file)

    validate_directories_and_files(csv_path)

    dataframe = load_csv(csv_path)
    catalog_text = load_catalog()

    start_index = args.start_row - 1
    dataframe = dataframe.iloc[start_index:]

    if args.limit is not None:
        dataframe = dataframe.head(args.limit)

    if dataframe.empty:
        logging.info(
            "No CSV rows matched the selected range."
        )
        return

    if args.save_input_prompts:
        AUDIT_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

    client = OpenAI()

    total = len(dataframe)
    completed = 0
    skipped = 0
    missing = 0
    failed = 0

    logging.info(
        "CSV file: %s",
        csv_path,
    )
    logging.info(
        "Catalog file: %s",
        CATALOG_FILE,
    )
    logging.info(
        "Responses directory: %s",
        RESPONSES_DIRECTORY,
    )
    logging.info(
        "Claims directory: %s",
        CLAIMS_DIRECTORY,
    )
    logging.info(
        "Processing %d records with model %s.",
        total,
        args.model,
    )

    for position, (_, row) in enumerate(
        dataframe.iterrows(),
        start=1,
    ):
        prompt_id = row[PROMPT_ID_COLUMN]
        original_prompt = row[PROMPT_COLUMN]

        safe_prompt_id = sanitize_filename(prompt_id)

        response_path = (
            RESPONSES_DIRECTORY
            / f"{safe_prompt_id}{response_extension}"
        )

        claim_output_path = (
            CLAIMS_DIRECTORY
            / f"{safe_prompt_id}{output_extension}"
        )

        if claim_output_path.exists() and not args.overwrite:
            skipped += 1

            logging.info(
                "[%d/%d] Skipping %s; claim file already exists.",
                position,
                total,
                prompt_id,
            )
            continue

        if not response_path.exists():
            missing += 1

            logging.warning(
                "[%d/%d] Missing response file for %s: %s",
                position,
                total,
                prompt_id,
                response_path,
            )
            continue

        logging.info(
            "[%d/%d] Classifying claims for %s",
            position,
            total,
            prompt_id,
        )

        try:
            response_text = read_response(response_path)

            analysis_prompt = build_analysis_prompt(
                prompt_id=prompt_id,
                original_prompt=original_prompt,
                response_text=response_text,
                catalog_text=catalog_text,
            )

            if args.save_input_prompts:
                audit_path = (
                    AUDIT_DIRECTORY
                    / f"{safe_prompt_id}.input.txt"
                )

                save_text_atomically(
                    output_path=audit_path,
                    text=analysis_prompt,
                )

            claim_analysis = call_openai_with_retry(
                client=client,
                model=args.model,
                analysis_prompt=analysis_prompt,
                prompt_id=prompt_id,
                max_retries=args.max_retries,
            )

            save_text_atomically(
                output_path=claim_output_path,
                text=claim_analysis,
            )

            completed += 1

            logging.info(
                "[%d/%d] Saved %s",
                position,
                total,
                claim_output_path,
            )

            if args.delay > 0:
                time.sleep(args.delay)

        except Exception as error:
            failed += 1

            logging.exception(
                "[%d/%d] Failed to classify %s",
                position,
                total,
                prompt_id,
            )

            save_error_report(
                prompt_id=prompt_id,
                response_path=response_path,
                error=error,
            )

    logging.info(
        (
            "Finished. Completed: %d | Skipped: %d | "
            "Missing responses: %d | Failed: %d"
        ),
        completed,
        skipped,
        missing,
        failed,
    )


if __name__ == "__main__":
    main()