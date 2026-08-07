"""
Run prompts from a CSV file through the OpenAI API.

Expected CSV columns:
    Prompt ID
    Prompt

Example output files:
    results/JH-001.txt
    results/JH-002.txt
    results/JH-003.txt

Installation:
    pip install --upgrade openai pandas

Environment variables:
    Windows PowerShell:
        $env:OPENAI_API_KEY="your-api-key"
        $env:OPENAI_MODEL="gpt-5.6"

    Windows Command Prompt:
        set OPENAI_API_KEY=your-api-key
        set OPENAI_MODEL=gpt-5.6

    macOS/Linux:
        export OPENAI_API_KEY="your-api-key"
        export OPENAI_MODEL="gpt-5.6"
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


PROMPT_ID_COLUMN = "Prompt ID"
PROMPT_COLUMN = "Prompt"

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6")
DEFAULT_SYSTEM_INSTRUCTIONS = """
Answer the supplied prompt accurately and completely.

Use clear, professional language appropriate for art-historical and educational
research. Distinguish documented facts from interpretation. Do not invent
artworks, dates, biographical details, quotations, exhibitions, or sources.
When the prompt requests analysis or evaluation, provide an explicit claim
supported by relevant evidence.
""".strip()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read prompts from a CSV file, submit them to the OpenAI API, "
            "and save each response in a separate file."
        )
    )

    parser.add_argument(
        "csv_file",
        type=Path,
        help="Path to the CSV file containing Prompt ID and Prompt columns.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory in which response files will be saved. Default: results",
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use. Default: {DEFAULT_MODEL}",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate responses even when an output file already exists.",
    )

    parser.add_argument(
        "--start-row",
        type=int,
        default=1,
        help="First data row to process, using 1-based numbering. Default: 1",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of prompts to process.",
    )

    parser.add_argument(
        "--max-retries",
        type=int,
        default=6,
        help="Maximum API attempts for each prompt. Default: 6",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to pause between successful requests. Default: 0.5",
    )

    parser.add_argument(
        "--extension",
        choices=["txt", "md"],
        default="txt",
        help="Output file extension. Default: txt",
    )

    return parser.parse_args()


def sanitize_filename(value: object) -> str:
    """
    Convert a prompt ID into a safe filename.

    Examples:
        JH-001      -> JH-001
        Prompt 1/2  -> Prompt_1_2
    """
    text = str(value).strip()

    # Replace characters prohibited or problematic in filenames.
    text = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", text)

    # Collapse repeated whitespace and underscores.
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)

    # Avoid trailing dots or spaces on Windows.
    text = text.strip(" ._")

    return text or "unnamed_prompt"


def load_prompts(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file was not found: {csv_path}")

    # utf-8-sig handles ordinary UTF-8 files and files exported with a BOM.
    dataframe = pd.read_csv(csv_path, encoding="utf-8-sig")

    # Normalize column headings to avoid problems caused by extra spaces.
    dataframe.columns = [
        str(column).strip() for column in dataframe.columns
    ]

    required_columns = {PROMPT_ID_COLUMN, PROMPT_COLUMN}
    missing_columns = required_columns - set(dataframe.columns)

    if missing_columns:
        available = ", ".join(map(str, dataframe.columns))
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"Missing required CSV column(s): {missing}. "
            f"Available columns: {available}"
        )

    # Remove rows without a usable ID or prompt.
    dataframe = dataframe.dropna(
        subset=[PROMPT_ID_COLUMN, PROMPT_COLUMN]
    ).copy()

    dataframe[PROMPT_ID_COLUMN] = (
        dataframe[PROMPT_ID_COLUMN].astype(str).str.strip()
    )
    dataframe[PROMPT_COLUMN] = (
        dataframe[PROMPT_COLUMN].astype(str).str.strip()
    )

    dataframe = dataframe[
        (dataframe[PROMPT_ID_COLUMN] != "")
        & (dataframe[PROMPT_COLUMN] != "")
    ]

    duplicate_ids = dataframe[
        dataframe[PROMPT_ID_COLUMN].duplicated(keep=False)
    ][PROMPT_ID_COLUMN].tolist()

    if duplicate_ids:
        duplicate_list = ", ".join(sorted(set(duplicate_ids))[:10])
        raise ValueError(
            "Duplicate Prompt ID values were found. Each output file must "
            f"have a unique ID. Examples: {duplicate_list}"
        )

    return dataframe.reset_index(drop=True)


def call_openai_with_retry(
    client: OpenAI,
    model: str,
    prompt: str,
    prompt_id: str,
    max_retries: int,
) -> str:
    """
    Submit one prompt, retrying temporary API failures with exponential backoff.
    """
    for attempt in range(1, max_retries + 1):
        try:
            response = client.responses.create(
                model=model,
                instructions=DEFAULT_SYSTEM_INSTRUCTIONS,
                input=prompt,
            )

            output_text = response.output_text

            if not output_text or not output_text.strip():
                raise RuntimeError(
                    f"The API returned no text for prompt {prompt_id}."
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
            # Retry server-side errors but not most client-side errors.
            status_code = error.status_code

            if status_code >= 500 and attempt < max_retries:
                wait_seconds = min(2 ** attempt, 60)

                logging.warning(
                    "OpenAI server error %s for %s. Retrying in %d seconds.",
                    status_code,
                    prompt_id,
                    wait_seconds,
                )

                time.sleep(wait_seconds)
                continue

            raise

    raise RuntimeError(
        f"Unable to obtain a response for prompt {prompt_id}."
    )


def save_response(
    output_path: Path,
    response_text: str,
) -> None:
    """
    Save atomically so an interrupted write does not leave a partial result.
    """
    temporary_path = output_path.with_suffix(
        output_path.suffix + ".temporary"
    )

    temporary_path.write_text(
        response_text + "\n",
        encoding="utf-8",
    )

    temporary_path.replace(output_path)


def save_error(
    error_directory: Path,
    prompt_id: str,
    prompt: str,
    error: Exception,
) -> None:
    error_directory.mkdir(parents=True, exist_ok=True)

    error_path = error_directory / f"{sanitize_filename(prompt_id)}.error.txt"

    error_path.write_text(
        (
            f"Prompt ID: {prompt_id}\n\n"
            f"Prompt:\n{prompt}\n\n"
            f"Error type: {type(error).__name__}\n"
            f"Error message: {error}\n"
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_arguments()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Store your API key in the "
            "OPENAI_API_KEY environment variable before running this script."
        )

    if args.start_row < 1:
        raise ValueError("--start-row must be at least 1.")

    if args.limit is not None and args.limit < 1:
        raise ValueError("--limit must be at least 1.")

    dataframe = load_prompts(args.csv_file)

    # Convert the user's 1-based row number to a zero-based DataFrame position.
    start_index = args.start_row - 1
    dataframe = dataframe.iloc[start_index:]

    if args.limit is not None:
        dataframe = dataframe.head(args.limit)

    if dataframe.empty:
        logging.info("No prompts met the selected row range.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    error_directory = args.output_dir / "errors"

    client = OpenAI()

    total = len(dataframe)
    completed = 0
    skipped = 0
    failed = 0

    logging.info(
        "Processing %d prompts using model %s.",
        total,
        args.model,
    )

    for sequence, (_, row) in enumerate(
        dataframe.iterrows(),
        start=1,
    ):
        prompt_id = row[PROMPT_ID_COLUMN]
        prompt = row[PROMPT_COLUMN]

        filename = (
            f"{sanitize_filename(prompt_id)}.{args.extension}"
        )
        output_path = args.output_dir / filename

        if output_path.exists() and not args.overwrite:
            skipped += 1
            logging.info(
                "[%d/%d] Skipping %s; output already exists.",
                sequence,
                total,
                prompt_id,
            )
            continue

        logging.info(
            "[%d/%d] Processing %s",
            sequence,
            total,
            prompt_id,
        )

        try:
            response_text = call_openai_with_retry(
                client=client,
                model=args.model,
                prompt=prompt,
                prompt_id=prompt_id,
                max_retries=args.max_retries,
            )

            save_response(
                output_path=output_path,
                response_text=response_text,
            )

            completed += 1

            logging.info(
                "[%d/%d] Saved %s",
                sequence,
                total,
                output_path,
            )

            if args.delay > 0:
                time.sleep(args.delay)

        except Exception as error:
            failed += 1

            logging.exception(
                "[%d/%d] Failed to process %s",
                sequence,
                total,
                prompt_id,
            )

            save_error(
                error_directory=error_directory,
                prompt_id=prompt_id,
                prompt=prompt,
                error=error,
            )

    logging.info(
        "Finished. Completed: %d | Skipped: %d | Failed: %d",
        completed,
        skipped,
        failed,
    )


if __name__ == "__main__":
    main()
