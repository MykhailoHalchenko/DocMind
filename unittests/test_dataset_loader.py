import asyncio
import csv
import json
import tempfile
import unittest
from pathlib import Path

from unittests._support import configure_test_environment

configure_test_environment()

from dataset_loader import load_csv_dataset, load_json_dataset


class DatasetLoaderTests(unittest.TestCase):
    def test_load_json_dataset_supports_list_and_records_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            list_path = Path(temp_dir) / "list.json"
            list_path.write_text(json.dumps([{"text": "alpha"}, {"text": "beta"}]), encoding="utf-8")

            records_path = Path(temp_dir) / "records.json"
            records_path.write_text(json.dumps({"records": [{"text": "gamma"}]}), encoding="utf-8")

            list_data = asyncio.run(load_json_dataset(str(list_path)))
            records_data = asyncio.run(load_json_dataset(str(records_path)))

        self.assertEqual(len(list_data), 2)
        self.assertEqual(records_data, [{"text": "gamma"}])

    def test_load_csv_dataset_returns_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "sample.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["text", "source"])
                writer.writeheader()
                writer.writerow({"text": "delta", "source": "paper.csv"})

            rows = asyncio.run(load_csv_dataset(str(csv_path)))

        self.assertEqual(rows, [{"text": "delta", "source": "paper.csv"}])


if __name__ == "__main__":
    unittest.main()