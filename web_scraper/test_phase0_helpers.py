import unittest

from enricher import normalize_sifra
from mapper import Mapper


class NormalizeSifraTests(unittest.TestCase):
    def test_integer_float_string(self):
        self.assertEqual(normalize_sifra("9275.0"), "9275")
        self.assertEqual(normalize_sifra(9275), "9275")
        self.assertEqual(normalize_sifra(9275.0), "9275")

    def test_non_numeric_code(self):
        self.assertEqual(normalize_sifra("S-0759"), "S-0759")

    def test_blank_and_placeholder(self):
        self.assertIsNone(normalize_sifra(""))
        self.assertIsNone(normalize_sifra("sifra"))
        self.assertIsNone(normalize_sifra(None))


class MapperValidationTests(unittest.TestCase):
    def test_missing_columns_raise(self):
        mapping = {
            "node": {"node_serial": "sifra"},
            "sensors": [
                {
                    "measurements": [
                        {"timestamp_utc": "Datum", "value": "vodostaj (cm)"}
                    ]
                }
            ],
        }
        mapper = Mapper(mapping)
        with self.assertRaises(ValueError):
            mapper.validate_source_columns(
                records=[{"Datum": "01.01.1994"}],
                headers=["Datum"],
            )

    def test_present_columns_pass(self):
        mapping = {
            "node": {"node_serial": "sifra"},
            "sensors": [
                {
                    "measurements": [
                        {"timestamp_utc": "Datum", "value": "vodostaj (cm)"}
                    ],
                    "metadata": {"sifra": "sifra"},
                }
            ],
        }
        mapper = Mapper(mapping)
        mapper.validate_source_columns(
            records=[{"Datum": "01.01.1994", "vodostaj (cm)": "76", "sifra": "9275"}],
            headers=["Datum", "vodostaj (cm)", "sifra"],
        )


class CSVExtractorSparseRowTests(unittest.TestCase):
    def test_sparse_rows_do_not_crash(self):
        from extractors.csv_extractor import CSVExtractor

        csv_bytes = b"Datum;vodostaj (cm)\n01.01.1994;\n02.01.1994;76\n"
        rows = CSVExtractor().extract(csv_bytes, ";")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["vodostaj (cm)"], "")
        self.assertEqual(rows[1]["vodostaj (cm)"], "76")


if __name__ == "__main__":
    unittest.main()
