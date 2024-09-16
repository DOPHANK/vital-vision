import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/mapper')))
import unittest
from easyocr_mapper import EasyOCRMapper

class TestEasyOCRMapper(unittest.TestCase):
    def setUp(self):
        self.mapper = EasyOCRMapper()

    def test_extract_ocr_properties(self):
        # Sample input data
        ocr_result = [
            ([[454, 214], [629, 214], [629, 290], [454, 290]], 'HSR', 0.9931),
            ([[664, 222], [925, 222], [925, 302], [664, 302]], 'Station', 0.3260)
        ]

        # Expected output
        expected_output = [
            {
                "description": "HSR",
                "boundingPoly": {
                    "vertices": [
                        {"x": 454, "y": 214},
                        {"x": 629, "y": 214},
                        {"x": 629, "y": 290},
                        {"x": 454, "y": 290}
                    ]
                },
                "confidence": 0.9931
            },
            {
                "description": "Station",
                "boundingPoly": {
                    "vertices": [
                        {"x": 664, "y": 222},
                        {"x": 925, "y": 222},
                        {"x": 925, "y": 302},
                        {"x": 664, "y": 302}
                    ]
                },
                "confidence": 0.3260
            }
        ]

        # Call the method and compare the result
        result = self.mapper.extract_ocr_properties(ocr_result)
        self.assertEqual(result, expected_output)

    def test_empty_input(self):
        # Test with empty input
        ocr_result = []
        result = self.mapper.extract_ocr_properties(ocr_result)
        self.assertEqual(result, [])

    def test_invalid_input(self):
        # Test with invalid input
        ocr_result = [("invalid", "data")]
        with self.assertRaises(ValueError):
            self.mapper.extract_ocr_properties(ocr_result)

if __name__ == '__main__':
    unittest.main()
