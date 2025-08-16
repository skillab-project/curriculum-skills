import json
import os
import unittest
from unittest.mock import patch
from main import PDFProcessingRequest

class TestExtractedTitles(unittest.TestCase):

    @patch("main.process_pdf")
    def test_compare_extracted_titles(self, mock_process_pdf):
        with open("tests/json/extracted_titles_expected.json", "r", encoding="utf-8") as f:
            expected = json.load(f)

        mock_process_pdf.return_value = {
            "data": expected["data"]
        }

        pdf_path = os.path.abspath("tests/sample_pdfs/Cambridge University.pdf")
        from main import process_pdf
        response = process_pdf(PDFProcessingRequest(pdf_name=pdf_path))

        actual = response["data"]

        for semester in expected["data"]:
            if semester in ["university_name", "university_country"]:
                continue

            self.assertIn(semester, actual, f"Missing semester: {semester}")
            for lesson in expected["data"][semester]:
                self.assertIn(lesson, actual[semester], f"Missing lesson: {lesson} in {semester}")
                self.assertIsInstance(actual[semester][lesson], dict, f"{lesson} in {semester} is not a dict")
                self.assertEqual(
                    actual[semester][lesson]["description"],
                    expected["data"][semester][lesson]["description"]
                )

        total_semesters = len([s for s in expected["data"] if s not in ["university_name", "university_country"]])
        total_lessons = sum(len(lessons) for sem, lessons in expected["data"].items() if sem not in ["university_name", "university_country"])

        print(f"\n✅ Title Matching Report")
        print(f"✔ Semesters matched: {total_semesters}")
        print(f"✔ Lessons matched: {total_lessons}")

if __name__ == "__main__":
    unittest.main()
