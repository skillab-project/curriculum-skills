import json
import unittest
from unittest.mock import patch

from main import TopSkillsRequest

class TestTopSkills(unittest.TestCase):

    @patch("main.get_top_skills")
    def test_compare_top_skills(self, mock_get_top_skills):
        with open("tests/json/top_skills_cambridge_expected.json", "r", encoding="utf-8") as f:
            expected = json.load(f)

        mock_get_top_skills.return_value = {
            "top_skills": expected["top_skills"]
        }

        from main import get_top_skills
        response = get_top_skills(TopSkillsRequest(university_name="University of Cambridge", top_n=50))
        actual = response["top_skills"]

        self.assertEqual(len(actual), len(expected["top_skills"]), f"Expected {len(expected['top_skills'])}, got {len(actual)}")

        for expected_skill, actual_skill in zip(expected["top_skills"], actual):
            self.assertEqual(actual_skill["skill"], expected_skill["skill"])
            self.assertEqual(actual_skill["frequency"], expected_skill["frequency"])

        matched_count = sum(
            1 for e, a in zip(expected["top_skills"], actual)
            if e["skill"] == a["skill"] and e["frequency"] == a["frequency"]
        )

        print("\n✅ Top Skills Matching Report")
        print(f"✔ Total top skills matched: {matched_count} / {len(expected['top_skills'])}")
        print("\n📋 Sample of actual top skills:")
        for skill in actual[:5]:
            print(f"- {skill['skill']} (freq: {skill['frequency']})")


if __name__ == "__main__":
    unittest.main()
