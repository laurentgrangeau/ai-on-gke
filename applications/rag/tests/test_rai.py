# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

# Add container directory to path
CONTAINER_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../frontend/container")
)
if CONTAINER_DIR not in sys.path:
    sys.path.insert(0, CONTAINER_DIR)


class TestNlpFilter(unittest.TestCase):

    def setUp(self):
        # Mock Client instantiations before importing nlp_filter
        self.language_patcher = patch("google.cloud.language_v1.LanguageServiceClient")
        self.mock_language_client_cls = self.language_patcher.start()
        self.mock_language_client = MagicMock()
        self.mock_language_client_cls.return_value = self.mock_language_client

        # Also patch dlp_client to prevent DefaultCredentialsError if dlp is loaded
        self.dlp_patcher = patch("google.cloud.dlp_v2.DlpServiceClient")
        self.mock_dlp_client_cls = self.dlp_patcher.start()

        # Import nlp_filter module
        if "rai.nlp_filter" in sys.modules:
            del sys.modules["rai.nlp_filter"]
        import rai.nlp_filter as nlp_filter
        self.nlp_filter = nlp_filter

    def tearDown(self):
        self.language_patcher.stop()
        self.dlp_patcher.stop()

    def test_is_nlp_api_enabled_when_parent_is_null(self):
        with patch.object(self.nlp_filter, "parent", "NULL"):
            self.assertFalse(self.nlp_filter.is_nlp_api_enabled())

    def test_is_nlp_api_enabled_success(self):
        with patch.object(self.nlp_filter, "parent", "projects/test-project"):
            with patch.object(self.nlp_filter, "sum_moderation_confidences", return_value=10):
                self.assertTrue(self.nlp_filter.is_nlp_api_enabled())

    def test_is_nlp_api_enabled_exception(self):
        with patch.object(self.nlp_filter, "parent", "projects/test-project"):
            with patch.object(self.nlp_filter, "sum_moderation_confidences", side_effect=Exception("API Error")):
                self.assertFalse(self.nlp_filter.is_nlp_api_enabled())

    def test_sum_moderation_confidences(self):
        # Mock moderation categories
        cat1 = MagicMock()
        cat1.name = "Toxic"
        cat1.confidence = 0.85

        cat2 = MagicMock()
        cat2.name = "Health"  # In excluding_names ["Health", "Politics", "Finance", "Legal"]
        cat2.confidence = 0.99

        cat3 = MagicMock()
        cat3.name = "Insult"
        cat3.confidence = 0.40

        mock_response = MagicMock()
        mock_response.moderation_categories = [cat1, cat2, cat3]
        self.mock_language_client.moderate_text.return_value = mock_response

        score = self.nlp_filter.sum_moderation_confidences("some test text")
        self.assertEqual(score, 85)

    def test_is_content_inappropriate_true(self):
        with patch.object(self.nlp_filter, "sum_moderation_confidences", return_value=80):
            # nlp_filter_level = 30 -> threshold is 100 - 30 = 70. 80 > 70 => True
            self.assertTrue(self.nlp_filter.is_content_inappropriate("test", 30))

    def test_is_content_inappropriate_false(self):
        with patch.object(self.nlp_filter, "sum_moderation_confidences", return_value=50):
            # nlp_filter_level = 30 -> threshold is 100 - 30 = 70. 50 > 70 => False
            self.assertFalse(self.nlp_filter.is_content_inappropriate("test", 30))


class TestDlpFilter(unittest.TestCase):

    def setUp(self):
        self.dlp_patcher = patch("google.cloud.dlp_v2.DlpServiceClient")
        self.mock_dlp_client_cls = self.dlp_patcher.start()
        self.mock_dlp_client = MagicMock()
        self.mock_dlp_client_cls.return_value = self.mock_dlp_client

        self.language_patcher = patch("google.cloud.language_v1.LanguageServiceClient")
        self.mock_language_client_cls = self.language_patcher.start()

        if "rai.dlp_filter" in sys.modules:
            del sys.modules["rai.dlp_filter"]
        import rai.dlp_filter as dlp_filter
        self.dlp_filter = dlp_filter

    def tearDown(self):
        self.dlp_patcher.stop()
        self.language_patcher.stop()

    def test_is_dlp_api_enabled_when_parent_is_null(self):
        with patch.object(self.dlp_filter, "parent", "NULL"):
            self.assertFalse(self.dlp_filter.is_dlp_api_enabled())

    def test_is_dlp_api_enabled_success(self):
        with patch.object(self.dlp_filter, "parent", "projects/test-project"):
            self.mock_dlp_client.list_info_types.return_value = MagicMock()
            self.assertTrue(self.dlp_filter.is_dlp_api_enabled())

    def test_is_dlp_api_enabled_exception(self):
        with patch.object(self.dlp_filter, "parent", "projects/test-project"):
            self.mock_dlp_client.list_info_types.side_effect = Exception("API Error")
            self.assertFalse(self.dlp_filter.is_dlp_api_enabled())

    def test_list_inspect_templates_from_parent(self):
        tmpl1 = MagicMock()
        tmpl1.name = "projects/test-project/inspectTemplates/template1"
        tmpl2 = MagicMock()
        tmpl2.name = "projects/test-project/inspectTemplates/template2"
        self.mock_dlp_client.list_inspect_templates.return_value = [tmpl1, tmpl2]

        result = self.dlp_filter.list_inspect_templates_from_parent()
        self.assertEqual(result, [
            "projects/test-project/inspectTemplates/template1",
            "projects/test-project/inspectTemplates/template2"
        ])

    def test_get_inspect_templates_from_name(self):
        mock_template = MagicMock()
        self.mock_dlp_client.get_inspect_template.return_value = mock_template

        result = self.dlp_filter.get_inspect_templates_from_name("projects/test-project/inspectTemplates/template1")
        self.assertEqual(result, mock_template)

    def test_list_deidentify_templates_from_parent(self):
        tmpl1 = MagicMock()
        tmpl1.name = "projects/test-project/deidentifyTemplates/template1"
        self.mock_dlp_client.list_deidentify_templates.return_value = [tmpl1]

        result = self.dlp_filter.list_deidentify_templates_from_parent()
        self.assertEqual(result, ["projects/test-project/deidentifyTemplates/template1"])

    def test_get_deidentify_templates_from_name(self):
        mock_template = MagicMock()
        self.mock_dlp_client.get_deidentify_template.return_value = mock_template

        result = self.dlp_filter.get_deidentify_templates_from_name("projects/test-project/deidentifyTemplates/template1")
        self.assertEqual(result, mock_template)

    def test_inspect_content(self):
        inspect_template = MagicMock()
        inspect_template.inspect_config = "mock_inspect_config"
        deidentify_template = MagicMock()
        deidentify_template.deidentify_config = "mock_deidentify_config"

        mock_deidentify_response = MagicMock()
        mock_deidentify_response.item.value = "redacted text"
        self.mock_dlp_client.deidentify_content.return_value = mock_deidentify_response

        with patch.object(self.dlp_filter, "get_inspect_templates_from_name", return_value=inspect_template), \
             patch.object(self.dlp_filter, "get_deidentify_templates_from_name", return_value=deidentify_template):
            res = self.dlp_filter.inspect_content("inspect_path", "deidentify_path", "sensitive text")
            self.assertEqual(res, "redacted text")


if __name__ == "__main__":
    unittest.main()
