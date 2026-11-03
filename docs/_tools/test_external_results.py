import unittest

from check_external_results import failures


class ExternalEvidence(unittest.TestCase):
    def test_service_outage_and_unfinished_url_do_not_silently_pass(self):
        sources = {"https://example.org/503": ["docs/page.html:1"], "https://example.org/timeout": ["README.md:2"]}
        results = [{"uri": "https://example.org/503", "status": "ignored", "info": "service unavailable"}]
        errors = failures(results, sources)
        self.assertEqual(len(errors), 2)
        self.assertIn("ignored: service unavailable", errors[0])
        self.assertIn("no completed check", errors[1])

    def test_successful_redirect_is_verified(self):
        sources = {"https://doi.org/example": ["docs/page.html:1"]}
        results = [{"uri": "https://doi.org/example", "status": "redirected", "info": "https://registry.example/metadata"}]
        self.assertEqual(failures(results, sources), [])


if __name__ == "__main__":
    unittest.main()
