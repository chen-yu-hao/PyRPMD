"""Offline regression fixtures for the documentation CI checker."""

from pathlib import Path
import subprocess
import tempfile
import unittest

from check_links import LinkChecker, parse_markdown


class LinkChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.checker = LinkChecker(self.root)

    def write(self, path, content):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return target

    def test_reference_images_fences_and_unicode_duplicate_anchors(self):
        self.write("guide.md", "# 中文 test\n## Run `job`!\n## Run `job`!\n")
        self.write("plot.svg", '<svg><g id="plot"/></svg>')
        source = self.write("README.md", "[go][guide]\n\n![plot](plot.svg#plot)\n\n[guide]: guide.md#run-job-1\n\n[中文](guide.md#中文-test)\n\n```md\n[ignored](missing.md)\n```\n")
        self.checker.check_page(source, self.root)
        self.assertEqual(self.checker.errors, [])
        self.assertEqual(self.checker.links_checked, 3)

    def test_broken_reference_and_fragment_fail(self):
        self.write("guide.md", "# Present\n")
        source = self.write("README.md", "[file](absent.md)\n\n[anchor](guide.md#absent)\n")
        self.checker.check_page(source, self.root)
        self.assertEqual(len(self.checker.errors), 2)
        self.assertIn("missing target", self.checker.errors[0])
        self.assertIn("missing fragment", self.checker.errors[1])

    def test_heading_suffixes_cannot_collide_with_literal_headings(self):
        self.write("guide.md", "## Foo\n## Foo-1\n## Foo\n")
        source = self.write("README.md", "[third](guide.md#foo-2)\n")
        self.checker.check_page(source, self.root)
        self.assertEqual(self.checker.errors, [])
        self.assertEqual(self.checker.page(self.root / "guide.md").anchors, {"foo", "foo-1", "foo-2"})

    def test_html_language_link_and_generated_anchors(self):
        self.write("html/index.html", '<a href="zh/index.html#install">中文</a><script src="_static/app.js"></script>')
        self.write("html/zh/index.html", '<h2 id="install">安装</h2><a href="../index.html">English</a>')
        self.write("html/_static/app.js", "// asset")
        self.write("html/_static/theme-macro.html", '<link href="{{ pathto(asset) }}">')
        self.assertEqual(self.checker.check_html(self.root / "html"), 2)
        self.assertEqual(self.checker.errors, [])
        (self.root / "html/zh/index.html").write_text('<h2 id="renamed">安装</h2>')
        another = LinkChecker(self.root)
        another.check_html(self.root / "html")
        self.assertTrue(any("missing fragment" in item for item in another.errors))

    def test_html_missing_asset_and_translated_page_fail(self):
        self.write("html/index.html", '<a href="zh/page.html">中文</a><img src="absent.svg">')
        self.checker.check_html(self.root / "html")
        self.assertEqual(len(self.checker.errors), 2)

    def test_main_urls_use_checkout_but_pinned_refs_are_live(self):
        self.write("docs/new.md", "# Added in this PR\n")
        source = self.write("README.md", "[new](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/new.md#added-in-this-pr)\n\n[pinned](https://github.com/chen-yu-hao/PyRPMD/blob/abc123/docs/old.md)\n")
        self.checker.check_page(source, self.root)
        self.assertEqual(self.checker.errors, [])
        self.assertEqual(len(self.checker.external), 1)
        self.assertIn("/abc123/", next(iter(self.checker.external)))

    def test_pinned_history_is_not_replaced_by_current_files_or_anchors(self):
        def git(*args):
            return subprocess.run(["git", *args], cwd=self.root, check=True, capture_output=True, text=True).stdout.strip()

        git("init", "-q")
        self.write("guide.md", "# Historical heading\n")
        git("add", "guide.md")
        git("-c", "user.name=Docs fixture", "-c", "user.email=docs@example.invalid", "commit", "-qm", "Historical fixture")
        commit = git("rev-parse", "HEAD")
        self.write("guide.md", "# Current heading\n")
        self.write("current-only.md", "# New file\n")
        prefix = f"https://github.com/chen-yu-hao/PyRPMD/blob/{commit}/"
        source = self.write("README.md", f"[old]({prefix}guide.md#historical-heading)\n\n[wrong anchor]({prefix}guide.md#current-heading)\n\n[wrong file]({prefix}current-only.md)\n\n[wrong line]({prefix}guide.md#L2)\n")
        self.checker.check_page(source, self.root)
        self.assertEqual(len(self.checker.errors), 3)
        self.assertEqual(self.checker.external, {})
        self.assertTrue(any("missing blob in exact commit" in item for item in self.checker.errors))
        self.assertTrue(any("current-heading" in item for item in self.checker.errors))

    def test_unavailable_full_sha_stays_in_live_inventory(self):
        url = "https://github.com/chen-yu-hao/PyRPMD/blob/" + "a" * 40 + "/guide.md"
        source = self.write("README.md", f"[unknown]({url})\n")
        self.checker.check_page(source, self.root)
        self.assertEqual(self.checker.errors, [])
        self.assertIn(url, self.checker.external)

    def test_readme_byte_budget_and_section_order(self):
        self.write("README.md", '<a id="install"></a>\n\n# Title\n## Install\n')
        self.write("README.zh-CN.md", '<a id="run"></a>\n\n# 标题\n## 安装\n' + "中" * 4000)
        self.checker.check_readmes()
        self.assertTrue(any("byte README budget" in item for item in self.checker.errors))
        self.assertTrue(any("section IDs/order" in item for item in self.checker.errors))

    def test_chapter_drift_is_an_error(self):
        self.write("docs/manual/new.md", "# New")
        self.write("docs/zh/manual/old.md", "# 旧")
        self.checker.check_chapters()
        self.assertEqual(len(self.checker.errors), 1)
        self.assertIn("new.md", self.checker.errors[0])
        self.assertIn("old.md", self.checker.errors[0])

    def test_new_guide_must_be_visible_in_both_indexes(self):
        self.write("docs/index.md", "# Manual\n\n## Guides\n\n[New](new.md)\n")
        self.write("docs/zh/index.md", "# 手册\n\n## 指南\n")
        self.checker.check_index_parity()
        self.assertTrue(any("English-only ['new.md']" in item for item in self.checker.errors))
        self.write("docs/zh/index.md", "# 手册\n\n## 指南\n\n[英文](https://github.com/chen-yu-hao/PyRPMD/blob/main/docs/new.md)\n")
        another = LinkChecker(self.root)
        another.check_index_parity()
        self.assertEqual(another.errors, [])

    def test_built_guide_must_be_reachable_from_index(self):
        self.write("docs/new.md", "# New")
        self.write("html/index.html", "<h1>Index</h1>")
        self.write("html/new.html", '<h1>New</h1><a href="index.html">Home</a>')
        self.checker.check_html(self.root / "html")
        self.assertTrue(any("not reachable" in item for item in self.checker.errors))
        self.write("html/index.html", '<a href="new.html">New guide</a>')
        another = LinkChecker(self.root)
        another.check_html(self.root / "html")
        self.assertEqual(another.errors, [])

    def test_external_project_deduplicates_targets_and_keeps_origins(self):
        source = self.write("README.md", "[a](https://example.org/page#one) [b](https://example.org/page#two)\n")
        self.checker.check_page(source, self.root)
        output = self.root / "external"
        self.checker.write_external_project(output)
        self.assertEqual(len(self.checker.external), 1)
        self.assertIn("https://example.org/page", (output / "index.rst").read_text())
        self.assertIn("README.md:1", (output / "sources.json").read_text())
        self.assertNotIn("linkcheck_ignore", (output / "conf.py").read_text())


if __name__ == "__main__":
    unittest.main()
