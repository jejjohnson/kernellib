"""Tests for ``scripts/build_docs.py``.

The documentation pipeline stitches two generators' output into one site, so
the pieces that repair and verify cross-half links are the ones most likely to
break silently. They are pure functions over HTML strings and a directory
tree, which makes them straightforward to pin down here.
"""

from __future__ import annotations

import importlib.util
import zlib
from pathlib import Path
from types import ModuleType

import pytest


def _load() -> ModuleType:
    path = Path(__file__).resolve().parent.parent / "scripts" / "build_docs.py"
    spec = importlib.util.spec_from_file_location("build_docs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_docs = _load()


def make_inventory(entries: list[str]) -> bytes:
    """Build a minimal Sphinx v2 inventory payload from raw entry lines."""
    header = (
        b"# Sphinx inventory version 2\n"
        b"# Project: kernellib\n"
        b"# Version: 0.0.0\n"
        b"# The remainder of this file is compressed using zlib.\n"
    )
    body = "\n".join(entries).encode("utf-8") + b"\n"
    return header + zlib.compress(body)


class TestParseInventory:
    def test_reads_entries(self) -> None:
        data = make_inventory(
            [
                "kernellib.Clip py:class 2 api/transforms/#kernellib.transforms.Clip -",
                "kernellib.summarize py:function 2 api/stats/#kernellib.stats.summarize"
                " -",
            ]
        )
        entries = build_docs.parse_inventory(data)
        assert entries["kernellib.Clip"] == "api/transforms/#kernellib.transforms.Clip"
        assert len(entries) == 2

    def test_expands_dollar_abbreviation(self) -> None:
        data = make_inventory(["kernellib.stats.Summary py:class 1 api/stats/#$ -"])
        entries = build_docs.parse_inventory(data)
        assert entries["kernellib.stats.Summary"] == (
            "api/stats/#kernellib.stats.Summary"
        )

    def test_rejects_unknown_version(self) -> None:
        with pytest.raises(ValueError, match="unsupported inventory header"):
            build_docs.parse_inventory(b"# Sphinx inventory version 1\n")

    def test_matches_the_real_inventory_format(self) -> None:
        """Guards against drift in what mkdocstrings actually emits."""
        real = Path(__file__).resolve().parent.parent / "site" / "objects.inv"
        if not real.is_file():
            pytest.skip("run `mkdocs build` first")
        entries = build_docs.parse_inventory(real.read_bytes())
        assert "kernellib.RBF" in entries
        assert all("$" not in uri for uri in entries.values())


class TestAnchorCaseMap:
    def test_maps_lowercased_to_true_spelling(self) -> None:
        entries = {"a": "api/stats/#kernellib.stats.RunningStats"}
        mapping = build_docs.anchor_case_map(entries)
        assert mapping["kernellib.stats.runningstats"] == (
            "kernellib.stats.RunningStats"
        )

    def test_omits_already_lowercase_anchors(self) -> None:
        entries = {"a": "api/stats/#kernellib.stats.summarize"}
        assert build_docs.anchor_case_map(entries) == {}

    def test_omits_ambiguous_anchors(self) -> None:
        """Two symbols colliding under lowercasing must not be guessed at."""
        entries = {
            "a": "api/x/#kernellib.Foo",
            "b": "api/y/#kernellib.FOO",
        }
        assert build_docs.anchor_case_map(entries) == {}

    def test_ignores_entries_without_anchors(self) -> None:
        assert build_docs.anchor_case_map({"a": "api/stats/"}) == {}


class TestRestoreAnchorCase:
    def test_repairs_a_lowercased_anchor(self) -> None:
        mapping = {"kernellib.stats.runningstats": "kernellib.stats.RunningStats"}
        html = '<a href="/reference/api/stats/#kernellib.stats.runningstats">x</a>'
        fixed, count = build_docs.restore_anchor_case(html, mapping)
        assert count == 1
        assert "#kernellib.stats.RunningStats" in fixed

    def test_leaves_unknown_anchors_alone(self) -> None:
        html = '<a href="/guide/#some-heading">x</a>'
        fixed, count = build_docs.restore_anchor_case(html, {})
        assert (fixed, count) == (html, 0)

    def test_leaves_fragmentless_links_alone(self) -> None:
        html = '<a href="/reference/api/stats/">x</a>'
        assert build_docs.restore_anchor_case(html, {"a": "A"}) == (html, 0)

    def test_repairs_every_occurrence(self) -> None:
        mapping = {"kernellib.clip": "kernellib.Clip"}
        html = '<a href="a#kernellib.clip">1</a><a href="b#kernellib.clip">2</a>'
        fixed, count = build_docs.restore_anchor_case(html, mapping)
        assert count == 2
        assert "kernellib.clip" not in fixed


class TestRewriteApiOrigin:
    def test_replaces_the_build_time_origin(self) -> None:
        html = '<a href="http://127.0.0.1:8910/api/stats/#x">x</a>'
        fixed, count = build_docs.rewrite_api_origin(
            html, "http://127.0.0.1:8910/", "/reference/"
        )
        assert count == 1
        assert fixed == '<a href="/reference/api/stats/#x">x</a>'

    def test_is_a_no_op_when_absent(self) -> None:
        html = '<a href="/guide/">x</a>'
        assert build_docs.rewrite_api_origin(html, "http://127.0.0.1:8910/", "/r/") == (
            html,
            0,
        )

    def test_honours_a_base_url_prefix(self) -> None:
        html = '<a href="http://127.0.0.1:8910/api/">x</a>'
        fixed, _ = build_docs.rewrite_api_origin(
            html, "http://127.0.0.1:8910/", "/kernellib/reference/"
        )
        assert fixed == '<a href="/kernellib/reference/api/">x</a>'


class TestVerifyLinks:
    @staticmethod
    def build_site(root: Path, pages: dict[str, str]) -> None:
        for name, body in pages.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8")

    def test_accepts_a_sound_site(self, tmp_path: Path) -> None:
        self.build_site(
            tmp_path,
            {
                "index.html": '<a href="reference/stats/#kernellib.Clip">api</a>',
                "reference/stats/index.html": '<h3 id="kernellib.Clip">Clip</h3>',
            },
        )
        assert build_docs.verify_links(tmp_path) == []

    def test_flags_a_dead_link(self, tmp_path: Path) -> None:
        self.build_site(tmp_path, {"index.html": '<a href="nope/">gone</a>'})
        problems = build_docs.verify_links(tmp_path)
        assert len(problems) == 1
        assert "dead link" in problems[0]

    def test_flags_a_missing_anchor(self, tmp_path: Path) -> None:
        """The exact failure mystmd's lowercasing produces."""
        self.build_site(
            tmp_path,
            {
                "index.html": '<a href="reference/stats/#kernellib.clip">api</a>',
                "reference/stats/index.html": '<h3 id="kernellib.Clip">Clip</h3>',
            },
        )
        problems = build_docs.verify_links(tmp_path)
        assert len(problems) == 1
        assert "missing anchor" in problems[0]

    def test_accepts_anchors_declared_by_name(self, tmp_path: Path) -> None:
        self.build_site(
            tmp_path,
            {
                "index.html": '<a href="t.html#here">x</a>',
                "t.html": '<a name="here"></a>',
            },
        )
        assert build_docs.verify_links(tmp_path) == []

    def test_ignores_external_and_same_page_links(self, tmp_path: Path) -> None:
        self.build_site(
            tmp_path,
            {
                "index.html": (
                    '<a href="https://example.com/x">e</a>'
                    '<a href="mailto:a@b.c">m</a>'
                    '<a href="#top">t</a>'
                )
            },
        )
        assert build_docs.verify_links(tmp_path) == []

    def test_strips_the_base_url_prefix(self, tmp_path: Path) -> None:
        self.build_site(
            tmp_path,
            {
                "index.html": '<a href="/proj/reference/stats/#a">api</a>',
                "reference/stats/index.html": '<h3 id="a">A</h3>',
            },
        )
        assert build_docs.verify_links(tmp_path, base_url="/proj") == []
        assert build_docs.verify_links(tmp_path) != []

    def test_resolves_directory_urls_to_index_html(self, tmp_path: Path) -> None:
        self.build_site(
            tmp_path,
            {
                "index.html": '<a href="guide/quickstart/">q</a>',
                "guide/quickstart/index.html": "<p>hi</p>",
            },
        )
        assert build_docs.verify_links(tmp_path) == []

    def test_rejects_links_escaping_the_site_root(self, tmp_path: Path) -> None:
        self.build_site(tmp_path, {"index.html": '<a href="../../etc/passwd">x</a>'})
        problems = build_docs.verify_links(tmp_path)
        assert len(problems) == 1
        assert "dead link" in problems[0]


class TestDollarAbbreviationRepair:
    """End-to-end cover for the mystmd `$`-expansion bug.

    mystmd lowercases an object's name when expanding a `$` anchor, so an
    inventory entry recorded as `stats/#$` yields a broken link. The inventory
    itself holds the correct spelling, which is what makes the repair exact.
    """

    def test_repairs_a_dollar_expanded_anchor(self) -> None:
        data = make_inventory(["kernellib.stats.Summary py:class 1 stats/#$ -"])
        mapping = build_docs.anchor_case_map(build_docs.parse_inventory(data))
        html = '<a href="/reference/stats/#kernellib.stats.summary">Summary</a>'
        fixed, count = build_docs.restore_anchor_case(html, mapping)
        assert count == 1
        assert "#kernellib.stats.Summary" in fixed

    def test_explicit_anchors_need_no_repair(self) -> None:
        """The form mkdocstrings uses for top-level re-exports."""
        data = make_inventory(
            ["kernellib.Summary py:class 2 stats/#kernellib.stats.Summary -"]
        )
        mapping = build_docs.anchor_case_map(build_docs.parse_inventory(data))
        html = '<a href="/reference/stats/#kernellib.stats.Summary">Summary</a>'
        assert build_docs.restore_anchor_case(html, mapping) == (html, 0)


class TestNavBaseUrlProblems:
    """Guards the site-nav regression that shipped in the first deployment.

    The MyST theme re-renders the site nav from the config it embeds for
    hydration and prepends ``BASE_URL`` to any URL starting with ``/``. The
    build script used to rewrite the nav entry to a root-relative path, so the
    deployed button pointed at ``/project/project/reference/`` and 404s — while
    the static HTML looked perfectly correct, which is why `verify_links` could
    not see it.
    """

    def test_flags_a_root_relative_nav_url(self) -> None:
        html = (
            '<script>{"nav":[{"title":"API","url":"/kernellib/reference/"}]}</script>'
        )
        assert build_docs.nav_base_url_problems(html) == ["/kernellib/reference/"]

    def test_accepts_an_absolute_nav_url(self) -> None:
        html = (
            '<script>{"nav":[{"title":"API",'
            '"url":"https://example.com/reference/"}]}</script>'
        )
        assert build_docs.nav_base_url_problems(html) == []

    def test_accepts_a_page_without_nav(self) -> None:
        assert build_docs.nav_base_url_problems("<html><body>hi</body></html>") == []

    def test_deduplicates_across_repeated_config_blocks(self) -> None:
        """The theme embeds the config more than once per page."""
        block = '{"nav":[{"title":"API","url":"/proj/reference/"}]}'
        assert build_docs.nav_base_url_problems(block + block) == ["/proj/reference/"]

    def test_ignores_urls_outside_the_nav_block(self) -> None:
        html = '<a href="/proj/reference/">x</a><script>{"nav":[]}</script>'
        assert build_docs.nav_base_url_problems(html) == []

    def test_the_real_nav_url_is_absolute(self) -> None:
        """`docs/myst.yml` must keep the nav URL absolute."""
        config = (
            Path(__file__).resolve().parent.parent / "docs" / "myst.yml"
        ).read_text(encoding="utf-8")
        nav = config.split("nav:", 1)[1]
        url = nav.split("url:", 1)[1].split("\n", 1)[0].strip()
        assert url.startswith("http"), f"nav url must be absolute, got {url!r}"


def test_port_matches_the_myst_config() -> None:
    """`docs/myst.yml` and this script must agree on the inventory port."""
    config = (Path(__file__).resolve().parent.parent / "docs" / "myst.yml").read_text(
        encoding="utf-8"
    )
    assert build_docs.API_ORIGIN in config
