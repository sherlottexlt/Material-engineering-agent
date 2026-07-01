"""
工艺手册导入脚本单元测试（M1-15）

测试内容：
- _parse_text: TXT/MD 文件解析
- chunk_fragments: 分块逻辑
- build_index: 完整索引构建
- _tokenize_query: 查询分词
- _bm25_like_score: BM25 评分
"""
import json
from pathlib import Path

import pytest

# ingest_handbooks 在 scripts/ 目录，需添加路径
import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from ingest_handbooks import (
    CHUNK_SIZE,
    _make_chunk,
    build_index,
    chunk_fragments,
    load_index,
    parse_file,
)
from agent.tools import _bm25_like_score, _load_handbook_index, _tokenize_query


class TestParseText:
    """测试 TXT/MD 解析"""

    def test_parse_md_file(self, tmp_path):
        """应正确解析 Markdown 文件"""
        md_file = tmp_path / "test.md"
        md_file.write_text(
            "# 标题一\n\n第一段内容。\n\n## 子标题\n\n第二段内容。\n",
            encoding="utf-8",
        )
        fragments = parse_file(md_file)
        assert len(fragments) >= 3
        # 第一个非空段落应识别 section
        assert any(f["section"] == "标题一" for f in fragments)

    def test_parse_txt_file(self, tmp_path):
        """应正确解析 TXT 文件"""
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("段落一\n\n段落二\n\n段落三", encoding="utf-8")
        fragments = parse_file(txt_file)
        assert len(fragments) == 3

    def test_parse_empty_file(self, tmp_path):
        """空文件应返回空列表"""
        empty = tmp_path / "empty.md"
        empty.write_text("", encoding="utf-8")
        fragments = parse_file(empty)
        assert fragments == []

    def test_parse_unsupported_type(self, tmp_path):
        """不支持的文件类型应返回空列表"""
        bad = tmp_path / "bad.xyz"
        bad.write_text("content", encoding="utf-8")
        fragments = parse_file(bad)
        assert fragments == []


class TestChunkFragments:
    """测试分块逻辑"""

    def test_short_fragment_single_chunk(self):
        """短段落应为单个 chunk"""
        fragments = [{"content": "短内容", "page": None, "section": ""}]
        chunks = chunk_fragments(fragments, "test.md")
        assert len(chunks) == 1
        assert chunks[0]["content"] == "短内容"
        assert chunks[0]["source"] == "test.md"
        assert chunks[0]["chunk_id"] == "hb_0000"

    def test_long_fragment_sliding_window(self):
        """长段落应滑窗切分"""
        long_text = "A" * (CHUNK_SIZE * 3)
        fragments = [{"content": long_text, "page": 1, "section": "S"}]
        chunks = chunk_fragments(fragments, "test.pdf")
        # 3 倍 chunk_size 应产生 >= 3 个 chunk
        assert len(chunks) >= 3
        # 每个 chunk 不超过 chunk_size
        for c in chunks:
            assert len(c["content"]) <= CHUNK_SIZE
        # chunk_id 递增
        ids = [c["chunk_id"] for c in chunks]
        assert ids == sorted(ids)

    def test_section_preserved(self):
        """section 应传递到 chunk"""
        fragments = [{"content": "内容", "page": None, "section": "我的章节"}]
        chunks = chunk_fragments(fragments, "test.md")
        assert chunks[0]["section"] == "我的章节"

    def test_chunk_ids_unique(self):
        """多个段落的 chunk_id 应唯一"""
        fragments = [
            {"content": "A", "page": None, "section": ""},
            {"content": "B", "page": None, "section": ""},
            {"content": "C", "page": None, "section": ""},
        ]
        chunks = chunk_fragments(fragments, "test.md")
        ids = [c["chunk_id"] for c in chunks]
        assert len(ids) == len(set(ids))


class TestBuildIndex:
    """测试索引构建"""

    def test_build_index_from_dir(self, tmp_path):
        """应从目录构建索引"""
        (tmp_path / "a.md").write_text("# A\n\n内容A", encoding="utf-8")
        (tmp_path / "b.txt").write_text("内容B", encoding="utf-8")
        index = build_index(tmp_path)
        assert index["version"] == "1.0"
        assert set(index["source_files"]) == {"a.md", "b.txt"}
        assert index["total_chunks"] == len(index["chunks"])
        assert index["total_chunks"] > 0

    def test_build_index_empty_dir(self, tmp_path):
        """空目录应返回空字典"""
        index = build_index(tmp_path)
        assert index == {}

    def test_make_chunk_structure(self):
        """_make_chunk 应返回正确结构"""
        chunk = _make_chunk(5, "test.md", "内容", 2, "章节")
        assert chunk["chunk_id"] == "hb_0005"
        assert chunk["source"] == "test.md"
        assert chunk["content"] == "内容"
        assert chunk["page"] == 2
        assert chunk["section"] == "章节"
        assert chunk["char_count"] == 2


class TestTokenizeQuery:
    """测试查询分词"""

    def test_chinese_query(self):
        """中文查询应生成 2-gram"""
        tokens = _tokenize_query("硬度偏低")
        assert "硬度偏低" in tokens
        assert "硬度" in tokens  # 2-gram
        assert "度偏" in tokens

    def test_english_query(self):
        """英文查询应整体保留"""
        tokens = _tokenize_query("hardness_low temperature")
        assert "hardness_low" in tokens
        assert "temperature" in tokens

    def test_mixed_query(self):
        """中英混合查询"""
        tokens = _tokenize_query("硬度偏低 cooling_rate 840")
        assert "硬度偏低" in tokens
        assert "cooling_rate" in tokens
        assert "840" in tokens

    def test_dedup(self):
        """重复 token 应去重"""
        tokens = _tokenize_query("硬度 硬度")
        assert tokens.count("硬度") == 1


class TestBM25Score:
    """测试 BM25 评分"""

    def test_match_higher_than_no_match(self):
        """匹配的文档得分应高于不匹配的"""
        tokens = _tokenize_query("硬度偏低")
        match_doc = "硬度偏低是常见缺陷"
        no_match_doc = "完全无关的内容"
        assert _bm25_like_score(tokens, match_doc) > _bm25_like_score(tokens, no_match_doc)

    def test_no_match_returns_zero(self):
        """无匹配应返回 0"""
        tokens = _tokenize_query("xyz")
        assert _bm25_like_score(tokens, "中文内容") == 0.0

    def test_empty_doc(self):
        """空文档应返回 0"""
        tokens = _tokenize_query("测试")
        assert _bm25_like_score(tokens, "") == 0.0

    def test_more_matches_higher_score(self):
        """多次命中得分应高于单次命中"""
        tokens = _tokenize_query("硬度")
        single = "硬度一次"
        multi = "硬度硬度硬度"
        assert _bm25_like_score(tokens, multi) > _bm25_like_score(tokens, single)


class TestLoadIndex:
    """测试索引加载"""

    def test_load_existing_index(self):
        """应能加载已生成的索引"""
        index = _load_handbook_index()
        if index is None:
            pytest.skip("手册索引未生成（先运行 ingest_handbooks.py）")
        assert "chunks" in index
        assert "total_chunks" in index
        assert index["total_chunks"] > 0

    def test_load_index_from_path(self, tmp_path):
        """load_index 应从指定路径加载"""
        idx_file = tmp_path / "idx.json"
        data = {"version": "1.0", "chunks": [], "total_chunks": 0}
        idx_file.write_text(json.dumps(data), encoding="utf-8")
        loaded = load_index(idx_file)
        assert loaded["version"] == "1.0"

    def test_load_nonexistent_returns_none(self, tmp_path):
        """加载不存在的文件应返回 None"""
        result = load_index(tmp_path / "nope.json")
        assert result is None
