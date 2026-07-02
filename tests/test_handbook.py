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
    compute_file_hash,
    detect_changes,
    incremental_ingest,
    ingest_handbooks,
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
        assert index["version"] == "1.1"
        assert set(index["source_files"]) == {"a.md", "b.txt"}
        assert index["total_chunks"] == len(index["chunks"])
        assert index["total_chunks"] > 0
        # M3-8: 索引应包含 file_hashes
        assert "file_hashes" in index
        assert set(index["file_hashes"].keys()) == {"a.md", "b.txt"}

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


class TestComputeFileHash:
    """M3-8: 文件 hash 计算测试"""

    def test_same_content_same_hash(self, tmp_path):
        """相同内容应产生相同 hash"""
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("内容一致", encoding="utf-8")
        f2.write_text("内容一致", encoding="utf-8")
        assert compute_file_hash(f1) == compute_file_hash(f2)

    def test_different_content_different_hash(self, tmp_path):
        """不同内容应产生不同 hash"""
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("内容A", encoding="utf-8")
        f2.write_text("内容B", encoding="utf-8")
        assert compute_file_hash(f1) != compute_file_hash(f2)

    def test_hash_is_hex_string(self, tmp_path):
        """hash 应为 32 位十六进制字符串"""
        f = tmp_path / "a.txt"
        f.write_text("test", encoding="utf-8")
        h = compute_file_hash(f)
        assert len(h) == 32
        assert all(c in "0123456789abcdef" for c in h)


class TestDetectChanges:
    """M3-8: 变更检测测试"""

    def test_detect_added_file(self, tmp_path):
        """检测新增文件"""
        (tmp_path / "a.md").write_text("# A\n\n内容A", encoding="utf-8")
        index = build_index(tmp_path)
        # 新增 b.md
        (tmp_path / "b.md").write_text("# B\n\n内容B", encoding="utf-8")
        changes = detect_changes(tmp_path, index)
        assert "b.md" in changes["added"]
        assert "a.md" in changes["unchanged"]

    def test_detect_modified_file(self, tmp_path):
        """检测修改文件"""
        (tmp_path / "a.md").write_text("# A\n\n内容A", encoding="utf-8")
        index = build_index(tmp_path)
        # 修改 a.md
        (tmp_path / "a.md").write_text("# A\n\n内容已修改", encoding="utf-8")
        changes = detect_changes(tmp_path, index)
        assert "a.md" in changes["modified"]

    def test_detect_deleted_file(self, tmp_path):
        """检测删除文件"""
        (tmp_path / "a.md").write_text("# A\n\n内容A", encoding="utf-8")
        (tmp_path / "b.md").write_text("# B\n\n内容B", encoding="utf-8")
        index = build_index(tmp_path)
        # 删除 b.md
        (tmp_path / "b.md").unlink()
        changes = detect_changes(tmp_path, index)
        assert "b.md" in changes["deleted"]
        assert "a.md" in changes["unchanged"]

    def test_detect_no_changes(self, tmp_path):
        """无变更时全部 unchanged"""
        (tmp_path / "a.md").write_text("# A\n\n内容A", encoding="utf-8")
        index = build_index(tmp_path)
        changes = detect_changes(tmp_path, index)
        assert changes["added"] == []
        assert changes["modified"] == []
        assert changes["deleted"] == []
        assert "a.md" in changes["unchanged"]


class TestIncrementalIngest:
    """M3-8: 增量导入完整流程测试"""

    @pytest.fixture
    def isolated_index(self, tmp_path, monkeypatch):
        """隔离索引路径，避免污染真实 handbook_index.json"""
        idx_path = tmp_path / "test_index.json"
        import ingest_handbooks
        monkeypatch.setattr(ingest_handbooks, "INDEX_PATH", idx_path)
        # load_index 默认用 INDEX_PATH，但也可以传参
        # incremental_ingest 内部调用 load_index() 无参数，用模块级 INDEX_PATH
        return idx_path

    def test_no_existing_index_fallback_full(self, tmp_path, isolated_index):
        """无旧索引应降级全量导入"""
        hb_dir = tmp_path / "handbooks"
        hb_dir.mkdir()
        (hb_dir / "a.md").write_text("# A\n\n内容A", encoding="utf-8")

        result = incremental_ingest(hb_dir)
        assert result["added"] == "all"
        # 索引应已生成
        assert isolated_index.exists()
        index = load_index(isolated_index)
        assert index is not None
        assert index["total_chunks"] > 0

    def test_no_changes_skips_import(self, tmp_path, isolated_index):
        """无变更应跳过导入"""
        hb_dir = tmp_path / "handbooks"
        hb_dir.mkdir()
        (hb_dir / "a.md").write_text("# A\n\n内容A", encoding="utf-8")

        # 第一次全量导入
        incremental_ingest(hb_dir)
        old_index = load_index(isolated_index)
        old_created = old_index["created_at"]

        # 第二次增量（无变更）
        import time
        time.sleep(0.05)  # 确保 created_at 不同
        result = incremental_ingest(hb_dir)
        assert result["added"] == 0
        assert result["modified"] == 0
        assert result["deleted"] == 0
        assert result["unchanged"] == 1

    def test_add_new_file(self, tmp_path, isolated_index):
        """新增文件：只导入新文件，保留旧 chunks"""
        hb_dir = tmp_path / "handbooks"
        hb_dir.mkdir()
        (hb_dir / "a.md").write_text("# A\n\n内容A", encoding="utf-8")
        incremental_ingest(hb_dir)

        # 新增 b.md
        (hb_dir / "b.md").write_text("# B\n\n内容B", encoding="utf-8")
        result = incremental_ingest(hb_dir)
        assert result["added"] == 1
        assert result["unchanged"] == 1

        index = load_index(isolated_index)
        sources = set(index["source_files"])
        assert sources == {"a.md", "b.md"}
        # 两个文件的 chunks 都应存在
        chunk_sources = {c["source"] for c in index["chunks"]}
        assert "a.md" in chunk_sources
        assert "b.md" in chunk_sources

    def test_modify_file(self, tmp_path, isolated_index):
        """修改文件：重导入该文件，保留其他"""
        hb_dir = tmp_path / "handbooks"
        hb_dir.mkdir()
        (hb_dir / "a.md").write_text("# A\n\n旧内容", encoding="utf-8")
        (hb_dir / "b.md").write_text("# B\n\n内容B", encoding="utf-8")
        incremental_ingest(hb_dir)
        old_b_chunks = [
            c for c in load_index(isolated_index)["chunks"] if c["source"] == "b.md"
        ]

        # 修改 a.md
        (hb_dir / "a.md").write_text("# A\n\n全新内容已修改", encoding="utf-8")
        result = incremental_ingest(hb_dir)
        assert result["modified"] == 1
        assert result["unchanged"] == 1

        index = load_index(isolated_index)
        new_b_chunks = [c for c in index["chunks"] if c["source"] == "b.md"]
        # b.md 的 chunks 应保留（内容不变）
        assert len(new_b_chunks) == len(old_b_chunks)
        assert new_b_chunks[0]["content"] == old_b_chunks[0]["content"]

    def test_delete_file(self, tmp_path, isolated_index):
        """删除文件：清理其 chunks"""
        hb_dir = tmp_path / "handbooks"
        hb_dir.mkdir()
        (hb_dir / "a.md").write_text("# A\n\n内容A", encoding="utf-8")
        (hb_dir / "b.md").write_text("# B\n\n内容B", encoding="utf-8")
        incremental_ingest(hb_dir)

        # 删除 b.md
        (hb_dir / "b.md").unlink()
        result = incremental_ingest(hb_dir)
        assert result["deleted"] == 1
        assert result["unchanged"] == 1

        index = load_index(isolated_index)
        # b.md 的 chunks 应被清理
        chunk_sources = {c["source"] for c in index["chunks"]}
        assert "b.md" not in chunk_sources
        assert "a.md" in chunk_sources
        # source_files 应不含 b.md
        assert "b.md" not in index["source_files"]

    def test_mixed_changes(self, tmp_path, isolated_index):
        """混合变更：同时增+改+删"""
        hb_dir = tmp_path / "handbooks"
        hb_dir.mkdir()
        (hb_dir / "keep.md").write_text("# Keep\n\n保留内容", encoding="utf-8")
        (hb_dir / "modify.md").write_text("# M\n\n旧内容", encoding="utf-8")
        (hb_dir / "delete.md").write_text("# D\n\n待删除", encoding="utf-8")
        incremental_ingest(hb_dir)

        # 混合变更
        (hb_dir / "add.md").write_text("# Add\n\n新增内容", encoding="utf-8")  # 新增
        (hb_dir / "modify.md").write_text("# M\n\n修改后内容", encoding="utf-8")  # 修改
        (hb_dir / "delete.md").unlink()  # 删除

        result = incremental_ingest(hb_dir)
        assert result["added"] == 1
        assert result["modified"] == 1
        assert result["deleted"] == 1
        assert result["unchanged"] == 1

        index = load_index(isolated_index)
        sources = set(index["source_files"])
        assert sources == {"keep.md", "modify.md", "add.md"}
        assert "delete.md" not in sources

    def test_index_has_file_hashes(self, tmp_path, isolated_index):
        """增量导入后索引应包含 file_hashes"""
        hb_dir = tmp_path / "handbooks"
        hb_dir.mkdir()
        (hb_dir / "a.md").write_text("# A\n\n内容A", encoding="utf-8")
        incremental_ingest(hb_dir)

        index = load_index(isolated_index)
        assert "file_hashes" in index
        assert "a.md" in index["file_hashes"]
        assert len(index["file_hashes"]["a.md"]) == 32

    def test_chunk_ids_sequential_after_incremental(self, tmp_path, isolated_index):
        """增量导入后 chunk_id 应连续编号"""
        hb_dir = tmp_path / "handbooks"
        hb_dir.mkdir()
        (hb_dir / "a.md").write_text("# A\n\n内容A", encoding="utf-8")
        (hb_dir / "b.md").write_text("# B\n\n内容B", encoding="utf-8")
        incremental_ingest(hb_dir)

        # 删除 a.md
        (hb_dir / "a.md").unlink()
        incremental_ingest(hb_dir)

        index = load_index(isolated_index)
        chunk_ids = [c["chunk_id"] for c in index["chunks"]]
        # chunk_id 应从 hb_0000 开始连续
        for i, cid in enumerate(chunk_ids):
            assert cid == f"hb_{i:04d}"
