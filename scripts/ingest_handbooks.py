"""
工艺手册导入脚本（M1-15 + M3-8 增量更新）

用途：将 PDF/Word/TXT/MD 工艺手册解析、分块、建立关键词索引。
运行：
  python scripts/ingest_handbooks.py                    # 全量导入
  python scripts/ingest_handbooks.py --incremental      # 增量更新（M3-8）
  python scripts/ingest_handbooks.py [data/handbooks/]  # 指定目录

设计说明：
- 不依赖 chromadb / embedding API（符合"不需要 LLM"的约束）
- 文件解析：pypdf(PDF) / python-docx(Word) / 内置(TXT,MD)
- 分块策略：按段落 + 长度上限（chunk_size=512, overlap=50）
- 索引格式：JSON（data/handbook_index.json），供 search_handbook 工具加载
- 检索方式：BM25-like 关键词评分（子串匹配 + 词频加权），支持中文
- M3-8 增量更新：基于文件 MD5 hash 检测增删改，只重导入变更文件

后续升级（可选）：安装 chromadb 后可同时写入向量库做语义检索。
"""
import hashlib
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# 添加项目根目录到 path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from loguru import logger

# ===== 配置 =====
HANDBOOK_DIR = PROJECT_ROOT / "data" / "handbooks"
INDEX_PATH = PROJECT_ROOT / "data" / "handbook_index.json"
CHUNK_SIZE = 512      # 每块最大字符数
CHUNK_OVERLAP = 50    # 块之间重叠字符数
SUPPORTED_SUFFIXES = (".pdf", ".docx", ".doc", ".txt", ".md")


# ===== 文件解析 =====

def parse_file(file_path: Path) -> list[dict]:
    """解析单个文件，返回 [{content, page, section}, ...]

    每个元素是一个"段落级"文本片段，后续再按 chunk_size 切分。
    """
    suffix = file_path.suffix.lower()
    try:
        if suffix == ".pdf":
            return _parse_pdf(file_path)
        elif suffix in (".docx", ".doc"):
            return _parse_docx(file_path)
        elif suffix in (".txt", ".md"):
            return _parse_text(file_path)
        else:
            logger.warning(f"不支持的文件类型: {file_path.name}")
            return []
    except Exception as e:
        logger.error(f"解析失败 {file_path.name}: {e}")
        return []


def _parse_pdf(file_path: Path) -> list[dict]:
    """解析 PDF（需 pypdf）"""
    from pypdf import PdfReader

    reader = PdfReader(str(file_path))
    fragments = []
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            fragments.append({
                "content": text,
                "page": i,
                "section": "",
            })
    logger.info(f"PDF {file_path.name}: {len(reader.pages)} 页, {len(fragments)} 个段落")
    return fragments


def _parse_docx(file_path: Path) -> list[dict]:
    """解析 Word（需 python-docx）"""
    import docx

    doc = docx.Document(str(file_path))
    fragments = []
    current_section = ""
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        # 标题样式作为 section
        if para.style and para.style.name and para.style.name.startswith("Heading"):
            current_section = text
        fragments.append({
            "content": text,
            "page": None,
            "section": current_section,
        })
    logger.info(f"DOCX {file_path.name}: {len(fragments)} 个段落")
    return fragments


def _parse_text(file_path: Path) -> list[dict]:
    """解析 TXT/MD（内置）"""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    fragments = []
    current_section = ""

    # 按空行分段
    raw_paragraphs = re.split(r"\n\s*\n", text)
    for para in raw_paragraphs:
        para = para.strip()
        if not para:
            continue

        # Markdown 标题作为 section（# 或 ## 开头）
        heading_match = re.match(r"^(#{1,6})\s+(.+)$", para)
        if heading_match:
            current_section = heading_match.group(2).strip()
            # 标题本身也作为内容保留（帮助检索）
            fragments.append({
                "content": para,
                "page": None,
                "section": current_section,
            })
        else:
            fragments.append({
                "content": para,
                "page": None,
                "section": current_section,
            })

    logger.info(f"TEXT {file_path.name}: {len(fragments)} 个段落")
    return fragments


# ===== 分块 =====

def chunk_fragments(fragments: list[dict], source_name: str) -> list[dict]:
    """将段落级片段切分为 chunk_size 大小的块

    策略：
    - 段落长度 ≤ chunk_size：整段作为一个 chunk
    - 段落长度 > chunk_size：按 chunk_size 滑窗切分，带 overlap
    """
    chunks = []
    chunk_idx = 0

    for frag in fragments:
        content = frag["content"]
        section = frag.get("section", "")
        page = frag.get("page")

        if len(content) <= CHUNK_SIZE:
            chunks.append(_make_chunk(chunk_idx, source_name, content, page, section))
            chunk_idx += 1
        else:
            # 滑窗切分
            start = 0
            while start < len(content):
                end = start + CHUNK_SIZE
                piece = content[start:end]
                chunks.append(_make_chunk(chunk_idx, source_name, piece, page, section))
                chunk_idx += 1
                start = end - CHUNK_OVERLAP
                if start >= len(content):
                    break

    return chunks


def _make_chunk(idx: int, source: str, content: str, page, section: str) -> dict:
    """构造单个 chunk"""
    return {
        "chunk_id": f"hb_{idx:04d}",
        "source": source,
        "page": page,
        "section": section,
        "content": content,
        "char_count": len(content),
    }


# ===== 索引构建 =====

def build_index(handbook_dir: Path) -> dict:
    """构建完整索引（全量导入）"""
    files = sorted(
        f for f in handbook_dir.glob("**/*") if f.suffix.lower() in SUPPORTED_SUFFIXES
    )

    if not files:
        logger.warning(f"目录中无可导入的文件: {handbook_dir}")
        logger.info(f"支持格式: {' '.join(SUPPORTED_SUFFIXES)}")
        return {}

    logger.info(f"找到 {len(files)} 个文件待导入")

    all_chunks = []
    source_files = []
    file_hashes = {}

    for file in files:
        logger.info(f"处理: {file.name} ({file.stat().st_size} bytes)")
        fragments = parse_file(file)
        chunks = chunk_fragments(fragments, file.name)
        all_chunks.extend(chunks)
        source_files.append(file.name)
        file_hashes[file.name] = compute_file_hash(file)

    index = {
        "version": "1.1",
        "created_at": datetime.now().isoformat(),
        "source_files": source_files,
        "file_hashes": file_hashes,
        "total_chunks": len(all_chunks),
        "chunks": all_chunks,
    }

    return index


def save_index(index: dict, path: Path | None = None):
    """保存索引到 JSON 文件"""
    if path is None:
        path = INDEX_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    logger.info(f"索引已保存: {path} ({path.stat().st_size} bytes, {index['total_chunks']} chunks)")


def load_index(path: Path | None = None) -> dict | None:
    """加载索引（供 search_handbook 调用）"""
    if path is None:
        path = INDEX_PATH
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"加载手册索引失败: {e}")
        return None


# ===== M3-8: 增量更新 =====

def compute_file_hash(file_path: Path) -> str:
    """计算文件 MD5 hash（用于变更检测）

    Args:
        file_path: 文件路径

    Returns:
        32 位十六进制 MD5 字符串
    """
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def detect_changes(
    handbook_dir: Path,
    existing_index: dict,
) -> dict:
    """检测手册目录与现有索引之间的变更

    比较当前文件的 MD5 hash 与索引中记录的 hash，分类为：
    - added: 新增文件（索引中没有）
    - modified: 修改文件（hash 不同）
    - deleted: 删除文件（索引中有但目录中没有）
    - unchanged: 未变更文件（hash 相同）

    Args:
        handbook_dir: 手册目录
        existing_index: 现有索引（含 file_hashes 字段）

    Returns:
        {"added": [str], "modified": [str], "deleted": [str], "unchanged": [str]}
    """
    current_files = sorted(
        f.name for f in handbook_dir.glob("**/*")
        if f.suffix.lower() in SUPPORTED_SUFFIXES
    )
    old_hashes = existing_index.get("file_hashes", {})
    old_files = set(old_hashes.keys())

    added, modified, unchanged = [], [], []
    for fname in current_files:
        current_hash = compute_file_hash(handbook_dir / fname)
        if fname not in old_files:
            added.append(fname)
        elif old_hashes[fname] != current_hash:
            modified.append(fname)
        else:
            unchanged.append(fname)

    deleted = [f for f in old_files if f not in current_files]

    return {
        "added": added,
        "modified": modified,
        "deleted": deleted,
        "unchanged": unchanged,
    }


def incremental_ingest(handbook_dir: str | Path = HANDBOOK_DIR) -> dict:
    """增量导入工艺手册（M3-8）

    只重新导入新增和修改的文件，保留未变更文件的 chunks，
    删除已删除文件的 chunks。

    如果现有索引不存在或版本 < 1.1（无 file_hashes），降级为全量导入。

    Args:
        handbook_dir: 手册文件目录

    Returns:
        导入统计 {"added": N, "modified": N, "deleted": N, "unchanged": N, "total_chunks": N}
    """
    handbook_dir = Path(handbook_dir)
    if not handbook_dir.exists():
        logger.error(f"目录不存在: {handbook_dir}")
        return {}

    existing = load_index()
    # 无旧索引或旧索引无 file_hashes → 降级全量
    if not existing or "file_hashes" not in existing:
        logger.info("无有效旧索引，执行全量导入")
        ingest_handbooks(handbook_dir)
        return {"added": "all", "modified": 0, "deleted": 0, "unchanged": 0}

    changes = detect_changes(handbook_dir, existing)
    added = changes["added"]
    modified = changes["modified"]
    deleted = changes["deleted"]
    unchanged = changes["unchanged"]

    logger.info(
        f"变更检测: +{len(added)} 新增, ~{len(modified)} 修改, "
        f"-{len(deleted)} 删除, ={len(unchanged)} 未变更"
    )

    # 无变更则跳过
    if not added and not modified and not deleted:
        logger.info("无变更，跳过导入")
        return {
            "added": 0, "modified": 0, "deleted": 0, "unchanged": len(unchanged),
            "total_chunks": existing.get("total_chunks", 0),
        }

    start = time.time()

    # 1. 保留 unchanged 文件的 chunks
    old_chunks = existing.get("chunks", [])
    kept_chunks = [c for c in old_chunks if c.get("source") in unchanged]

    # 2. 重新导入 added + modified 文件
    new_chunks = []
    new_hashes = {}
    for fname in added + modified:
        file_path = handbook_dir / fname
        logger.info(f"导入: {fname} ({file_path.stat().st_size} bytes)")
        fragments = parse_file(file_path)
        chunks = chunk_fragments(fragments, fname)
        new_chunks.extend(chunks)
        new_hashes[fname] = compute_file_hash(file_path)

    # 3. 合并：保留 + 新增，重新编号 chunk_id
    all_chunks = kept_chunks + new_chunks
    for i, chunk in enumerate(all_chunks):
        chunk["chunk_id"] = f"hb_{i:04d}"

    # 4. 合并 file_hashes（unchanged 用旧 hash，added/modified 用新 hash）
    old_hashes = existing.get("file_hashes", {})
    merged_hashes = {f: old_hashes[f] for f in unchanged}
    merged_hashes.update(new_hashes)

    # 5. 构建新索引
    index = {
        "version": "1.1",
        "created_at": datetime.now().isoformat(),
        "source_files": sorted(unchanged + added + modified),
        "file_hashes": merged_hashes,
        "total_chunks": len(all_chunks),
        "chunks": all_chunks,
    }

    save_index(index)

    elapsed = time.time() - start
    logger.info(
        f"增量导入完成: +{len(added)} 新增, ~{len(modified)} 修改, "
        f"-{len(deleted)} 删除, ={len(unchanged)} 保留, "
        f"{len(all_chunks)} 个块, 耗时 {elapsed:.1f}s"
    )

    return {
        "added": len(added),
        "modified": len(modified),
        "deleted": len(deleted),
        "unchanged": len(unchanged),
        "total_chunks": len(all_chunks),
    }


# ===== 主入口 =====

def ingest_handbooks(handbook_dir: str | Path = HANDBOOK_DIR):
    """导入工艺手册到关键词索引

    Args:
        handbook_dir: 手册文件目录
    """
    handbook_dir = Path(handbook_dir)
    if not handbook_dir.exists():
        logger.error(f"目录不存在: {handbook_dir}")
        return

    start = time.time()
    index = build_index(handbook_dir)

    if not index:
        return

    save_index(index)
    elapsed = time.time() - start
    logger.info(
        f"导入完成: {len(index['source_files'])} 个文件, "
        f"{index['total_chunks']} 个块, 耗时 {elapsed:.1f}s"
    )
    logger.info(f"索引路径: {INDEX_PATH}")
    logger.info("search_handbook 工具将自动加载此索引")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="导入工艺手册到关键词索引（支持全量/增量）"
    )
    parser.add_argument(
        "handbook_dir",
        type=str,
        default=str(HANDBOOK_DIR),
        nargs="?",
        help="手册文件目录",
    )
    parser.add_argument(
        "--incremental", "-i",
        action="store_true",
        help="增量更新模式：基于文件 hash 只重导入变更文件（M3-8）",
    )
    args = parser.parse_args()

    if args.incremental:
        incremental_ingest(args.handbook_dir)
    else:
        ingest_handbooks(args.handbook_dir)
