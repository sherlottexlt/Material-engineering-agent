"""
工艺手册导入脚本（M1-15）

用途：将 PDF/Word/TXT/MD 工艺手册解析、分块、建立关键词索引。
运行：python scripts/ingest_handbooks.py [data/handbooks/]

设计说明：
- 不依赖 chromadb / embedding API（符合"不需要 LLM"的约束）
- 文件解析：pypdf(PDF) / python-docx(Word) / 内置(TXT,MD)
- 分块策略：按段落 + 长度上限（chunk_size=512, overlap=50）
- 索引格式：JSON（data/handbook_index.json），供 search_handbook 工具加载
- 检索方式：BM25-like 关键词评分（子串匹配 + 词频加权），支持中文

后续升级（可选）：安装 chromadb 后可同时写入向量库做语义检索。
"""
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
    """构建完整索引"""
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

    for file in files:
        logger.info(f"处理: {file.name} ({file.stat().st_size} bytes)")
        fragments = parse_file(file)
        chunks = chunk_fragments(fragments, file.name)
        all_chunks.extend(chunks)
        source_files.append(file.name)

    index = {
        "version": "1.0",
        "created_at": datetime.now().isoformat(),
        "source_files": source_files,
        "total_chunks": len(all_chunks),
        "chunks": all_chunks,
    }

    return index


def save_index(index: dict, path: Path = INDEX_PATH):
    """保存索引到 JSON 文件"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    logger.info(f"索引已保存: {path} ({path.stat().st_size} bytes, {index['total_chunks']} chunks)")


def load_index(path: Path = INDEX_PATH) -> dict | None:
    """加载索引（供 search_handbook 调用）"""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"加载手册索引失败: {e}")
        return None


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

    parser = argparse.ArgumentParser(description="导入工艺手册到关键词索引")
    parser.add_argument(
        "handbook_dir",
        type=str,
        default=str(HANDBOOK_DIR),
        nargs="?",
        help="手册文件目录",
    )
    args = parser.parse_args()

    ingest_handbooks(args.handbook_dir)
