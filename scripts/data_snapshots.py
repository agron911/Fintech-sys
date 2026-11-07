"""Verified stock snapshots and patches; compatible with Python 3.9."""

import difflib
from contextlib import contextmanager
import gzip
import hashlib
import heapq
import io
import json
import os
from pathlib import Path
import re
import tarfile
import tempfile

MANIFEST_NAME = "stock-data-manifest.json.gz"
FULL_NAME = "stock-data.tar.gz"
STATE_NAME = ".snapshot.json"
SCHEMA = 1


class SnapshotError(ValueError):
    pass


def json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def file_name(name):
    if (not isinstance(name, str) or not name.endswith(".txt")
            or "/" in name or "\\" in name or ":" in name or name in {".txt", "..txt"}):
        raise SnapshotError("Invalid stock filename")
    return name


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(directory, names=None):
    directory = Path(directory)
    names = names if names is not None else [p.name for p in directory.glob("*.txt")]
    result = {}
    for name in sorted(set(names)):
        path = directory / file_name(name)
        if path.is_file():
            result[name] = {"sha256": sha256_file(path), "size": path.stat().st_size}
    return result


def snapshot_id(files):
    return hashlib.sha256(json_bytes(files)).hexdigest()


def validate_files(files):
    if not isinstance(files, dict):
        raise SnapshotError("Invalid file inventory")
    for name, entry in files.items():
        file_name(name)
        if (not isinstance(entry, dict) or not isinstance(entry.get("size"), int)
                or entry["size"] < 0
                or not re.fullmatch(r"[0-9a-f]{64}", str(entry.get("sha256", "")))):
            raise SnapshotError("Invalid file checksum or size")


def validate_manifest(manifest):
    if manifest.get("schema") != SCHEMA:
        raise SnapshotError("Unsupported snapshot manifest")
    validate_files(manifest.get("files"))
    if not manifest["files"]:
        raise SnapshotError("Snapshot manifest contains no stock files")
    if manifest.get("snapshot") != snapshot_id(manifest["files"]):
        raise SnapshotError("Snapshot inventory checksum mismatch")
    for name in manifest.get("known_files", manifest["files"]):
        file_name(name)
    for asset in [manifest["full"]] + manifest.get("deltas", []):
        name = asset.get("name", "")
        if (not isinstance(name, str) or "/" in name or "\\" in name
                or not name.endswith(".tar.gz")
                or not isinstance(asset.get("size"), int) or asset["size"] < 0
                or not re.fullmatch(r"[0-9a-f]{64}", str(asset.get("sha256", "")))):
            raise SnapshotError("Invalid snapshot asset")
    return manifest


def read_json(path):
    data = Path(path).read_bytes()
    if data.startswith(b"\x1f\x8b"):
        data = gzip.decompress(data)
    return json.loads(data)


def artifact(path):
    path = Path(path)
    return {"name": path.name, "size": path.stat().st_size, "sha256": sha256_file(path)}


def add_bytes(tar, name, content):
    info = tarfile.TarInfo(name)
    info.size = len(content)
    info.mode = 0o644
    tar.addfile(info, io.BytesIO(content))


@contextmanager
def archive_writer(output, compresslevel):
    with Path(output).open("wb") as stream:
        with gzip.GzipFile(filename="", mode="wb", fileobj=stream,
                           compresslevel=compresslevel, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as tar:
                yield tar


def write_full(directory, output, files):
    with archive_writer(output, 6) as tar:
        for name, entry in sorted(files.items()):
            info = tarfile.TarInfo("data/raw/" + name)
            info.size = entry["size"]
            info.mode = 0o644
            with (Path(directory) / name).open("rb") as stream:
                tar.addfile(info, stream)


def make_patch(before, after):
    """Reuse unchanged historical rows and send only changed bytes."""
    if before and after.startswith(before):
        added = after[len(before):]
        return [["copy", 0, len(before)], ["add", 0, len(added)]], added
    old_lines = before.splitlines(keepends=True)
    new_lines = after.splitlines(keepends=True)
    offsets = [0]
    for line in old_lines:
        offsets.append(offsets[-1] + len(line))
    operations = []
    payload = bytearray()
    matcher = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
    for kind, old_start, old_end, new_start, new_end in matcher.get_opcodes():
        if kind == "equal":
            operations.append(["copy", offsets[old_start], offsets[old_end] - offsets[old_start]])
        elif kind != "delete":
            chunk = b"".join(new_lines[new_start:new_end])
            operations.append(["add", len(payload), len(chunk)])
            payload.extend(chunk)
    if len(payload) + len(json_bytes(operations)) >= len(after):
        return [["add", 0, len(after)]], after
    return operations, bytes(payload)


def write_delta(previous, current, output, before_files, after_files):
    changes = {}
    with archive_writer(output, 9) as tar:
        for name, entry in sorted(after_files.items()):
            if before_files.get(name) == entry:
                continue
            new_bytes = (Path(current) / name).read_bytes()
            old_bytes = (Path(previous) / name).read_bytes() if name in before_files else b""
            operations, payload = make_patch(old_bytes, new_bytes)
            changes[name] = {"operations": operations, "payload": "payload/" + name + ".bin"}
            add_bytes(tar, changes[name]["payload"], payload)
        index = {"schema": SCHEMA, "base": snapshot_id(before_files),
                 "target": snapshot_id(after_files), "base_files": before_files,
                 "files": after_files, "changes": changes,
                 "deleted": sorted(set(before_files) - set(after_files))}
        add_bytes(tar, "delta.json", json_bytes(index))
    return len(changes), len(index["deleted"])


def build_release(current, output, *, previous=None, previous_manifest=None, keep_deltas=8):
    """Build the full fallback, compressed manifest, and one immutable delta."""
    from datetime import datetime, timezone

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    files = inventory(current)
    if not files:
        raise SnapshotError("No stock files to publish")
    old_manifest = {}
    if previous_manifest and Path(previous_manifest).is_file():
        old_manifest = validate_manifest(read_json(previous_manifest))
    before_files = inventory(previous) if previous and Path(previous).is_dir() else {}
    base = snapshot_id(before_files)
    target = snapshot_id(files)
    full_path = output / FULL_NAME
    write_full(current, full_path, files)
    deltas = list(old_manifest.get("deltas", []))
    created = None
    changed = deleted = 0
    if before_files and base != target:
        path = output / f"stock-data-delta-{base[:16]}-{target[:16]}.tar.gz"
        changed, deleted = write_delta(previous, current, path, before_files, files)
        created = dict(artifact(path), base=base, target=target)
        deltas = [d for d in deltas if not (d["base"] == base and d["target"] == target)]
        deltas.append(created)
    deltas = deltas[-keep_deltas:]
    known = sorted(set(files) | set(before_files) | set(old_manifest.get("known_files", [])))
    manifest = {"schema": SCHEMA, "snapshot": target,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "files": files, "known_files": known,
                "full": artifact(full_path), "deltas": deltas}
    validate_manifest(manifest)
    (output / MANIFEST_NAME).write_bytes(gzip.compress(json_bytes(manifest), mtime=0))
    report = {"snapshot": target, "files": len(files), "full": manifest["full"],
              "delta": created, "changed_files": changed, "deleted_files": deleted,
              "retained_deltas": [d["name"] for d in deltas]}
    (output / "publish.json").write_bytes(json_bytes(report))
    return report


def delta_path(start, manifest):
    """Choose the smallest retained chain to reach the latest snapshot."""
    queue = [(0, start, [])]
    visited = set()
    while queue:
        cost, node, path = heapq.heappop(queue)
        if node in visited:
            continue
        visited.add(node)
        if node == manifest["snapshot"]:
            return path
        for index, delta in enumerate(manifest.get("deltas", [])):
            if delta.get("base") == node and delta.get("target") not in visited:
                # The sequence number keeps ties from comparing dictionaries.
                heapq.heappush(queue, (cost + delta["size"], delta["target"], path + [index]))
    return None


def load_state(directory):
    try:
        state = read_json(Path(directory) / STATE_NAME)
        validate_files(state["files"])
        return state
    except (OSError, ValueError, KeyError, TypeError):
        return {}


def save_state(directory, files, **extra):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = dict(extra, schema=SCHEMA, snapshot=snapshot_id(files), files=files)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
            temporary = stream.name
            stream.write(json_bytes(state))
        os.replace(temporary, directory / STATE_NAME)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def verify_content(content, entry):
    if len(content) != entry["size"] or hashlib.sha256(content).hexdigest() != entry["sha256"]:
        raise SnapshotError("Stock file checksum mismatch")


def install_full(archive, directory, files):
    """Stage and check all stock files before replacing any current data."""
    validate_files(files)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    state = load_state(directory)
    with tempfile.TemporaryDirectory(dir=directory.parent, prefix=".stock-update-") as staging:
        staging = Path(staging)
        seen = set()
        with tarfile.open(archive, "r:gz") as tar:
            for member in tar:
                if member.isdir():
                    continue
                if not member.isfile() or not member.name.startswith("data/raw/"):
                    raise SnapshotError("Unexpected archive member")
                name = file_name(member.name[len("data/raw/"):])
                if name not in files or name in seen or member.size != files[name]["size"]:
                    raise SnapshotError("Unexpected or duplicate stock file")
                content = tar.extractfile(member).read()
                verify_content(content, files[name])
                (staging / name).write_bytes(content)
                seen.add(name)
        if seen != set(files):
            raise SnapshotError("Snapshot is missing stock files")
        for name in sorted(files):
            os.replace(staging / name, directory / name)
        for name in set(state.get("files", {})) - set(files):
            (directory / file_name(name)).unlink(missing_ok=True)
    if inventory(directory, files) != files:
        raise SnapshotError("Snapshot changed during installation")
    save_state(directory, files)


def apply_delta(archive, directory, expected_base, expected_target):
    directory = Path(directory)
    with tarfile.open(archive, "r:gz") as tar:
        index_member = tar.getmember("delta.json")
        if not index_member.isfile():
            raise SnapshotError("Invalid delta index")
        index = json.load(tar.extractfile(index_member))
        validate_files(index["base_files"])
        validate_files(index["files"])
        if (index.get("schema") != SCHEMA or index.get("base") != expected_base
                or index.get("target") != expected_target
                or snapshot_id(index["base_files"]) != expected_base
                or snapshot_id(index["files"]) != expected_target):
            raise SnapshotError("Delta snapshot mismatch")
        if inventory(directory, index["base_files"]) != index["base_files"]:
            raise SnapshotError("Local history does not match the delta base")
        deleted = set(index["deleted"])
        if deleted != set(index["base_files"]) - set(index["files"]):
            raise SnapshotError("Invalid removed file inventory")
        proposed = dict(index["base_files"])
        for name in deleted:
            proposed.pop(name)
        with tempfile.TemporaryDirectory(dir=directory.parent, prefix=".stock-update-") as staging:
            staging = Path(staging)
            for name, plan in sorted(index["changes"].items()):
                file_name(name)
                if name not in index["files"] or plan["payload"] != "payload/" + name + ".bin":
                    raise SnapshotError("Invalid delta stock file")
                member = tar.getmember(plan["payload"])
                if not member.isfile():
                    raise SnapshotError("Invalid delta payload")
                payload = tar.extractfile(member).read()
                before = (directory / name).read_bytes() if name in index["base_files"] else b""
                content = bytearray()
                for kind, offset, length in plan["operations"]:
                    source = before if kind == "copy" else payload if kind == "add" else None
                    if (source is None or not isinstance(offset, int) or not isinstance(length, int)
                            or offset < 0 or length < 0 or offset + length > len(source)
                            or len(content) + length > index["files"][name]["size"]):
                        raise SnapshotError("Invalid historical byte range")
                    content.extend(source[offset:offset + length])
                verify_content(content, index["files"][name])
                (staging / name).write_bytes(content)
                proposed[name] = index["files"][name]
            if proposed != index["files"]:
                raise SnapshotError("Delta omitted changed stock files")
            for name in sorted(index["changes"]):
                os.replace(staging / name, directory / name)
            for name in sorted(deleted):
                (directory / file_name(name)).unlink(missing_ok=True)
    if inventory(directory, index["files"]) != index["files"]:
        raise SnapshotError("Snapshot changed during installation")
    save_state(directory, index["files"])
    return len(index["changes"])
