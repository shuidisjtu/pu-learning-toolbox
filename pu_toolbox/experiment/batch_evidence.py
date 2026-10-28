"""One evidence packet per batch: the identity a survey run cites to reproduce it.

Protocol §5 clause 6 asks every run artifact to fix the code commit, the
dependency lock and the software stack it ran on.  D44 puts that identity at the
batch layer instead of in every manifest: a survey batch is one unit script
invocation covering many runs, and repeating one commit and one lock hash in each
of them is noise rather than evidence.  :data:`survey_audit.BATCH_IDENTITY_FIELDS`
names the two fields a packet has to carry -- ``code_commit`` and
``dependency_lock`` -- and the packet adds what a reproducer on another machine
needs on top of them: the protocol matrix it planned from, the frozen recipe
registry it bound, and a compact record of the stack it trained on.

Two properties make the packet evidence rather than a log line.

* **It is content-addressed.**  ``batch_id`` is the digest of everything in the
  packet except itself, and the file name ends in the same digest.  A rerun of
  one batch under a different commit therefore writes a second packet instead of
  overwriting the first, and a packet whose content was edited no longer matches
  its own identity.
* **A run cannot cite a packet that does not describe it.**  The unit script
  re-reads the packet and checks, before it trains, the unit, the seeds, the c
  tokens, the protocol digest and the checkout the packet was written from.  So
  ``batch.batch_id`` in a manifest is a claim that was checked rather than copied.

The packet lives in ``<results>/batch_evidence/`` beside the run trees.  A
manifest records the packet's file name and digest only: an absolute path would
name the host that happened to write it, and the reader's host is not that one.
The file itself is written with ``\n`` endings and replaced atomically, so the
same packet has the same bytes on either platform and a crash leaves no
half-written packet behind.

**What this packet is not.**  It records one attempt: the commit, the lock, the
stack and the sizes of that attempt.  Two hosts running the same unit with the
same seeds therefore produce two packets, and a partially finished batch that is
resumed as per-seed batches produces one packet per seed.  Nothing here rewrites
or renumbers an existing packet; a reader compares runs by their unit, seeds and
c tokens, never by ``batch_id``.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pu_toolbox.utils.serialization import file_hash, strict_canonical_hash

from .resources import runtime_environment
from .survey_provenance import (
    CODE_COMMIT_RECORDED,
    CODE_COMMIT_UNAVAILABLE,
    resolve_code_commit,
)

#: Format version of the packet.  Moved only when a field is removed, renamed or
#: retyped: adding one is backward compatible and leaves this alone.
SCHEMA_VERSION = "batch-evidence-v1"

#: Directory under the results root that holds the packets, one per batch.
DIRECTORY = "batch_evidence"

#: The dependency lock this project ships; ``check_project_metadata`` requires it
#: to exist and to stay committed, so its digest is the environment identity.
LOCK_FILE = "uv.lock"

#: This checkout's root, where both the commit and the lock are read from.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

#: Environment keys a reproducer needs and a host can name.  The probe also
#: reports paths (its own ``executable``) and host facts; those stay out because
#: they describe this machine rather than the run.
_ENVIRONMENT_KEYS = (
    "python_version",
    "python_implementation",
    "platform",
    "torch_version",
    "cuda_available",
    "cuda_runtime_version",
    "cudnn_version",
    "gpu_devices",
    "gpu_driver_version",
    "numpy_version",
    "scikit_learn_version",
)

#: The keys each half of the identity may carry, so a caller cannot smuggle a
#: different ``unit`` or ``schema_version`` in through an injected block.
_CODE_KEYS = frozenset({"code_commit", "code_commit_dirty", "code_commit_state"})
_LOCK_KEYS = frozenset({"path", "sha256", "state"})
_UNIT_KEYS = ("dataset", "method", "training_path", "mechanism")

#: Fields a readable packet must carry, checked before any of them is trusted.
_REQUIRED_FIELDS = (
    "schema_version",
    "batch_id",
    "unit",
    "seeds",
    "c_tokens",
    "run_count",
    "protocol_version",
    "protocol_sha256",
    "code_commit",
    "code_commit_dirty",
    "code_commit_state",
    "dependency_lock",
    "environment",
)

#: Characters that would turn a unit name into a path, or into a Windows device
#: name, when it is used as a file name.
_UNSAFE_IN_NAME = ("/", "\\", "..", ":")


def code_identity(cwd: str | Path | None = None) -> dict[str, Any]:
    """The commit a batch ran from, and whether the tree that ran it was clean.

    ``code_commit_state`` is a first-class field, never absent: a packet produced
    outside a git work tree says ``unavailable`` rather than leaving a reader to
    guess whether the commit was forgotten.

    ``cwd`` defaults to this module's checkout, not the process's directory, so
    the commit always belongs to the code that is writing the packet.
    """
    try:
        commit, dirty = resolve_code_commit(cwd=REPOSITORY_ROOT if cwd is None else cwd)
    except (OSError, ValueError):
        commit, dirty = None, None
    return {
        "code_commit": commit,
        "code_commit_dirty": dirty,
        "code_commit_state": CODE_COMMIT_RECORDED if commit else CODE_COMMIT_UNAVAILABLE,
    }


def dependency_lock(path: str | Path | None = None) -> dict[str, Any]:
    """The lock file's digest, recorded by name rather than by read path.

    The name is what a reproducer on another machine needs; where this checkout
    kept it is not.  An unreadable lock is recorded as ``unavailable`` and
    reported by the caller, never replaced by a guess.
    """
    lock = Path(path) if path is not None else REPOSITORY_ROOT / LOCK_FILE
    try:
        digest = file_hash(lock) if lock.is_file() else None
    except OSError:
        digest = None
    return {
        "path": lock.name,
        "sha256": digest,
        "state": "recorded" if digest is not None else "unavailable",
    }


def environment_identity() -> dict[str, Any]:
    """The software stack the batch trained on, or why it could not be read.

    ``state`` separates "trained on a CPU-only machine" from "the probe failed":
    both would otherwise read as the same packet, and only one of them is a
    reproducible fact.
    """
    try:
        measured = runtime_environment()
    except Exception as exc:  # noqa: BLE001 - a probe must not stop a batch
        return {"state": "unavailable", "reason": f"{type(exc).__name__}: {exc}"}
    environment = {key: measured.get(key) for key in _ENVIRONMENT_KEYS}
    processor = measured.get("processor")
    if not processor:
        try:
            with Path("/proc/cpuinfo").open(encoding="utf-8") as cpuinfo:
                processor = next(
                    (
                        line.split(":", 1)[1].strip()
                        for line in cpuinfo
                        if line.startswith("model name") and ":" in line
                    ),
                    None,
                )
        except OSError:
            pass
    environment["hardware"] = {
        "machine": measured.get("machine"),
        "processor": processor,
        "cpu_count": os.cpu_count(),
    }
    environment["thread_environment"] = {
        key: os.environ.get(key)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    }
    environment["state"] = "recorded"
    return environment


def build_evidence(
    *,
    unit: Mapping[str, Any],
    seeds: Sequence[int],
    c_tokens: Sequence[str],
    protocol_version: str,
    protocol_sha256: str,
    recipe_registry: Mapping[str, Any] | None = None,
    code: Mapping[str, Any] | None = None,
    lock: Mapping[str, Any] | None = None,
    environment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the packet for one unit script invocation.

    ``code``, ``lock`` and ``environment`` are read from the running checkout when
    they are not passed, which is what a driver does; a test passes them so the
    packet it asserts on does not depend on the machine that runs it.  The two
    fixed-shape blocks are checked for their own keys, so an injected mapping
    cannot overwrite the unit or the schema version of the packet it is part of.

    ``run_count`` is the batch's own product, not a count of what happened: an
    oracle has no c axis, so its tokens are empty and each seed is one run.
    """
    seed_values = [int(seed) for seed in seeds]
    token_values = [str(token) for token in c_tokens]
    mechanism = unit.get("mechanism")
    code_block = _checked_keys(
        "code", code if code is not None else code_identity(), allowed=_CODE_KEYS
    )
    lock_block = _checked_keys(
        "dependency lock", lock if lock is not None else dependency_lock(), allowed=_LOCK_KEYS
    )
    environment_block = dict(environment if environment is not None else environment_identity())
    body = {
        "schema_version": SCHEMA_VERSION,
        "unit": {
            "dataset": str(unit["dataset"]),
            "method": str(unit["method"]),
            "training_path": str(unit["training_path"]),
            "mechanism": None if mechanism is None else str(mechanism),
        },
        "seeds": seed_values,
        "c_tokens": token_values,
        "run_count": len(seed_values) * max(len(token_values), 1),
        "protocol_version": protocol_version,
        "protocol_sha256": protocol_sha256,
        **code_block,
        "dependency_lock": lock_block,
        "environment": environment_block,
    }
    if recipe_registry is not None:
        body["recipe_registry"] = dict(recipe_registry)
    return {"batch_id": batch_identity(body), **body}


def batch_identity(payload: Mapping[str, Any]) -> str:
    """Recompute the identity: the strict digest of everything but ``batch_id``.

    ``strict_canonical_hash`` refuses non-finite numbers, which is why the
    identity cannot be quietly built from a value that would not round-trip.
    """
    return strict_canonical_hash(
        {key: value for key, value in payload.items() if key != "batch_id"}
    )


def evidence_name(payload: Mapping[str, Any]) -> str:
    """The file name: the unit spelled out, then the identity's own prefix."""
    unit = _checked_unit(payload.get("unit"))
    mechanism = unit["mechanism"]
    parts = [
        unit["dataset"],
        unit["method"],
        unit["training_path"],
        "oracle" if mechanism is None else mechanism,
    ]
    for part in parts:
        if not part or any(unsafe in part for unsafe in _UNSAFE_IN_NAME):
            raise ValueError(f"batch unit part {part!r} cannot name a file")
        if any(character.isspace() for character in part):
            raise ValueError(f"batch unit part {part!r} cannot name a file")
    batch_id = payload.get("batch_id")
    if not isinstance(batch_id, str) or len(batch_id) < 12:
        raise ValueError("a packet needs a batch_id before it can name a file")
    return f"{'__'.join(parts)}__{batch_id[:12]}.json"


def serialise(payload: Mapping[str, Any]) -> str:
    """The packet's bytes, deterministic so a resume rewrites an identical file."""
    return json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n"


def write(directory: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write the packet atomically and return where it landed.

    The name is derived from the content, so meeting an existing file with
    other bytes means the packet there was edited by hand -- which is exactly
    what this refusal is for.  The bytes are written to a temporary file in the
    same directory and moved into place, so a crash leaves either nothing or a
    whole packet, never a truncated one that would block the batch forever.
    """
    expected = payload.get("batch_id")
    if expected != batch_identity(payload):
        raise ValueError("the packet's batch_id does not match its own contents")
    directory = Path(directory)
    path = directory / evidence_name(payload)
    text = serialise(payload)
    if path.is_file():
        if path.read_bytes() != text.encode("utf-8"):
            raise ValueError(
                f"{path} already holds a batch packet with this identity but different bytes"
            )
        return path
    directory.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 - closed and replaced below
        mode="w",
        encoding="utf-8",
        newline="\n",
        dir=directory,
        prefix=".packet-",
        suffix=".tmp",
        delete=False,
    )
    try:
        with handle:
            handle.write(text)
        os.replace(handle.name, path)
    except BaseException:
        Path(handle.name).unlink(missing_ok=True)
        raise
    return path


def load(path: str | Path) -> dict[str, Any]:
    """Read a packet, check its shape, and prove it still matches its identity."""
    packet = Path(path)
    try:
        payload = json.loads(packet.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"{packet} is not a readable batch evidence packet: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{packet} is not a batch evidence packet")
    missing = [field for field in _REQUIRED_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"{packet} is missing {missing}")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError(
            f"{packet} is a {payload['schema_version']!r} packet; this code reads {SCHEMA_VERSION}"
        )
    if payload["code_commit_state"] not in (CODE_COMMIT_RECORDED, CODE_COMMIT_UNAVAILABLE):
        raise ValueError(f"{packet} records an unknown code_commit_state")
    _checked_unit(payload["unit"])
    if payload.get("batch_id") != batch_identity(payload):
        raise ValueError(f"{packet} does not match its own batch_id; the packet was edited")
    return payload


def require_matches(
    payload: Mapping[str, Any],
    *,
    unit: Mapping[str, Any],
    seeds: Sequence[int],
    c_tokens: Sequence[str],
    protocol_sha256: str,
) -> None:
    """Refuse a run whose own request this packet does not describe.

    ``seeds`` and ``c_tokens`` are compared as sets: which order a batch spells
    them in is not part of what it covers, while a missing or extra member is.
    """
    recorded = _checked_unit(payload.get("unit"))
    requested = _checked_unit(unit)
    for key in _UNIT_KEYS:
        if recorded[key] != requested[key]:
            raise ValueError(
                f"batch evidence describes {key}={recorded[key]!r}, this run asks for "
                f"{requested[key]!r}"
            )
    recorded_seeds = sorted(int(seed) for seed in payload.get("seeds") or ())
    if sorted(int(seed) for seed in seeds) != recorded_seeds:
        raise ValueError(
            f"batch evidence covers seed(s) {payload.get('seeds')}, this run asks for "
            f"{[int(seed) for seed in seeds]}"
        )
    recorded_tokens = sorted(str(token) for token in payload.get("c_tokens") or ())
    if sorted(str(token) for token in c_tokens) != recorded_tokens:
        raise ValueError(
            f"batch evidence covers c {payload.get('c_tokens')}, this run asks for "
            f"{[str(token) for token in c_tokens]}"
        )
    if payload.get("protocol_sha256") != protocol_sha256:
        raise ValueError(
            "batch evidence was planned against another protocol matrix "
            f"({payload.get('protocol_sha256')} != {protocol_sha256})"
        )


def require_same_checkout(
    payload: Mapping[str, Any],
    *,
    code: Mapping[str, Any],
    lock: Mapping[str, Any],
) -> None:
    """Refuse a packet that was written from a different checkout or lock.

    The unit and seed checks alone would accept yesterday's packet for today's
    batch: same unit, same seeds, but code that has since changed.  Comparing
    the two halves of the identity the packet records is what keeps the citation
    true of the bytes that actually train.
    """
    for key in ("code_commit", "code_commit_dirty", "code_commit_state"):
        if payload.get(key) != code.get(key):
            raise ValueError(
                f"batch evidence records {key}={payload.get(key)!r} but this checkout "
                f"is {code.get(key)!r}; the packet belongs to another attempt"
            )
    recorded = payload.get("dependency_lock")
    if not isinstance(recorded, Mapping):
        raise ValueError("batch evidence records no readable dependency lock")
    for key in ("sha256", "state"):
        if recorded.get(key) != lock.get(key):
            raise ValueError(
                f"batch evidence records dependency_lock.{key}={recorded.get(key)!r} but "
                f"this checkout's lock is {lock.get(key)!r}"
            )


def manifest_block(payload: Mapping[str, Any], path: str | Path) -> dict[str, Any]:
    """What a run manifest records about the batch it belongs to.

    The file name locates the packet under the results root and the digest
    proves it was not swapped afterwards; both are host-independent, which an
    absolute path would not be.  The packet must sit in the ``batch_evidence``
    directory the results root names, so the name in a manifest resolves to one
    place instead of to whichever copy a reader happens to find.
    """
    packet = Path(path)
    if not packet.is_file():
        raise ValueError(f"{packet} does not exist; a manifest cannot cite it")
    if packet.parent.name != DIRECTORY:
        raise ValueError(
            f"{packet} is not inside a {DIRECTORY} directory, so its name would not "
            "locate it beside the run trees"
        )
    if packet.name != evidence_name(payload):
        raise ValueError(f"{packet} is not this packet's name; it would not be findable")
    return {
        "schema_version": SCHEMA_VERSION,
        "batch_id": payload["batch_id"],
        "evidence_file": packet.name,
        "evidence_sha256": file_hash(packet),
    }


def _checked_keys(
    name: str, block: Mapping[str, Any], *, allowed: frozenset[str]
) -> dict[str, Any]:
    """The block as a plain dict, refusing keys it is not allowed to carry."""
    unexpected = sorted(set(block) - allowed)
    if unexpected:
        raise ValueError(f"{name} carries unexpected key(s) {unexpected}")
    return dict(block)


def _checked_unit(unit: Any) -> dict[str, str | None]:
    """One unit's four parts, refusing anything a later comparison would misread."""
    if not isinstance(unit, Mapping):
        raise ValueError(f"a batch unit must be a mapping, got {type(unit).__name__}")
    missing = [key for key in _UNIT_KEYS if key not in unit]
    if missing:
        raise ValueError(f"a batch unit is missing {missing}")
    checked: dict[str, str | None] = {}
    for key in _UNIT_KEYS:
        value = unit[key]
        if key == "mechanism":
            if value is not None and not isinstance(value, str):
                raise ValueError("a batch unit's mechanism must be a string or None")
            checked[key] = value
            continue
        if not isinstance(value, str) or not value:
            raise ValueError(f"a batch unit's {key} must be a non-empty string")
        checked[key] = value
    return checked
