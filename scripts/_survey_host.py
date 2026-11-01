"""Host-only locking, imported lazily so planning/monitoring remain portable."""

from pathlib import Path


def lock_controller(path):
    """Acquire a real nonblocking POSIX lock; never silently run without one."""
    try:
        import fcntl
    except ImportError as error:
        raise RuntimeError(
            "Background GPU controllers require a POSIX host with fcntl; "
            "planning and read-only monitoring remain available."
        ) from error
    lock = Path(path).open("a")  # noqa: SIM115 -- caller retains the lock for its lifetime
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        lock.close()
        raise
    return lock
