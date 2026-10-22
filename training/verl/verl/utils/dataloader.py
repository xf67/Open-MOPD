
import logging
import sys
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@contextmanager
def managed_dataloaders(owner, *names):
    """Join loader workers before their owning Ray actor exits, including on failure."""
    try:
        yield
    finally:
        active_error = sys.exc_info()[1]
        cleanup_error = None
        for name in names:
            loader = getattr(owner, name, None)
            iterator = getattr(loader, "_iterator", None)
            # StatefulDataLoader has no public close API. Single-process and
            # unstarted loaders have no worker shutdown method and need no join.
            shutdown = getattr(iterator, "_shutdown_workers", None)
            if not callable(shutdown):
                continue
            try:
                shutdown()
            except Exception as error:
                logger.exception("Failed to close DataLoader %s", name)
                if cleanup_error is None:
                    cleanup_error = error
        # Report cleanup failures without replacing the original training error.
        if cleanup_error is not None and active_error is None:
            raise cleanup_error
