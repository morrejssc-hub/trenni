"""Bundle repository manager for bare clone and worktree lifecycle.

Per ADR-0021:
- Each bundle has a bare clone at ~/trenni/bundles/<name>.git
- Control-plane capabilities use master@switched_sha (PR-gated)
- Job-side capabilities use evolve@job_bundle_sha (free evolution)
- Worktrees are materialized on-demand per job and cleaned after completion
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class BundleRepositoryManager:
    """Manages bare clones and worktrees for bundle repositories.

    Per ADR-0021 A.4: master ref materialized via git worktree.
    Each bundle has one bare clone; worktrees are created per job.
    """

    BUNDLES_DIR = Path("~/trenni/bundles").expanduser()

    bundles: dict[str, str] = field(default_factory=dict)  # bundle_name -> repo_url
    # Worktrees are created here so job containers (separate podman containers)
    # can bind-mount them. Must be a host-visible path, not /tmp inside the
    # container (which is not bind-mounted to the host).
    workspace_root: Path = field(default_factory=lambda: Path(tempfile.gettempdir()))

    def __post_init__(self) -> None:
        self.BUNDLES_DIR.mkdir(parents=True, exist_ok=True)
        self.workspace_root.mkdir(parents=True, exist_ok=True)

    def ensure_bare_clone(self, bundle: str, url: str) -> Path:
        """Ensure a bare clone exists for the bundle.

        Creates if missing, otherwise verifies it points to the correct URL.

        Args:
            bundle: Bundle name (used as directory name)
            url: Git repository URL

        Returns:
            Path to the bare clone directory
        """
        bare_path = self.BUNDLES_DIR / f"{bundle}.git"
        self.bundles[bundle] = url

        if bare_path.is_dir():
            # Verify URL matches
            try:
                result = subprocess.run(
                    ["git", "config", "--get", "remote.origin.url"],
                    cwd=bare_path,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0 and result.stdout.strip() == url:
                    logger.debug(f"Bare clone for {bundle} already exists at {bare_path}")
                    return bare_path
                # URL mismatch — recreate
                logger.warning(f"Bare clone URL mismatch for {bundle}, recreating")
                subprocess.run(["rm", "-rf", str(bare_path)], check=True)
            except Exception as e:
                logger.warning(f"Failed to verify bare clone: {e}, recreating")
                subprocess.run(["rm", "-rf", str(bare_path)], check=False)

        # Create bare clone
        logger.info(f"Creating bare clone for {bundle} at {bare_path}")
        subprocess.run(
            ["git", "clone", "--bare", url, str(bare_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        return bare_path

    def fetch(self, bundle: str, ref: str) -> str:
        """Fetch a ref from the bundle's bare clone and return its SHA.

        Args:
            bundle: Bundle name
            ref: Git ref to fetch (e.g., "master", "evolve", "HEAD")

        Returns:
            SHA of the fetched ref
        """
        bare_path = self.BUNDLES_DIR / f"{bundle}.git"
        if not bare_path.is_dir():
            raise ValueError(f"No bare clone for bundle {bundle}")

        # Prefer materializing a remote-tracking ref so downstream worktrees can
        # branch from origin/<ref> and configure upstream tracking normally.
        fetch_result = subprocess.run(
            ["git", "fetch", "origin", f"{ref}:refs/remotes/origin/{ref}"],
            cwd=bare_path,
            capture_output=True,
            text=True,
            check=False,
        )
        if fetch_result.returncode != 0:
            subprocess.run(
                ["git", "fetch", "origin", ref],
                cwd=bare_path,
                capture_output=True,
                text=True,
                check=True,
            )

        # Bare clones may expose fetched refs either as origin/<ref> or as a
        # local branch ref. Resolve whichever exists.
        for candidate in (f"origin/{ref}", ref, "FETCH_HEAD"):
            result = subprocess.run(
                ["git", "rev-parse", candidate],
                cwd=bare_path,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                return result.stdout.strip()

        raise subprocess.CalledProcessError(
            returncode=128,
            cmd=["git", "rev-parse", f"origin/{ref}"],
            stderr=f"Unable to resolve fetched ref {ref!r} in {bare_path}",
        )

    def create_worktree(
        self,
        bundle: str,
        sha: str,
        *,
        writable: bool = False,
        prefix: str = "wt",
    ) -> Path:
        """Create a worktree at a specific SHA from the bundle's bare clone.

        Per ADR-0021 A.4: worktrees are ephemeral per-job.

        Args:
            bundle: Bundle name
            sha: Git SHA to checkout
            writable: True for target workspace (RW), False for bundle workspace (RO)
            prefix: Prefix for worktree directory name

        Returns:
            Path to the worktree directory
        """
        bare_path = self.BUNDLES_DIR / f"{bundle}.git"
        if not bare_path.is_dir():
            raise ValueError(f"No bare clone for bundle {bundle}")

        # Create unique worktree directory under workspace_root so the path is
        # visible on the host and can be bind-mounted into job containers.
        worktree_base = Path(tempfile.mkdtemp(prefix=f"{prefix}-{bundle}-", dir=self.workspace_root))
        logger.debug(f"Creating worktree for {bundle}@{sha} at {worktree_base}")

        subprocess.run(
            ["git", "worktree", "add", str(worktree_base), sha],
            cwd=bare_path,
            capture_output=True,
            text=True,
            check=True,
        )

        return worktree_base

    def remove_worktree(self, path: Path) -> None:
        """Remove a worktree and its parent directory.

        Args:
            path: Path to the worktree directory
        """
        if not path.is_dir():
            logger.debug(f"Worktree {path} does not exist, skipping removal")
            return

        # Find the bare clone that owns this worktree
        # Worktrees are in temp dirs, so we need to prune from any bare clone
        # that might reference them
        try:
            # Clean up the worktree using git worktree remove
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(path)],
                cwd=path,
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception:
            pass  # Might not be a valid git dir anymore

        # Remove the directory
        try:
            subprocess.run(["rm", "-rf", str(path)], check=False)
            logger.debug(f"Removed worktree at {path}")
        except Exception as e:
            logger.warning(f"Failed to remove worktree {path}: {e}")

    def prune_worktrees(self, bundle: str) -> None:
        """Prune stale worktree references from the bare clone.

        Called periodically to clean up references to deleted worktrees.
        """
        bare_path = self.BUNDLES_DIR / f"{bundle}.git"
        if not bare_path.is_dir():
            return

        subprocess.run(
            ["git", "worktree", "prune"],
            cwd=bare_path,
            capture_output=True,
            text=True,
            check=False,
        )
