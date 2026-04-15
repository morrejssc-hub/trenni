"""Workspace management for bundle and target sources (ADR-0015, ADR-0021).

Trenni clones repos to ephemeral directories before job dispatch.
Palimpsest receives workspace paths in JobConfig.

Per ADR-0021:
- Bundle workspace uses evolve_selector (free evolution for job-side)
- Same-source target (bundle authoring roles) uses dual worktrees from same bare clone
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from dataclasses import dataclass

from yoitsu_contracts.config import BundleSource, TargetSource
from .config import BundleConfig, TrenniConfig
from .bundle_repository import BundleRepositoryManager

logger = logging.getLogger(__name__)


@dataclass
class PreparedWorkspaces:
    """Result of workspace preparation.

    Contains BundleSource and TargetSource with workspace paths filled.
    """
    bundle_source: BundleSource | None
    target_source: TargetSource | None
    temp_dirs: list[Path]  # For cleanup after job


class WorkspaceManager:
    """Manages ephemeral workspace directories for jobs.

    Per ADR-0015 + ADR-0021:
    - Bundle workspace: loads code/tools for palimpsest (evolve ref)
    - Target workspace: execution area for task
    - Same-source target: bundle authoring roles get dual worktrees from same bare clone

    Uses BundleRepositoryManager for bare clones and worktrees.
    """

    def __init__(
        self,
        config: TrenniConfig,
        bundle_repo_manager: BundleRepositoryManager | None = None,
    ) -> None:
        self.config = config
        self._base_dir = Path(config.workspace_root or "/tmp/yoitsu-workspaces")
        self.bundle_repo = bundle_repo_manager or BundleRepositoryManager(
            workspace_root=self._base_dir,
        )
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def _is_same_bundle_repo(
        self,
        target_uri: str,
        bundle_source_config,
    ) -> bool:
        """Check if target repo is the same bundle repo.

        Per ADR-0021: bundle-authoring roles (implementer/optimizer) declare
        target_source.uri == bundle_source.uri with ref="evolve".

        Args:
            target_uri: Target repository URI
            bundle_source_config: BundleSourceConfig from Trenni config

        Returns:
            True if target is the same bundle repo
        """
        # Normalize URLs for comparison
        bundle_url = bundle_source_config.url
        if bundle_url.startswith("git+"):
            bundle_url = bundle_url[4:]

        target_url = target_uri
        if target_url.startswith("git+"):
            target_url = target_url[4:]

        return target_url == bundle_url

    def prepare(
        self,
        job_id: str,
        bundle: str,
        repo: str,
        init_branch: str,
        bundle_sha: str | None = None,
    ) -> PreparedWorkspaces:
        """Prepare workspaces for a job.

        Per ADR-0021:
        - Bundle workspace: evolve@job_bundle_sha (RO for code loading)
        - Target workspace:
          - Same-source: evolve ref from same bare clone (RW for bundle authoring)
          - External: separate clone (RW for repository authoring)

        Args:
            job_id: Unique job identifier
            bundle: Bundle name (for code loading)
            repo: Target repo URL (for task execution)
            init_branch: Branch to checkout for target
            bundle_sha: Optional SHA pin for bundle

        Returns:
            PreparedWorkspaces with filled BundleSource/TargetSource
        """
        temp_dirs: list[Path] = []
        bundle_source = None
        target_source = None

        bundle_config = self.config.bundles.get(bundle) if bundle else None

        # Prepare bundle workspace (for code loading) — uses evolve_selector
        if bundle and bundle_config and bundle_config.source.url:
            bundle_source, bundle_temp = self._prepare_bundle_workspace(
                job_id, bundle, bundle_config, bundle_sha
            )
            if bundle_temp:
                temp_dirs.append(bundle_temp)

        # Prepare target workspace (for task execution)
        if repo:
            # Same-source detection (ADR-0021: bundle authoring roles)
            if bundle_config and self._is_same_bundle_repo(repo, bundle_config.source):
                target_source, target_temp = self._prepare_same_source_target(
                    job_id, bundle, bundle_config, init_branch
                )
            else:
                # External target repo
                target_source, target_temp = self._prepare_external_target(
                    job_id, repo, init_branch
                )
            if target_temp:
                temp_dirs.append(target_temp)

        return PreparedWorkspaces(
            bundle_source=bundle_source,
            target_source=target_source,
            temp_dirs=temp_dirs,
        )

    def _prepare_bundle_workspace(
        self,
        job_id: str,
        bundle: str,
        bundle_config: BundleConfig,
        bundle_sha: str | None,
    ) -> tuple[BundleSource | None, Path | None]:
        """Prepare bundle workspace using evolve_selector.

        Per ADR-0021: bundle workspace is RO for code loading.
        Uses BundleRepositoryManager worktrees.

        Args:
            job_id: Job identifier
            bundle: Bundle name
            bundle_config: Bundle configuration
            bundle_sha: Optional SHA pin

        Returns:
            (BundleSource, worktree_path) or (None, None) on failure
        """
        url = bundle_config.source.url
        evolve_ref = bundle_config.source.evolve_selector or bundle_config.source.selector

        # Ensure bare clone exists
        try:
            self.bundle_repo.ensure_bare_clone(bundle, url)
        except Exception as e:
            logger.error(f"Failed to ensure bare clone for {bundle}: {e}")
            return None, None

        # Fetch evolve ref and get SHA
        try:
            if bundle_sha:
                sha = bundle_sha
            else:
                sha = self.bundle_repo.fetch(bundle, evolve_ref)
        except Exception as e:
            logger.error(f"Failed to fetch bundle {bundle} ref {evolve_ref}: {e}")
            return None, None

        # Create worktree (RO for code loading)
        # Per ADR-0020: Child task IDs use '/' as hierarchy separator (e.g. "abc123/fv7o-eval").
        # Replace '/' with '-' in prefix to avoid nested directory creation in tempfile.mkdtemp.
        try:
            worktree = self.bundle_repo.create_worktree(
                bundle, sha, writable=False, prefix=f"bundle-{job_id.replace('/', '-')}"
            )
        except Exception as e:
            logger.error(f"Failed to create bundle worktree for {bundle}: {e}")
            return None, None

        return BundleSource(
            name=bundle,
            repo_uri=url,
            selector=evolve_ref,
            resolved_ref=sha,
            workspace=str(worktree),
        ), worktree

    def _prepare_same_source_target(
        self,
        job_id: str,
        bundle: str,
        bundle_config: BundleConfig,
        init_branch: str,
    ) -> tuple[TargetSource | None, Path | None]:
        """Prepare target workspace from same bundle repo.

        Per ADR-0021: bundle-authoring roles use target_source.uri == bundle_source.uri.
        Creates RW worktree from same bare clone at evolve ref.

        Args:
            job_id: Job identifier
            bundle: Bundle name
            bundle_config: Bundle configuration
            init_branch: Target branch (should be evolve)

        Returns:
            (TargetSource, worktree_path) or (None, None) on failure
        """
        url = bundle_config.source.url
        evolve_ref = init_branch or bundle_config.source.evolve_selector or "evolve"

        # Fetch evolve ref for target
        try:
            sha = self.bundle_repo.fetch(bundle, evolve_ref)
        except Exception as e:
            logger.error(f"Failed to fetch target ref {evolve_ref} for {bundle}: {e}")
            return None, None

        # Create RW worktree at branch (not SHA) to avoid detached HEAD
        # Per ADR-0020: Replace '/' in job_id to avoid nested directory creation.
        try:
            worktree = self.bundle_repo.create_worktree(
                bundle, sha, writable=True, prefix=f"target-{job_id.replace('/', '-')}"
            )
            # Materialize a local branch from the fetched remote ref so plain
            # `git push` works from finalize().
            subprocess.run(
                ["git", "checkout", "-B", evolve_ref, f"origin/{evolve_ref}"],
                cwd=worktree,
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["git", "config", f"branch.{evolve_ref}.remote", "origin"],
                cwd=worktree,
                capture_output=True,
                check=True,
            )
            subprocess.run(
                ["git", "config", f"branch.{evolve_ref}.merge", f"refs/heads/{evolve_ref}"],
                cwd=worktree,
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to checkout branch {evolve_ref} in target worktree: {e}")
            if worktree:
                self.bundle_repo.remove_worktree(worktree)
            return None, None
        except Exception as e:
            logger.error(f"Failed to create target worktree for {bundle}: {e}")
            return None, None

        return TargetSource(
            repo_uri=url,
            selector=evolve_ref,
            resolved_ref=sha,
            workspace=str(worktree),
        ), worktree

    def _prepare_external_target(
        self,
        job_id: str,
        repo: str,
        init_branch: str,
    ) -> tuple[TargetSource | None, Path | None]:
        """Prepare target workspace for external repo.

        Legacy path for non-bundle-authoring roles.

        Args:
            job_id: Job identifier
            repo: Target repo URL
            init_branch: Branch to checkout

        Returns:
            (TargetSource, workspace_path) or (None, None) on failure
        """
        ws_dir = self._base_dir / f"{job_id}-target"
        ws_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Normalize URL
            git_url = repo
            if git_url.startswith("git+"):
                git_url = git_url[4:]

            logger.info(f"Cloning target {git_url} ({init_branch}) to {ws_dir}")

            subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", init_branch, git_url, str(ws_dir)],
                check=True,
                capture_output=True,
            )

            return TargetSource(
                repo_uri=repo,
                selector=init_branch,
                workspace=str(ws_dir),
            ), ws_dir

        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone target {repo}: {e.stderr.decode() if e.stderr else e}")
            shutil.rmtree(ws_dir, ignore_errors=True)
            return None, None

    def cleanup(self, temp_dirs: list[Path]) -> None:
        """Cleanup ephemeral workspaces after job.

        Args:
            temp_dirs: List of temp directories to remove
        """
        for ws_dir in temp_dirs:
            try:
                # Try worktree removal first (for BundleRepositoryManager worktrees)
                self.bundle_repo.remove_worktree(ws_dir)
            except Exception:
                # Fallback to direct removal (for legacy clones)
                try:
                    shutil.rmtree(ws_dir)
                    logger.info(f"Cleaned up workspace: {ws_dir}")
                except Exception as e:
                    logger.warning(f"Failed to cleanup {ws_dir}: {e}")
