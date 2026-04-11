"""Workspace management for bundle and target sources (ADR-0015).

Trenni clones repos to ephemeral directories before job dispatch.
Palimpsest receives workspace paths in JobConfig.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass

from yoitsu_contracts.config import BundleSource, TargetSource
from .config import BundleConfig, TrenniConfig

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
    
    Per ADR-0015:
    - Bundle workspace: loads code/tools for palimpsest
    - Target workspace: execution area for task
    
    Trenni clones repos, Palimpsest just uses the paths.
    """
    
    def __init__(self, config: TrenniConfig) -> None:
        self.config = config
        self._base_dir = Path(tempfile.gettempdir()) / "yoitsu-workspaces"
        self._base_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare(
        self,
        job_id: str,
        bundle: str,
        repo: str,
        init_branch: str,
        bundle_sha: str | None = None,
    ) -> PreparedWorkspaces:
        """Prepare workspaces for a job.
        
        Args:
            job_id: Unique job identifier
            bundle: Bundle name (for code loading)
            repo: Target repo URL (for task execution)
            init_branch: Branch to checkout
            bundle_sha: Optional SHA pin for bundle
            
        Returns:
            PreparedWorkspaces with filled BundleSource/TargetSource
        """
        temp_dirs: list[Path] = []
        bundle_source = None
        target_source = None
        
        # Prepare bundle workspace (for code loading)
        if bundle:
            bundle_config = self.config.bundles.get(bundle)
            if bundle_config and bundle_config.source.url:
                bundle_ws = self._clone_bundle(job_id, bundle, bundle_config, bundle_sha)
                if bundle_ws:
                    temp_dirs.append(bundle_ws)
                    bundle_source = BundleSource(
                        name=bundle,
                        repo_uri=bundle_config.source.url,
                        selector=bundle_config.source.selector,
                        resolved_ref=bundle_sha or "",
                        workspace=str(bundle_ws),
                    )
        
        # Prepare target workspace (for task execution)
        if repo:
            target_ws = self._clone_target(job_id, repo, init_branch)
            if target_ws:
                temp_dirs.append(target_ws)
                target_source = TargetSource(
                    repo_uri=repo,
                    selector=init_branch,
                    workspace=str(target_ws),
                )
        
        return PreparedWorkspaces(
            bundle_source=bundle_source,
            target_source=target_source,
            temp_dirs=temp_dirs,
        )
    
    def _clone_bundle(
        self,
        job_id: str,
        bundle: str,
        bundle_config: BundleConfig,
        bundle_sha: str | None,
    ) -> Path | None:
        """Clone bundle repo for code loading.
        
        Args:
            job_id: Job identifier
            bundle: Bundle name
            bundle_config: Bundle configuration
            bundle_sha: Optional SHA pin (if None, resolve from selector)
            
        Returns:
            Path to cloned bundle workspace, or None if failed
        """
        url = bundle_config.source.url
        selector = bundle_config.source.selector
        
        # Create unique temp dir for this job's bundle
        ws_dir = self._base_dir / f"{job_id}-bundle"
        ws_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Clone
            logger.info(f"Cloning bundle {bundle} from {url} to {ws_dir}")
            
            # Parse URL scheme
            if url.startswith("git+file://"):
                # Local file path
                local_path = url.replace("git+file://", "")
                subprocess.run(
                    ["git", "clone", "--depth", "1", local_path, str(ws_dir)],
                    check=True,
                    capture_output=True,
                )
            elif url.startswith("git+ssh://") or url.startswith("git+https://"):
                # Remote URL
                git_url = url.replace("git+", "")
                subprocess.run(
                    ["git", "clone", "--depth", "1", "--branch", selector, git_url, str(ws_dir)],
                    check=True,
                    capture_output=True,
                )
            else:
                # Assume it's a direct git URL
                subprocess.run(
                    ["git", "clone", "--depth", "1", "--branch", selector, url, str(ws_dir)],
                    check=True,
                    capture_output=True,
                )
            
            # Checkout specific SHA if provided
            if bundle_sha:
                subprocess.run(
                    ["git", "checkout", bundle_sha],
                    cwd=ws_dir,
                    check=True,
                    capture_output=True,
                )
            
            return ws_dir
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone bundle {bundle}: {e.stderr.decode() if e.stderr else e}")
            shutil.rmtree(ws_dir, ignore_errors=True)
            return None
    
    def _clone_target(
        self,
        job_id: str,
        repo: str,
        init_branch: str,
    ) -> Path | None:
        """Clone target repo for task execution.
        
        Args:
            job_id: Job identifier
            repo: Repo URL
            init_branch: Branch to checkout
            
        Returns:
            Path to cloned target workspace, or None if failed
        """
        # Create unique temp dir for this job's target
        ws_dir = self._base_dir / f"{job_id}-target"
        ws_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            logger.info(f"Cloning target {repo} ({init_branch}) to {ws_dir}")
            
            subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", init_branch, repo, str(ws_dir)],
                check=True,
                capture_output=True,
            )
            
            return ws_dir
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone target {repo}: {e.stderr.decode() if e.stderr else e}")
            shutil.rmtree(ws_dir, ignore_errors=True)
            return None
    
    def cleanup(self, temp_dirs: list[Path]) -> None:
        """Cleanup ephemeral workspaces after job.
        
        Args:
            temp_dirs: List of temp directories to remove
        """
        for ws_dir in temp_dirs:
            try:
                shutil.rmtree(ws_dir)
                logger.info(f"Cleaned up workspace: {ws_dir}")
            except Exception as e:
                logger.warning(f"Failed to cleanup {ws_dir}: {e}")