"""Tests for directory-based team resolution.

ADR-0011 D2/D3: Team membership is determined by directory location.
- evo/roles/<name>.py → available to all teams (teams = ["*"])
- evo/teams/<team>/roles/<name>.py → available only to <team> (teams = [team_name])
- The `teams` field in @role() is deprecated
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestLoadRoleCatalogDirectoryBased:
    """Tests for _load_role_catalog scanning team-specific directories."""

    @pytest.fixture
    def temp_evo_root(self, tmp_path: Path) -> Path:
        """Create a temporary evo directory structure with global and team-specific roles."""
        evo_root = tmp_path / "evo"
        evo_root.mkdir()

        # Create global roles directory
        global_roles = evo_root / "roles"
        global_roles.mkdir()

        # Create global planner role (available to all teams)
        (global_roles / "planner.py").write_text('''
from palimpsest.runtime import role

@role(
    name="planner",
    description="Global planner available to all teams",
    teams=["default"],  # DEPRECATED: ignored
    role_type="planner",
)
def planner_role(**params):
    pass
''')

        # Create global worker role
        (global_roles / "worker.py").write_text('''
from palimpsest.runtime import role

@role(
    name="worker",
    description="Global worker available to all teams",
    teams=["default"],  # DEPRECATED: ignored
    role_type="worker",
)
def worker_role(**params):
    pass
''')

        # Create global evaluator role
        (global_roles / "evaluator.py").write_text('''
from palimpsest.runtime import role

@role(
    name="evaluator",
    description="Global evaluator available to all teams",
    teams=["default"],  # DEPRECATED: ignored
    role_type="evaluator",
)
def evaluator_role(**params):
    pass
''')

        # Create teams directory structure
        teams_dir = evo_root / "teams"
        teams_dir.mkdir()

        # Create team-specific roles for "backend" team
        backend_team = teams_dir / "backend"
        backend_team.mkdir()
        backend_roles = backend_team / "roles"
        backend_roles.mkdir()

        # Backend-specific planner (overrides global for backend team)
        (backend_roles / "planner.py").write_text('''
from palimpsest.runtime import role

@role(
    name="backend-planner",
    description="Backend-specific planner",
    teams=["backend"],  # DEPRECATED: ignored
    role_type="planner",
)
def backend_planner(**params):
    pass
''')

        # Backend-specific worker
        (backend_roles / "backend-worker.py").write_text('''
from palimpsest.runtime import role

@role(
    name="backend-worker",
    description="Backend-specific worker",
    role_type="worker",
)
def backend_worker(**params):
    pass
''')

        # Backend-specific evaluator
        (backend_roles / "evaluator.py").write_text('''
from palimpsest.runtime import role

@role(
    name="backend-evaluator",
    description="Backend-specific evaluator",
    role_type="evaluator",
)
def backend_evaluator(**params):
    pass
''')

        # Create team-specific roles for "frontend" team
        frontend_team = teams_dir / "frontend"
        frontend_team.mkdir()
        frontend_roles = frontend_team / "roles"
        frontend_roles.mkdir()

        # Frontend-specific planner
        (frontend_roles / "planner.py").write_text('''
from palimpsest.runtime import role

@role(
    name="frontend-planner",
    description="Frontend-specific planner",
    role_type="planner",
)
def frontend_planner(**params):
    pass
''')

        # Frontend-specific worker
        (frontend_roles / "frontend-worker.py").write_text('''
from palimpsest.runtime import role

@role(
    name="frontend-worker",
    description="Frontend-specific worker",
    role_type="worker",
)
def frontend_worker(**params):
    pass
''')

        return evo_root

    def test_load_role_catalog_includes_global_roles(self, temp_evo_root: Path):
        """Global roles from evo/roles/ are included and marked available to all teams."""
        from trenni.supervisor import Supervisor
        from trenni.config import TrenniConfig

        config = TrenniConfig(evo_root=str(temp_evo_root))
        supervisor = Supervisor(config)

        catalog = supervisor._load_role_catalog()

        # Global roles should be present
        assert "planner" in catalog
        assert "worker" in catalog
        assert "evaluator" in catalog

        # Global roles should have teams = ["*"] (available to all)
        assert catalog["planner"]["teams"] == ["*"]
        assert catalog["worker"]["teams"] == ["*"]
        assert catalog["evaluator"]["teams"] == ["*"]

        # source_team should be None for global roles
        assert catalog["planner"]["source_team"] is None
        assert catalog["worker"]["source_team"] is None

    def test_load_role_catalog_includes_team_specific_roles(self, temp_evo_root: Path):
        """Team-specific roles from evo/teams/<team>/roles/ are included."""
        from trenni.supervisor import Supervisor
        from trenni.config import TrenniConfig

        config = TrenniConfig(evo_root=str(temp_evo_root))
        supervisor = Supervisor(config)

        catalog = supervisor._load_role_catalog()

        # Backend team roles should be present
        assert "backend-planner" in catalog
        assert "backend-worker" in catalog
        assert "backend-evaluator" in catalog

        # Backend roles should have teams = ["backend"]
        assert catalog["backend-planner"]["teams"] == ["backend"]
        assert catalog["backend-worker"]["teams"] == ["backend"]
        assert catalog["backend-evaluator"]["teams"] == ["backend"]

        # source_team should indicate the team
        assert catalog["backend-planner"]["source_team"] == "backend"
        assert catalog["backend-worker"]["source_team"] == "backend"

    def test_load_role_catalog_multiple_teams_separate_roles(self, temp_evo_root: Path):
        """Roles from different teams are kept separate."""
        from trenni.supervisor import Supervisor
        from trenni.config import TrenniConfig

        config = TrenniConfig(evo_root=str(temp_evo_root))
        supervisor = Supervisor(config)

        catalog = supervisor._load_role_catalog()

        # Frontend team roles should also be present
        assert "frontend-planner" in catalog
        assert "frontend-worker" in catalog

        # Frontend roles should have teams = ["frontend"]
        assert catalog["frontend-planner"]["teams"] == ["frontend"]
        assert catalog["frontend-worker"]["teams"] == ["frontend"]

    def test_load_role_catalog_no_teams_directory(self, tmp_path: Path):
        """When no teams directory exists, only global roles are loaded."""
        from trenni.supervisor import Supervisor
        from trenni.config import TrenniConfig

        evo_root = tmp_path / "evo"
        evo_root.mkdir()
        global_roles = evo_root / "roles"
        global_roles.mkdir()

        (global_roles / "planner.py").write_text('''
from palimpsest.runtime import role

@role(name="planner", description="Planner", role_type="planner")
def planner_role(**params):
    pass
''')

        config = TrenniConfig(evo_root=str(evo_root))
        supervisor = Supervisor(config)

        catalog = supervisor._load_role_catalog()

        assert "planner" in catalog
        assert catalog["planner"]["teams"] == ["*"]
        # No team-specific roles
        assert all(meta["source_team"] is None for meta in catalog.values())

    def test_load_role_catalog_empty_roles_directory(self, tmp_path: Path):
        """Empty roles directories are handled gracefully."""
        from trenni.supervisor import Supervisor
        from trenni.config import TrenniConfig

        evo_root = tmp_path / "evo"
        evo_root.mkdir()
        (evo_root / "roles").mkdir()

        config = TrenniConfig(evo_root=str(evo_root))
        supervisor = Supervisor(config)

        catalog = supervisor._load_role_catalog()

        assert catalog == {}


class TestResolveTeamDefinitionDirectoryBased:
    """Tests for _resolve_team_definition using directory-based membership."""

    @pytest.fixture
    def temp_evo_root(self, tmp_path: Path) -> Path:
        """Create a minimal evo structure for team resolution tests."""
        evo_root = tmp_path / "evo"
        evo_root.mkdir()

        # Create global roles
        global_roles = evo_root / "roles"
        global_roles.mkdir()

        (global_roles / "planner.py").write_text('''
from palimpsest.runtime import role

@role(name="planner", description="Global planner", role_type="planner")
def planner(**params):
    pass
''')

        (global_roles / "worker.py").write_text('''
from palimpsest.runtime import role

@role(name="worker", description="Global worker", role_type="worker")
def worker(**params):
    pass
''')

        (global_roles / "evaluator.py").write_text('''
from palimpsest.runtime import role

@role(name="evaluator", description="Global evaluator", role_type="evaluator")
def evaluator(**params):
    pass
''')

        return evo_root

    def test_resolve_team_default_uses_global_roles(self, temp_evo_root: Path):
        """Default team uses global roles when no team-specific roles exist."""
        from trenni.supervisor import Supervisor
        from trenni.config import TrenniConfig

        config = TrenniConfig(evo_root=str(temp_evo_root))
        supervisor = Supervisor(config)

        team_def = supervisor._resolve_team_definition("default")

        assert team_def.name == "default"
        assert "planner" in team_def.roles
        assert "worker" in team_def.roles
        assert "evaluator" in team_def.roles
        assert team_def.planner_role == "planner"
        assert team_def.eval_role == "evaluator"

    def test_resolve_team_finds_global_roles_for_any_team(self, temp_evo_root: Path):
        """Any team can use global roles (teams = ["*"])."""
        from trenni.supervisor import Supervisor
        from trenni.config import TrenniConfig

        config = TrenniConfig(evo_root=str(temp_evo_root))
        supervisor = Supervisor(config)

        # "unknown" team should still find global roles
        team_def = supervisor._resolve_team_definition("unknown-team")

        assert team_def.name == "unknown-team"
        assert "planner" in team_def.roles  # Global planner
        assert "worker" in team_def.roles  # Global worker

    def test_resolve_team_prefers_team_specific_planner(self, tmp_path: Path):
        """Team-specific planner is preferred over global planner."""
        from trenni.supervisor import Supervisor
        from trenni.config import TrenniConfig

        evo_root = tmp_path / "evo"
        evo_root.mkdir()

        # Global roles
        global_roles = evo_root / "roles"
        global_roles.mkdir()

        (global_roles / "planner.py").write_text('''
from palimpsest.runtime import role

@role(name="planner", description="Global planner", role_type="planner")
def planner(**params):
    pass
''')

        (global_roles / "worker.py").write_text('''
from palimpsest.runtime import role

@role(name="worker", description="Global worker", role_type="worker")
def worker(**params):
    pass
''')

        # Team-specific roles for "backend"
        teams_dir = evo_root / "teams"
        backend_dir = teams_dir / "backend"
        backend_roles = backend_dir / "roles"
        backend_roles.mkdir(parents=True)

        (backend_roles / "planner.py").write_text('''
from palimpsest.runtime import role

@role(name="backend-planner", description="Backend planner", role_type="planner")
def backend_planner(**params):
    pass
''')

        (backend_roles / "worker.py").write_text('''
from palimpsest.runtime import role

@role(name="backend-worker", description="Backend worker", role_type="worker")
def backend_worker(**params):
    pass
''')

        config = TrenniConfig(evo_root=str(evo_root))
        supervisor = Supervisor(config)

        team_def = supervisor._resolve_team_definition("backend")

        # Backend team should use backend-specific planner
        assert team_def.planner_role == "backend-planner"
        assert "backend-worker" in team_def.roles

    def test_resolve_team_includes_both_global_and_team_roles(self, tmp_path: Path):
        """Team definition includes both global roles (teams=*) and team-specific roles."""
        from trenni.supervisor import Supervisor
        from trenni.config import TrenniConfig

        evo_root = tmp_path / "evo"
        evo_root.mkdir()

        # Global roles
        global_roles = evo_root / "roles"
        global_roles.mkdir()

        (global_roles / "planner.py").write_text('''
from palimpsest.runtime import role

@role(name="planner", description="Global planner", role_type="planner")
def planner(**params):
    pass
''')

        (global_roles / "worker.py").write_text('''
from palimpsest.runtime import role

@role(name="worker", description="Global worker", role_type="worker")
def worker(**params):
    pass
''')

        (global_roles / "evaluator.py").write_text('''
from palimpsest.runtime import role

@role(name="evaluator", description="Global evaluator", role_type="evaluator")
def evaluator(**params):
    pass
''')

        # Team-specific role for "backend"
        teams_dir = evo_root / "teams"
        backend_dir = teams_dir / "backend"
        backend_roles = backend_dir / "roles"
        backend_roles.mkdir(parents=True)

        (backend_roles / "special-worker.py").write_text('''
from palimpsest.runtime import role

@role(name="special-worker", description="Backend special worker", role_type="worker")
def special_worker(**params):
    pass
''')

        config = TrenniConfig(evo_root=str(evo_root))
        supervisor = Supervisor(config)

        team_def = supervisor._resolve_team_definition("backend")

        # Should include both global and team-specific roles
        assert "planner" in team_def.roles  # Global
        assert "worker" in team_def.roles  # Global
        assert "evaluator" in team_def.roles  # Global
        assert "special-worker" in team_def.roles  # Backend-specific

    def test_resolve_team_missing_planner_raises_error(self, tmp_path: Path):
        """Team without a planner raises an error."""
        from trenni.supervisor import Supervisor
        from trenni.config import TrenniConfig

        evo_root = tmp_path / "evo"
        evo_root.mkdir()

        # Only global worker role (no planner)
        global_roles = evo_root / "roles"
        global_roles.mkdir()

        (global_roles / "worker.py").write_text('''
from palimpsest.runtime import role

@role(name="worker", description="Global worker", role_type="worker")
def worker(**params):
    pass
''')

        config = TrenniConfig(evo_root=str(evo_root))
        supervisor = Supervisor(config)

        with pytest.raises(ValueError, match="must have exactly one planner role"):
            supervisor._resolve_team_definition("default")

    def test_resolve_team_missing_worker_raises_error(self, tmp_path: Path):
        """Team without at least one worker raises an error."""
        from trenni.supervisor import Supervisor
        from trenni.config import TrenniConfig

        evo_root = tmp_path / "evo"
        evo_root.mkdir()

        # Only planner role (no workers)
        global_roles = evo_root / "roles"
        global_roles.mkdir()

        (global_roles / "planner.py").write_text('''
from palimpsest.runtime import role

@role(name="planner", description="Planner", role_type="planner")
def planner(**params):
    pass
''')

        config = TrenniConfig(evo_root=str(evo_root))
        supervisor = Supervisor(config)

        with pytest.raises(ValueError, match="must have at least one worker role"):
            supervisor._resolve_team_definition("default")

    def test_resolve_team_multiple_planners_raises_error(self, tmp_path: Path):
        """Team with multiple planners raises an error."""
        from trenni.supervisor import Supervisor
        from trenni.config import TrenniConfig

        evo_root = tmp_path / "evo"
        evo_root.mkdir()

        global_roles = evo_root / "roles"
        global_roles.mkdir()

        (global_roles / "planner1.py").write_text('''
from palimpsest.runtime import role

@role(name="planner1", description="Planner 1", role_type="planner")
def planner1(**params):
    pass
''')

        (global_roles / "planner2.py").write_text('''
from palimpsest.runtime import role

@role(name="planner2", description="Planner 2", role_type="planner")
def planner2(**params):
    pass
''')

        (global_roles / "worker.py").write_text('''
from palimpsest.runtime import role

@role(name="worker", description="Worker", role_type="worker")
def worker(**params):
    pass
''')

        config = TrenniConfig(evo_root=str(evo_root))
        supervisor = Supervisor(config)

        with pytest.raises(ValueError, match="must have exactly one planner role"):
            supervisor._resolve_team_definition("default")


class TestTeamResolutionIgnoresDeprecatedTeamsField:
    """Tests that the deprecated 'teams' field in @role() is ignored."""

    def test_teams_field_ignored_global_roles_available_to_all(self, tmp_path: Path):
        """Global roles are available to all teams regardless of 'teams' field."""
        from trenni.supervisor import Supervisor
        from trenni.config import TrenniConfig

        evo_root = tmp_path / "evo"
        evo_root.mkdir()

        global_roles = evo_root / "roles"
        global_roles.mkdir()

        # Role with deprecated teams=["default"] should still be available to all
        (global_roles / "planner.py").write_text('''
from palimpsest.runtime import role

@role(name="planner", description="Planner", teams=["default"], role_type="planner")
def planner(**params):
    pass
''')

        (global_roles / "worker.py").write_text('''
from palimpsest.runtime import role

@role(name="worker", description="Worker", teams=["default"], role_type="worker")
def worker(**params):
    pass
''')

        config = TrenniConfig(evo_root=str(evo_root))
        supervisor = Supervisor(config)

        catalog = supervisor._load_role_catalog()

        # Global roles should have teams = ["*"], not ["default"]
        assert catalog["planner"]["teams"] == ["*"]
        assert catalog["worker"]["teams"] == ["*"]

        # Both roles should be available to "other-team"
        team_def = supervisor._resolve_team_definition("other-team")
        assert "planner" in team_def.roles
        assert "worker" in team_def.roles

    def test_team_specific_role_only_for_that_team(self, tmp_path: Path):
        """Team-specific roles are only available to that team."""
        from trenni.supervisor import Supervisor
        from trenni.config import TrenniConfig

        evo_root = tmp_path / "evo"
        evo_root.mkdir()

        # Global roles
        global_roles = evo_root / "roles"
        global_roles.mkdir()

        (global_roles / "planner.py").write_text('''
from palimpsest.runtime import role

@role(name="planner", description="Global planner", role_type="planner")
def planner(**params):
    pass
''')

        (global_roles / "worker.py").write_text('''
from palimpsest.runtime import role

@role(name="worker", description="Global worker", role_type="worker")
def worker(**params):
    pass
''')

        # Team-specific role for "backend"
        teams_dir = evo_root / "teams"
        backend_dir = teams_dir / "backend"
        backend_roles = backend_dir / "roles"
        backend_roles.mkdir(parents=True)

        (backend_roles / "backend-worker.py").write_text('''
from palimpsest.runtime import role

@role(name="backend-worker", description="Backend worker", role_type="worker")
def backend_worker(**params):
    pass
''')

        config = TrenniConfig(evo_root=str(evo_root))
        supervisor = Supervisor(config)

        # Backend team should have both global and backend-specific workers
        backend_def = supervisor._resolve_team_definition("backend")
        assert "worker" in backend_def.roles  # Global
        assert "backend-worker" in backend_def.roles  # Backend-specific

        # Other team should only have global worker, not backend-worker
        other_def = supervisor._resolve_team_definition("other-team")
        assert "worker" in other_def.roles  # Global
        assert "backend-worker" not in other_def.roles  # Not available