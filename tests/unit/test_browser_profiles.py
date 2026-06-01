from __future__ import annotations

from pathlib import Path

from supporter.tools.browser import profiles


def test_chrome_profile_dataclass() -> None:
    p = profiles.ChromeProfile(
        dir_name="Profile 1", display_name="Work", email="user@example.com"
    )
    assert p.dir_name == "Profile 1"
    assert p.display_name == "Work"
    assert p.email == "user@example.com"


def test_list_profiles_reads_local_state(tmp_path: Path) -> None:
    local_state = tmp_path / "Local State"
    local_state.write_text(
        """{
            "profile": {
                "info_cache": {
                    "Profile 1": {"name": "Work", "user_name": "user@corp.com"},
                    "Profile 2": {"name": "Personal", "user_name": ""}
                }
            }
        }"""
    )

    result = profiles.list_profiles(tmp_path)
    assert len(result) == 2
    assert result[0].dir_name == "Profile 1"
    assert result[0].display_name == "Work"
    assert result[0].email == "user@corp.com"
    assert result[1].dir_name == "Profile 2"
    assert result[1].display_name == "Personal"
    assert result[1].email == ""


def test_list_profiles_sorts_signed_in_first(tmp_path: Path) -> None:
    local_state = tmp_path / "Local State"
    local_state.write_text(
        """{
            "profile": {
                "info_cache": {
                    "Zebra": {"name": "Z", "user_name": ""},
                    "Alpha": {"name": "A", "user_name": "a@b.com"},
                    "Beta": {"name": "B", "user_name": ""}
                }
            }
        }"""
    )

    result = profiles.list_profiles(tmp_path)
    assert result[0].dir_name == "Alpha"
    assert result[1].dir_name == "Beta"
    assert result[2].dir_name == "Zebra"


def test_list_profiles_returns_empty_when_missing(tmp_path: Path) -> None:
    result = profiles.list_profiles(tmp_path)
    assert result == []


def test_list_profiles_returns_empty_on_invalid_json(tmp_path: Path) -> None:
    local_state = tmp_path / "Local State"
    local_state.write_text("not valid json")
    result = profiles.list_profiles(tmp_path)
    assert result == []


def test_list_profiles_returns_empty_on_missing_info_cache(tmp_path: Path) -> None:
    local_state = tmp_path / "Local State"
    local_state.write_text('{"profile": {}}')
    result = profiles.list_profiles(tmp_path)
    assert result == []


def test_list_profiles_handles_missing_name(tmp_path: Path) -> None:
    local_state = tmp_path / "Local State"
    local_state.write_text(
        """{
            "profile": {
                "info_cache": {
                    "Profile 1": {"user_name": "user@example.com"}
                }
            }
        }"""
    )
    result = profiles.list_profiles(tmp_path)
    assert result[0].display_name == "user@example.com"


def test_list_profiles_defaults_to_personal_for_default_profile(tmp_path: Path) -> None:
    local_state = tmp_path / "Local State"
    local_state.write_text(
        """{
            "profile": {
                "info_cache": {
                    "Default": {}
                }
            }
        }"""
    )
    result = profiles.list_profiles(tmp_path)
    assert result[0].display_name == "Personal"


def test_list_profiles_combines_gaia_and_local_name(tmp_path: Path) -> None:
    local_state = tmp_path / "Local State"
    local_state.write_text(
        """{
            "profile": {
                "info_cache": {
                    "Profile 1": {
                        "name": "Work",
                        "gaia_name": "Ritu Pal",
                        "user_name": "user@example.com"
                    }
                }
            }
        }"""
    )
    result = profiles.list_profiles(tmp_path)
    assert result[0].display_name == "Ritu Pal (Work)"


def test_list_profiles_deduplicates_by_email(tmp_path: Path) -> None:
    local_state = tmp_path / "Local State"
    local_state.write_text(
        """{
            "profile": {
                "info_cache": {
                    "Profile 1": {
                        "name": "Older Profile",
                        "user_name": "user@example.com",
                        "active_time": 1000.0
                    },
                    "Profile 2": {
                        "name": "Newer Profile",
                        "user_name": "user@example.com",
                        "active_time": 2000.0
                    },
                    "Profile 3": {
                        "name": "Unsigned Profile",
                        "user_name": "",
                        "active_time": 3000.0
                    }
                }
            }
        }"""
    )
    result = profiles.list_profiles(tmp_path)
    assert len(result) == 2
    assert result[0].dir_name == "Profile 2"
    assert result[0].display_name == "Newer Profile"
    assert result[0].email == "user@example.com"
    assert result[1].dir_name == "Profile 3"
    assert result[1].display_name == "Unsigned Profile"
    assert result[1].email == ""
