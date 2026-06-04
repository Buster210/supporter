from __future__ import annotations

import json
from pathlib import Path

from supporter.tools.browser.profiles import (
    _friendly_name,
    list_profiles,
)


def test_friendly_name_default() -> None:
    assert _friendly_name("Default") == "Personal"


def test_friendly_name_other() -> None:
    assert _friendly_name("Profile 1") == "Profile 1"


def test_list_profiles_no_local_state(tmp_path: Path) -> None:
    assert list_profiles(tmp_path) == []


def test_list_profiles_empty_cache(tmp_path: Path) -> None:
    local_state = tmp_path / "Local State"
    local_state.write_text(json.dumps({"profile": {"info_cache": {}}}))
    assert list_profiles(tmp_path) == []


def test_list_profiles_with_gaia_name(tmp_path: Path) -> None:
    local_state = tmp_path / "Local State"
    local_state.write_text(
        json.dumps(
            {
                "profile": {
                    "info_cache": {
                        "Default": {
                            "name": "John",
                            "gaia_name": "John",
                            "user_name": "john@gmail.com",
                            "active_time": 100.0,
                        }
                    }
                }
            }
        )
    )
    profiles = list_profiles(tmp_path)
    assert len(profiles) == 1
    assert profiles[0].display_name == "John"
    assert profiles[0].email == "john@gmail.com"


def test_list_profiles_gaia_name_with_different_local_name(tmp_path: Path) -> None:
    local_state = tmp_path / "Local State"
    local_state.write_text(
        json.dumps(
            {
                "profile": {
                    "info_cache": {
                        "Profile 1": {
                            "name": "Work",
                            "gaia_name": "John",
                            "user_name": "john@work.com",
                            "active_time": 200.0,
                        }
                    }
                }
            }
        )
    )
    profiles = list_profiles(tmp_path)
    assert len(profiles) == 1
    assert profiles[0].display_name == "John (Work)"


def test_list_profiles_no_gaia_name_uses_name(tmp_path: Path) -> None:
    local_state = tmp_path / "Local State"
    local_state.write_text(
        json.dumps(
            {
                "profile": {
                    "info_cache": {
                        "Profile 1": {
                            "name": "Work Profile",
                            "gaia_name": "",
                            "user_name": "",
                            "active_time": 100.0,
                        }
                    }
                }
            }
        )
    )
    profiles = list_profiles(tmp_path)
    assert len(profiles) == 1
    assert profiles[0].display_name == "Work Profile"


def test_list_profiles_no_gaia_no_name_uses_friendly(tmp_path: Path) -> None:
    local_state = tmp_path / "Local State"
    local_state.write_text(
        json.dumps(
            {
                "profile": {
                    "info_cache": {
                        "Default": {
                            "name": "",
                            "gaia_name": "",
                            "user_name": "",
                            "active_time": 100.0,
                        }
                    }
                }
            }
        )
    )
    profiles = list_profiles(tmp_path)
    assert len(profiles) == 1
    assert profiles[0].display_name == "Personal"


def test_list_profiles_deduplicates_by_email(tmp_path: Path) -> None:
    local_state = tmp_path / "Local State"
    local_state.write_text(
        json.dumps(
            {
                "profile": {
                    "info_cache": {
                        "Profile 1": {
                            "name": "P1",
                            "gaia_name": "",
                            "user_name": "same@test.com",
                            "active_time": 200.0,
                        },
                        "Profile 2": {
                            "name": "P2",
                            "gaia_name": "",
                            "user_name": "same@test.com",
                            "active_time": 100.0,
                        },
                    }
                }
            }
        )
    )
    profiles = list_profiles(tmp_path)
    assert len(profiles) == 1


def test_list_profiles_sorts_by_active_time_desc(tmp_path: Path) -> None:
    local_state = tmp_path / "Local State"
    local_state.write_text(
        json.dumps(
            {
                "profile": {
                    "info_cache": {
                        "Profile 1": {
                            "name": "Old",
                            "gaia_name": "",
                            "user_name": "old@test.com",
                            "active_time": 100.0,
                        },
                        "Profile 2": {
                            "name": "New",
                            "gaia_name": "",
                            "user_name": "new@test.com",
                            "active_time": 200.0,
                        },
                    }
                }
            }
        )
    )
    profiles = list_profiles(tmp_path)
    assert len(profiles) == 2
    assert profiles[0].dir_name == "Profile 1"
    assert profiles[1].dir_name == "Profile 2"


def test_list_profiles_invalid_json(tmp_path: Path) -> None:
    local_state = tmp_path / "Local State"
    local_state.write_text("not json")
    assert list_profiles(tmp_path) == []


def test_list_profiles_non_dict_info_cache(tmp_path: Path) -> None:
    local_state = tmp_path / "Local State"
    local_state.write_text(json.dumps({"profile": {"info_cache": "not a dict"}}))
    assert list_profiles(tmp_path) == []


def test_list_profiles_non_dict_profile_entry(tmp_path: Path) -> None:
    local_state = tmp_path / "Local State"
    local_state.write_text(
        json.dumps({"profile": {"info_cache": {"Profile 1": "not a dict"}}})
    )
    assert list_profiles(tmp_path) == []


def test_list_profiles_profiles_sorted_alphabetically(tmp_path: Path) -> None:
    local_state = tmp_path / "Local State"
    local_state.write_text(
        json.dumps(
            {
                "profile": {
                    "info_cache": {
                        "Zebra": {
                            "name": "Z",
                            "gaia_name": "",
                            "user_name": "z@test.com",
                            "active_time": 100.0,
                        },
                        "Alpha": {
                            "name": "A",
                            "gaia_name": "",
                            "user_name": "a@test.com",
                            "active_time": 100.0,
                        },
                    }
                }
            }
        )
    )
    profiles = list_profiles(tmp_path)
    assert len(profiles) == 2
    assert profiles[0].dir_name == "Alpha"
    assert profiles[1].dir_name == "Zebra"
