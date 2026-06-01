from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ChromeProfile:
    dir_name: str
    display_name: str
    email: str


def _friendly_name(dir_name: str) -> str:
    if dir_name == "Default":
        return "Personal"
    return dir_name


def list_profiles(user_data_dir: Path) -> list[ChromeProfile]:
    local_state = user_data_dir / "Local State"
    if not local_state.exists():
        return []

    try:
        content = local_state.read_text(encoding="utf-8")
        data = json.loads(content)
    except OSError, json.JSONDecodeError:
        return []

    info_cache = data.get("profile", {}).get("info_cache", {})
    if not isinstance(info_cache, dict):
        return []

    raw_profiles = []
    for dir_name, info in info_cache.items():
        if not isinstance(info, dict):
            continue
        name = info.get("name", "")
        gaia_name = info.get("gaia_name", "")
        email = info.get("user_name", "") or ""
        active_time = info.get("active_time", 0.0)

        if gaia_name:
            if name and name != gaia_name:
                display_name = f"{gaia_name} ({name})"
            else:
                display_name = gaia_name
        else:
            display_name = name or email or _friendly_name(dir_name)

        raw_profiles.append(
            {
                "dir_name": dir_name,
                "display_name": display_name,
                "email": email,
                "active_time": active_time,
            }
        )

    raw_profiles.sort(key=lambda x: x["active_time"], reverse=True)

    seen_emails = set()
    profiles: list[ChromeProfile] = []
    for p in raw_profiles:
        email = p["email"]
        if email:
            if email in seen_emails:
                continue
            seen_emails.add(email)
        profiles.append(
            ChromeProfile(
                dir_name=p["dir_name"],
                display_name=p["display_name"],
                email=email,
            )
        )

    profiles.sort(key=lambda p: (not bool(p.email), p.dir_name))
    return profiles
