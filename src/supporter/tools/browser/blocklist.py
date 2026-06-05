from __future__ import annotations

import os
from urllib.parse import urlsplit

# Ad / tracker exchanges and data brokers. Session-recording and tag analytics
# (hotjar, fullstory, clarity, segment) are deliberately NOT blocked: a site that
# expects its own beacon to fire reads the absence as a bot signal.
AD_HOSTS: frozenset[str] = frozenset(
    {
        "googleadservices.com",
        "doubleclick.net",
        "googlesyndication.com",
        "adnxs.com",
        "adsrvr.org",
        "amazon-adsystem.com",
        "moatads.com",
        "taboola.com",
        "outbrain.com",
        "criteo.com",
        "criteo.net",
        "pubmatic.com",
        "openx.net",
        "rubiconproject.com",
        "indexww.com",
        "casalemedia.com",
        "sharethrough.com",
        "bidswitch.net",
        "demdex.net",
        "adskeeper.com",
        "adroll.com",
        "quantserve.com",
        "scorecardresearch.com",
        "bluekai.com",
        "krxd.net",
        "liadm.com",
        "bounceexchange.com",
        "wunderkind.co",
    }
)

BLOCKED_TYPES: frozenset[str] = frozenset({"media", "font"})


def host_listed(host: str) -> bool:
    host = host.lower()
    if not host:
        return False
    return any(host == h or host.endswith("." + h) for h in AD_HOSTS)


def host_blocked(url: str) -> bool:
    return host_listed(urlsplit(url).hostname or "")


def should_block_resources() -> bool:
    raw = os.getenv("BROWSER_BLOCK_RESOURCES", "1").strip().lower()
    return raw not in ("0", "false", "no", "")
