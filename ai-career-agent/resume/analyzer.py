"""Small deterministic helpers used to present parsed analysis."""
def get_profile_section(profile: dict, key: str, fallback="Not specified"):
    value = profile.get(key)
    return value if value not in (None, "", []) else fallback
