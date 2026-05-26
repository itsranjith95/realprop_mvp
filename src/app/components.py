def section_header(title: str) -> str:
    return f"## {title}"


def status_badge(status: str) -> str:
    return status.replace("_", " ").title()