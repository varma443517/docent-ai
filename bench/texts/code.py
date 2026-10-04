def ready_pods(pods: list[dict]) -> list[str]:
    """Return the names of pods whose Ready condition is True."""
    names = []
    for pod in pods:
        conditions = pod.get("status", {}).get("conditions", [])
        if any(c["type"] == "Ready" and c["status"] == "True" for c in conditions):
            names.append(pod["metadata"]["name"])
    return sorted(names)
