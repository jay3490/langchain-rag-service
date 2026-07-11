"""Consistency checks for the Kubernetes manifests and the compose file.

These read the YAML in the repository; they do not need a cluster or Docker.
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

from rag_service.config import Settings
from tests.helpers import REPO_ROOT

K8S = REPO_ROOT / "deploy" / "k8s"
BASE = K8S / "base"
PRODUCTION = K8S / "overlays" / "production"
SETTING_NAMES = {f"RAG_{name.upper()}" for name in Settings.model_fields}


def load(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_dir(directory: Path) -> dict[str, dict[str, Any]]:
    """Kind -> manifest for every resource listed in a kustomization."""
    kustomization = load(directory / "kustomization.yaml")
    manifests: dict[str, dict[str, Any]] = {}
    for entry in kustomization["resources"]:
        path = directory / entry
        if path.is_dir():
            continue
        manifest = load(path)
        manifests[manifest["kind"]] = manifest
    return manifests


@pytest.mark.parametrize("directory", [BASE, PRODUCTION])
def test_kustomizations_reference_existing_files(directory: Path) -> None:
    kustomization = load(directory / "kustomization.yaml")
    for entry in kustomization["resources"]:
        assert (directory / entry).exists(), entry
    for patch in kustomization.get("patches", []):
        if "path" in patch:
            assert (directory / patch["path"]).exists(), patch["path"]


def test_base_wires_the_deployment_together() -> None:
    manifests = load_dir(BASE)
    deployment = manifests["Deployment"]
    pod = deployment["spec"]["template"]["spec"]
    (container,) = pod["containers"]

    assert deployment["spec"]["strategy"]["type"] == "Recreate"
    assert deployment["spec"]["replicas"] == 1
    claims = [
        v["persistentVolumeClaim"]["claimName"]
        for v in pod["volumes"]
        if "persistentVolumeClaim" in v
    ]
    assert claims == [manifests["PersistentVolumeClaim"]["metadata"]["name"]]
    mounts = {m["name"]: m["mountPath"] for m in container["volumeMounts"]}
    assert mounts["index"] == "/app/var"
    assert manifests["ConfigMap"]["data"]["RAG_INDEX_DIR"].startswith("/app/var/")

    env_from = container["envFrom"]
    assert env_from[0]["configMapRef"]["name"] == manifests["ConfigMap"]["metadata"]["name"]
    assert env_from[1]["secretRef"] == {"name": "rag-service-secrets", "optional": True}

    port = container["ports"][0]
    for probe in ("startupProbe", "readinessProbe", "livenessProbe"):
        assert container[probe]["httpGet"] == {"path": "/health", "port": port["name"]}
    assert manifests["Service"]["spec"]["ports"][0]["targetPort"] == port["name"]
    assert pod["securityContext"]["runAsUser"] == 10001
    assert container["securityContext"]["readOnlyRootFilesystem"] is True


def test_service_selects_the_pods() -> None:
    manifests = load_dir(BASE)
    pod_labels = manifests["Deployment"]["spec"]["template"]["metadata"]["labels"]
    selector = manifests["Service"]["spec"]["selector"]
    assert selector.items() <= pod_labels.items()


def test_production_overlay_scales_a_stateless_deployment() -> None:
    hpa = load(PRODUCTION / "hpa.yaml")
    patch = load(PRODUCTION / "deployment-patch.yaml")
    base_deployment = load(BASE / "deployment.yaml")

    assert hpa["spec"]["scaleTargetRef"]["name"] == base_deployment["metadata"]["name"]
    assert hpa["spec"]["minReplicas"] >= 2
    assert hpa["spec"]["maxReplicas"] > hpa["spec"]["minReplicas"]
    assert patch["spec"]["strategy"]["type"] == "RollingUpdate"
    (index_volume,) = [
        v for v in patch["spec"]["template"]["spec"]["volumes"] if v["name"] == "index"
    ]
    assert "emptyDir" in index_volume

    config = load(PRODUCTION / "configmap-patch.yaml")["data"]
    assert config["RAG_VECTOR_STORE"] == "qdrant"
    assert config["RAG_REDIS_URL"].startswith("redis://")


@pytest.mark.parametrize("path", [BASE / "configmap.yaml", PRODUCTION / "configmap-patch.yaml"])
def test_configmap_keys_are_real_settings(path: Path) -> None:
    data = load(path)["data"]
    assert set(data) <= SETTING_NAMES, set(data) - SETTING_NAMES
    Settings(_env_file=None, **{k.removeprefix("RAG_").lower(): v for k, v in data.items()})


def test_compose_runs_the_api_with_qdrant_and_redis() -> None:
    compose = load(REPO_ROOT / "docker-compose.yml")
    services = compose["services"]
    assert {"api", "qdrant", "redis"} <= set(services)

    api = services["api"]
    assert set(api["environment"]) <= SETTING_NAMES, set(api["environment"]) - SETTING_NAMES
    assert api["environment"]["RAG_QDRANT_URL"] == "http://qdrant:6333"
    assert api["environment"]["RAG_REDIS_URL"] == "redis://redis:6379/0"
    assert api["depends_on"]["redis"]["condition"] == "service_healthy"
    assert "healthcheck" in services["redis"]
    assert all(":" in services[name]["image"] for name in ("qdrant", "redis"))
