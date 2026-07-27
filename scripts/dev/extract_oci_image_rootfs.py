"""Extract a public Docker/OCI image rootfs without a Docker daemon."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tarfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

DOCKER_HUB_REGISTRY = "registry-1.docker.io"
DOCKER_HUB_AUTH = "https://auth.docker.io/token"
INDEX_ACCEPT = ", ".join(
    [
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ]
)
MANIFEST_ACCEPT = ", ".join(
    [
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        required=True,
        help="Repository, e.g. openfoldconsortium/openfold3",
    )
    parser.add_argument("--reference", required=True, help="Tag or digest")
    parser.add_argument("--platform", default="linux/amd64")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output-rootfs", type=Path, required=True)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Resolve manifests/config/layers without downloading layer blobs",
    )
    parser.add_argument(
        "--start-layer",
        type=int,
        default=0,
        help="Skip lower layer indexes; use only when rootfs already has them",
    )
    parser.add_argument("--registry", default=DOCKER_HUB_REGISTRY)
    return parser.parse_args()


def request_json(
    url: str,
    headers: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            payload = json.load(response)
            return payload, dict(response.headers)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body[:1000]}") from exc


def request_bytes(url: str, headers: dict[str, str], dest: Path) -> int:
    req = urllib.request.Request(url, headers=headers)
    try:
        with (
            urllib.request.urlopen(req, timeout=300) as response,
            dest.open("wb") as out,
        ):
            shutil.copyfileobj(response, out)
        return dest.stat().st_size
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} for {url}: {body[:1000]}") from exc


def parse_bearer_challenge(header: str) -> dict[str, str]:
    prefix = "Bearer "
    if not header.startswith(prefix):
        raise RuntimeError(f"Unsupported registry auth challenge: {header}")
    values: dict[str, str] = {}
    for item in header.removeprefix(prefix).split(","):
        key, _, raw_value = item.strip().partition("=")
        values[key] = raw_value.strip('"')
    if "realm" not in values:
        raise RuntimeError(f"Registry auth challenge has no realm: {header}")
    return values


def dockerhub_token(repo: str) -> str:
    url = f"{DOCKER_HUB_AUTH}?service=registry.docker.io"
    url += f"&scope=repository:{repo}:pull"
    payload, _ = request_json(url)
    return str(payload["token"])


def challenge_token(registry: str, repo: str, reference: str) -> str | None:
    req = urllib.request.Request(
        manifest_url(registry, repo, reference),
        headers={"Accept": INDEX_ACCEPT},
    )
    try:
        with urllib.request.urlopen(req, timeout=120):
            return None
    except urllib.error.HTTPError as exc:
        if exc.code != 401:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Expected auth challenge from {registry}, got HTTP {exc.code}: "
                f"{body[:1000]}"
            ) from exc
        challenge = exc.headers.get("WWW-Authenticate")
        if challenge is None:
            raise RuntimeError(
                f"{registry} returned 401 without WWW-Authenticate"
            ) from exc

    params = parse_bearer_challenge(challenge)
    query = {
        "service": params.get("service", registry),
        "scope": params.get("scope", f"repository:{repo}:pull"),
    }
    url = params["realm"] + "?" + urllib.parse.urlencode(query)
    payload, _ = request_json(url)
    return str(payload.get("token") or payload["access_token"])


def registry_token(registry: str, repo: str, reference: str) -> str | None:
    if registry == DOCKER_HUB_REGISTRY:
        return dockerhub_token(repo)
    return challenge_token(registry, repo, reference)


def blob_url(registry: str, repo: str, digest: str) -> str:
    return f"https://{registry}/v2/{repo}/blobs/{digest}"


def manifest_url(registry: str, repo: str, reference: str) -> str:
    return f"https://{registry}/v2/{repo}/manifests/{reference}"


def auth_headers(token: str | None, accept: str | None = None) -> dict[str, str]:
    headers = {}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    if accept:
        headers["Accept"] = accept
    return headers


def split_platform(platform: str) -> tuple[str, str]:
    parts = platform.split("/", 1)
    if len(parts) != 2:
        raise ValueError("platform must look like linux/amd64")
    return parts[0], parts[1]


def resolve_manifest(
    args: argparse.Namespace,
    token: str | None,
) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    payload, headers = request_json(
        manifest_url(args.registry, args.repo, args.reference),
        auth_headers(token, INDEX_ACCEPT),
    )
    index_digest = headers.get("Docker-Content-Digest") or args.reference
    manifests = payload.get("manifests") or []
    if not manifests:
        return index_digest, payload, None

    want_os, want_arch = split_platform(args.platform)
    for entry in manifests:
        platform = entry.get("platform", {})
        if platform.get("os") == want_os and platform.get("architecture") == want_arch:
            digest = entry["digest"]
            resolved, _ = request_json(
                manifest_url(args.registry, args.repo, digest),
                auth_headers(token, MANIFEST_ACCEPT),
            )
            return digest, resolved, payload
    raise RuntimeError(f"No manifest for platform {args.platform}")


def safe_extract_member(
    tf: tarfile.TarFile,
    member: tarfile.TarInfo,
    root: Path,
) -> None:
    target = (root / member.name).resolve()
    root_resolved = root.resolve()
    common = os.path.commonpath([str(root_resolved), str(target)])
    if common != str(root_resolved):
        raise RuntimeError(f"Refusing path traversal in layer member {member.name}")
    tf.extract(member, root, filter="fully_trusted")


def apply_whiteout(root: Path, member_name: str) -> bool:
    path = Path(member_name)
    name = path.name
    parent = root / path.parent
    if name == ".wh..wh..opq":
        if parent.exists():
            for child in parent.iterdir():
                if child.name.startswith(".wh."):
                    continue
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink(missing_ok=True)
        return True
    if name.startswith(".wh."):
        target = parent / name.removeprefix(".wh.")
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink(missing_ok=True)
        return True
    return False


def extract_layer(layer_path: Path, root: Path) -> None:
    with tarfile.open(layer_path, "r:*") as tf:
        for member in tf.getmembers():
            if apply_whiteout(root, member.name):
                continue
            safe_extract_member(tf, member, root)


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output_rootfs.mkdir(parents=True, exist_ok=True)

    token = registry_token(args.registry, args.repo, args.reference)
    manifest_digest, manifest, index = resolve_manifest(args, token)
    (args.work_dir / "manifest_resolved_digest.txt").write_text(manifest_digest + "\n")
    write_json(args.work_dir / "manifest_resolved.json", manifest)
    if index is not None:
        write_json(args.work_dir / "manifest_index.json", index)

    config_digest = manifest["config"]["digest"]
    config_path = args.work_dir / "config.json"
    if not args.metadata_only:
        request_bytes(
            blob_url(args.registry, args.repo, config_digest),
            auth_headers(token),
            config_path,
        )

    layer_records = []
    for index_no, layer in enumerate(manifest.get("layers", [])):
        digest = layer["digest"]
        size = int(layer.get("size") or 0)
        safe_digest = digest.replace(":", "_")
        layer_path = args.work_dir / f"layer_{index_no:03d}_{safe_digest}.tar.gz"
        record = {
            "index": index_no,
            "digest": digest,
            "mediaType": layer.get("mediaType"),
            "declared_size": size,
            "path": str(layer_path),
        }
        if not args.metadata_only and index_no >= args.start_layer:
            print(f"download layer {index_no}: {digest} ({size} bytes)", flush=True)
            layer_token = registry_token(args.registry, args.repo, args.reference)
            if layer_path.exists() and layer_path.stat().st_size == size:
                actual_size = size
                print(f"reuse layer {index_no}: {layer_path}", flush=True)
            else:
                actual_size = request_bytes(
                    blob_url(args.registry, args.repo, digest),
                    auth_headers(layer_token),
                    layer_path,
                )
            record["downloaded_size"] = actual_size
            print(f"extract layer {index_no}: {layer_path}", flush=True)
            extract_layer(layer_path, args.output_rootfs)
        layer_records.append(record)
    write_json(args.work_dir / "layers.json", layer_records)
    print(
        json.dumps(
            {
                "manifest_digest": manifest_digest,
                "config_digest": config_digest,
                "layers": len(layer_records),
                "declared_layer_bytes": sum(r["declared_size"] for r in layer_records),
                "metadata_only": args.metadata_only,
                "rootfs": str(args.output_rootfs),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
