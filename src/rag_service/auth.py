"""API-key authentication that maps each key to a tenant.

Only SHA-256 digests of the keys are configured (``RAG_API_KEYS``), so the settings
never contain a usable credential. Keys are expected to be long random strings such as
those printed by ``rag-apikey new``; a fast hash is appropriate for those, unlike for
passwords chosen by people.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import re
import secrets
import sys
from collections.abc import Mapping, Sequence

TENANT_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,62}$"
_TENANT = re.compile(TENANT_PATTERN)
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
KEY_PREFIX = "rag_"


def hash_api_key(key: str) -> str:
    """Hex SHA-256 digest of ``key``; this is what goes into the configuration."""
    return hashlib.sha256(key.encode()).hexdigest()


def generate_api_key() -> str:
    return KEY_PREFIX + secrets.token_urlsafe(32)


def validate_tenant_id(tenant_id: str) -> str:
    """Tenant ids become directory and collection names, so they are kept simple."""
    if not _TENANT.match(tenant_id):
        raise ValueError(
            f"invalid tenant id {tenant_id!r}: use 1-63 lowercase letters, digits, '_' or '-'"
        )
    return tenant_id


def validate_key_digests(digests: Mapping[str, str]) -> dict[str, str]:
    checked: dict[str, str] = {}
    for digest, tenant_id in digests.items():
        normalised = digest.strip().lower()
        if not _DIGEST.match(normalised):
            raise ValueError("RAG_API_KEYS keys must be hex SHA-256 digests of API keys")
        checked[normalised] = validate_tenant_id(tenant_id)
    return checked


class ApiKeyAuthenticator:
    """Resolve a presented API key to its tenant.

    With no digests configured, authentication is off and every request belongs to
    ``open_tenant``.
    """

    def __init__(self, digests: Mapping[str, str], open_tenant: str) -> None:
        self._digests = validate_key_digests(digests)
        self.open_tenant = open_tenant

    @property
    def enabled(self) -> bool:
        return bool(self._digests)

    def tenant_for(self, key: str | None) -> str | None:
        """Return the tenant owning ``key``, or ``None`` when it is missing or unknown."""
        if not self.enabled:
            return self.open_tenant
        if not key:
            return None
        presented = hash_api_key(key)
        tenant: str | None = None
        # Compare against every digest so the time taken does not depend on which matched.
        for digest, owner in self._digests.items():
            if hmac.compare_digest(presented, digest):
                tenant = owner
        return tenant


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="rag-apikey", description="Create API keys and the digests RAG_API_KEYS expects."
    )
    commands = parser.add_subparsers(dest="command", required=True)
    new = commands.add_parser("new", help="generate a key for a tenant")
    new.add_argument("tenant")
    digest = commands.add_parser("hash", help="print the digest of an existing key")
    digest.add_argument("key")
    args = parser.parse_args(argv)

    if args.command == "new":
        tenant = validate_tenant_id(args.tenant)
        key = generate_api_key()
        print(f"api key (give to the client):   {key}")
        print(f'RAG_API_KEYS entry (configure): "{hash_api_key(key)}": "{tenant}"')
    else:
        print(hash_api_key(args.key))
    return 0


if __name__ == "__main__":
    sys.exit(main())
