import pytest
from pydantic import ValidationError

from rag_service.auth import (
    ApiKeyAuthenticator,
    generate_api_key,
    hash_api_key,
    main,
    validate_tenant_id,
)
from rag_service.config import Settings


def test_hash_is_hex_sha256() -> None:
    digest = hash_api_key("secret")
    assert digest == "2bb80d537b1da3e38bd30361aa855686bde0eacd7162fef6a25fe97bf527a25b"


def test_generated_keys_are_unique_and_prefixed() -> None:
    first, second = generate_api_key(), generate_api_key()
    assert first != second
    assert first.startswith("rag_")
    assert len(first) > 40


def test_authenticator_maps_keys_to_tenants() -> None:
    authenticator = ApiKeyAuthenticator(
        {hash_api_key("key-a"): "acme", hash_api_key("key-b"): "globex"}, "default"
    )
    assert authenticator.enabled
    assert authenticator.tenant_for("key-a") == "acme"
    assert authenticator.tenant_for("key-b") == "globex"
    assert authenticator.tenant_for("key-c") is None
    assert authenticator.tenant_for(None) is None
    assert authenticator.tenant_for("") is None


def test_without_keys_every_request_is_the_open_tenant() -> None:
    authenticator = ApiKeyAuthenticator({}, "default")
    assert not authenticator.enabled
    assert authenticator.tenant_for(None) == "default"


def test_digests_are_normalised() -> None:
    digest = hash_api_key("k").upper()
    assert ApiKeyAuthenticator({digest: "acme"}, "default").tenant_for("k") == "acme"


@pytest.mark.parametrize("tenant", ["Acme", "a/b", "", "-x", "a" * 64])
def test_invalid_tenant_ids(tenant: str) -> None:
    with pytest.raises(ValueError, match="tenant"):
        validate_tenant_id(tenant)


def test_settings_read_api_keys_from_json(monkeypatch: pytest.MonkeyPatch) -> None:
    digest = hash_api_key("key-a")
    monkeypatch.setenv("RAG_API_KEYS", '{"' + digest + '": "acme"}')
    assert Settings(_env_file=None).api_keys == {digest: "acme"}


@pytest.mark.parametrize("keys", [{"not-a-digest": "acme"}, {hash_api_key("k"): "Not Valid"}])
def test_settings_reject_bad_api_keys(keys: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, api_keys=keys)


def test_cli_new_and_hash(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["new", "acme"]) == 0
    lines = capsys.readouterr().out.splitlines()
    key = lines[0].split()[-1]
    assert hash_api_key(key) in lines[1]
    assert '"acme"' in lines[1]

    assert main(["hash", key]) == 0
    assert capsys.readouterr().out.strip() == hash_api_key(key)
