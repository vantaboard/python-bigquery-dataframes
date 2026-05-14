# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pytest fixtures for ``samples/snippets``.

When ``BIGQUERY_EMULATOR_HOST`` is set (see go-googlesql ``.envrc``), snippet
tests use ``AnonymousCredentials``, REST and gRPC endpoints from the
environment, and the project from ``GOOGLE_CLOUD_PROJECT`` / ``GCLOUD_PROJECT``
/ ``EMULATOR_PROJECT_ID`` so they can run against the local BigQuery emulator.

Most snippet tests depend on public tables (for example ``ml_datasets.penguins``)
that the go-googlesql emulator seeds when ``GOOGLESQL_BOOTSTRAP_EXTENDED_PUBLIC_SAMPLES``
is set (see ``storage/bootstrap_public_sample.go``). When ``BIGQUERY_EMULATOR_HOST``
is set, ``conftest.py`` wires anonymous clients and endpoint overrides; tests that
still need unsupported GCP APIs or BigFrames SQL remain skipped via
``pytest_collection_modifyitems``. The allowlist runs snippet modules that
primarily exercise ``bigquery-public-data.ml_datasets.penguins``; unset
``BIGQUERY_EMULATOR_HOST`` for the full suite on GCP.
"""

from __future__ import annotations

import os
from typing import Generator, Iterator

import google.auth.credentials
import google.api_core.client_options as client_options_lib
from google.cloud import bigquery, storage
from google.cloud.bigquery._helpers import BIGQUERY_EMULATOR_HOST
import pytest
import test_utils.prefixer

import bigframes.pandas as bpd

prefixer = test_utils.prefixer.Prefixer(
    "python-bigquery-dataframes", "samples/snippets"
)

routine_prefixer = test_utils.prefixer.Prefixer("bigframes", "")


def _bigquery_emulator_enabled() -> bool:
    return bool(os.environ.get(BIGQUERY_EMULATOR_HOST, "").strip())


def _emulator_project() -> str:
    return (
        os.environ.get("GOOGLE_CLOUD_PROJECT")
        or os.environ.get("GCLOUD_PROJECT")
        or os.environ.get("EMULATOR_PROJECT_ID")
        or "dev"
    )


def _emulator_rest_api_endpoint() -> str:
    raw = os.environ.get(BIGQUERY_EMULATOR_HOST, "").strip()
    if raw.startswith(("http://", "https://")):
        return raw.rstrip("/")
    return "http://{}".format(raw).rstrip("/")


def _grpc_endpoint_from_env() -> str:
    return os.environ.get("BIGQUERY_STORAGE_GRPC_ENDPOINT", "").strip()


def _configure_bigframes_for_emulator() -> None:
    credentials: google.auth.credentials.Credentials = (
        google.auth.credentials.AnonymousCredentials()
    )
    project = _emulator_project()
    bpd.options.bigquery.credentials = credentials
    bpd.options.bigquery.project = project
    bpd.options.bigquery.skip_bq_connection_check = True
    overrides: dict[str, str] = {"bqclient": _emulator_rest_api_endpoint()}
    grpc_host = _grpc_endpoint_from_env()
    if grpc_host:
        overrides["bqstoragereadclient"] = grpc_host
        overrides["bqstoragewriteclient"] = grpc_host
        overrides["bqconnectionclient"] = grpc_host
    bpd.options.bigquery.client_endpoints_override = overrides


# Snippet tests that only need tables/APIs the go-googlesql emulator does not
# provide are skipped when BIGQUERY_EMULATOR_HOST is set (see storage bootstrap).
# BigFrames-generated SQL may still exceed the emulator parser for some modules;
# keep an allowlist of snippet tests that primarily use seeded ``ml_datasets.penguins``.
_ALLOW_ON_EMULATOR = frozenset(
    {
        "set_options_test.py",
        "quickstart_test.py",
        "explore_query_result_test.py",
        "load_data_from_bigquery_test.py",
        "pandas_methods_test.py",
        "regression_model_test.py",
        "bigquery_modules_test.py",
        "performance_optimizations_test.py",
        "linear_regression_tutorial_test.py",
        "clustering_model_test.py",
        "udf_test.py",
    }
)


def pytest_configure(config: pytest.Config) -> None:
    if _bigquery_emulator_enabled():
        _configure_bigframes_for_emulator()
    config.addinivalue_line(
        "markers",
        "requires_real_gcp: needs live BigQuery public datasets or GCP APIs "
        "(auto-skipped when "
        + BIGQUERY_EMULATOR_HOST
        + " is set)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if not _bigquery_emulator_enabled():
        return
    skip = pytest.mark.skip(
        reason=(
            "Snippet needs BigQuery public datasets or APIs not available on the "
            "go-googlesql emulator. Unset BIGQUERY_EMULATOR_HOST to run against GCP."
        )
    )
    for item in items:
        try:
            modname = item.path.name  # type: ignore[attr-defined]
        except AttributeError:
            modname = os.path.basename(str(item.fspath))  # type: ignore[attr-defined]
        if modname not in _ALLOW_ON_EMULATOR:
            item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def cleanup_datasets(bigquery_client: bigquery.Client) -> None:
    for dataset in bigquery_client.list_datasets():
        if prefixer.should_cleanup(dataset.dataset_id):
            bigquery_client.delete_dataset(
                dataset, delete_contents=True, not_found_ok=True
            )


@pytest.fixture(scope="session")
def bigquery_client() -> bigquery.Client:
    if _bigquery_emulator_enabled():
        credentials: google.auth.credentials.Credentials = (
            google.auth.credentials.AnonymousCredentials()
        )
        project = _emulator_project()
        opts = client_options_lib.ClientOptions(api_endpoint=_emulator_rest_api_endpoint())
        return bigquery.Client(
            credentials=credentials, project=project, client_options=opts
        )
    return bigquery.Client()


@pytest.fixture(scope="session")
def storage_client(project_id: str) -> storage.Client:
    if _bigquery_emulator_enabled():
        credentials: google.auth.credentials.Credentials = (
            google.auth.credentials.AnonymousCredentials()
        )
        return storage.Client(project=project_id, credentials=credentials)
    return storage.Client(project=project_id)


@pytest.fixture(scope="session")
def project_id(bigquery_client: bigquery.Client) -> str:
    return bigquery_client.project


@pytest.fixture(scope="session")
def gcs_bucket(storage_client: storage.Client) -> Generator[str, None, None]:
    bucket_name = "bigframes_blob_test_with_data_wipeout"

    if _bigquery_emulator_enabled():
        bucket = storage_client.bucket(bucket_name)
        if not bucket.exists():
            bucket.create()
        yield bucket_name
        return

    yield bucket_name

    bucket = storage_client.get_bucket(bucket_name)
    for blob in bucket.list_blobs():
        blob.delete()


@pytest.fixture(scope="session")
def gcs_bucket_snippets(storage_client: storage.Client) -> Generator[str, None, None]:
    bucket_name = "bigframes_blob_test_snippet_with_data_wipeout"

    if _bigquery_emulator_enabled():
        bucket = storage_client.bucket(bucket_name)
        if not bucket.exists():
            bucket.create()
        yield bucket_name
        return

    yield bucket_name

    bucket = storage_client.get_bucket(bucket_name)
    for blob in bucket.list_blobs():
        blob.delete()


@pytest.fixture(autouse=True)
def reset_session() -> None:
    """An autouse fixture ensuring each sample runs in a fresh session.

    This allows us to have samples that query data in different locations.
    """
    bpd.reset_session()
    bpd.options.bigquery.location = None
    if _bigquery_emulator_enabled():
        _configure_bigframes_for_emulator()


@pytest.fixture(scope="session")
def dataset_id(bigquery_client: bigquery.Client, project_id: str) -> Iterator[str]:
    dataset_id = prefixer.create_prefix()
    full_dataset_id = f"{project_id}.{dataset_id}"
    dataset = bigquery.Dataset(full_dataset_id)
    bigquery_client.create_dataset(dataset)
    yield dataset_id
    bigquery_client.delete_dataset(dataset, delete_contents=True, not_found_ok=True)


@pytest.fixture(scope="session")
def dataset_id_eu(bigquery_client: bigquery.Client, project_id: str) -> Iterator[str]:
    dataset_id = prefixer.create_prefix()
    full_dataset_id = f"{project_id}.{dataset_id}"
    dataset = bigquery.Dataset(full_dataset_id)
    dataset.location = "EU"
    bigquery_client.create_dataset(dataset)
    yield dataset_id
    bigquery_client.delete_dataset(dataset, delete_contents=True, not_found_ok=True)


@pytest.fixture
def random_model_id(
    bigquery_client: bigquery.Client, project_id: str, dataset_id: str
) -> Iterator[str]:
    """Create a new table ID each time, so random_model_id can be used as
    target for load jobs.
    """
    random_model_id = prefixer.create_prefix()
    full_model_id = f"{project_id}.{dataset_id}.{random_model_id}"
    yield full_model_id
    bigquery_client.delete_model(full_model_id, not_found_ok=True)


@pytest.fixture
def random_model_id_eu(
    bigquery_client: bigquery.Client, project_id: str, dataset_id_eu: str
) -> Iterator[str]:
    """
    Create a new table ID each time, so random_model_id_eu can be used
    as a target for load jobs.
    """
    random_model_id_eu = prefixer.create_prefix()
    full_model_id = f"{project_id}.{dataset_id_eu}.{random_model_id_eu}"
    yield full_model_id
    bigquery_client.delete_model(full_model_id, not_found_ok=True)


@pytest.fixture
def routine_id() -> Iterator[str]:
    """Create a new BQ routine ID each time, so random_routine_id can be used as
    target for udf creation.
    """
    random_routine_id = routine_prefixer.create_prefix()
    yield random_routine_id
