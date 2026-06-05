"""Dataset registry helpers for MLflow evaluation datasets."""

from typing import Protocol

from . import civil_war_16, testA, testB

EvalRecord = dict[str, dict[str, str | list[str]]]


class DatasetModule(Protocol):
    """Protocol describing the dataset module interface used by the app."""

    def get_dataset_name(self) -> str:
        """Return the long-form dataset name used by MLflow."""
        ...

    def get_url_listings(self) -> list[str]:
        """Return the source document URLs for retrieval ingestion."""
        ...

    def get_eval_records(self) -> list[EvalRecord]:
        """Return the evaluation records for MLflow GenAI evaluation."""
        ...


DATASET_REGISTRY: dict[str, DatasetModule] = {
    "testA": testA,
    "testB": testB,
    "civil_war_16": civil_war_16,
}


def get_available_dataset_names() -> tuple[str, ...]:
    """Return the available short dataset names that config may select."""

    return tuple(DATASET_REGISTRY)


def get_dataset_module(short_name: str) -> DatasetModule:
    """Return the dataset module registered for a short dataset name.

    Args:
        short_name: The short dataset name from configuration.

    Returns:
        The dataset module implementing the standard dataset interface.

    Raises:
        ValueError: If the short name is not registered.
    """

    try:
        return DATASET_REGISTRY[short_name]
    except KeyError as exc:
        available_names = ", ".join(get_available_dataset_names())
        raise ValueError(
            f"Unsupported dataset '{short_name}'. "
            f"Choose from: {available_names}"
        ) from exc
