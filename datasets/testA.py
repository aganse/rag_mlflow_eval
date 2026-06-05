"""Evaluation dataset definitions for the small Civil War document subset."""

EvalRecord = dict[str, dict[str, str | list[str]]]


def get_dataset_name() -> str:
    """Return the MLflow dataset name for the small evaluation set.
    """

    return "usmilestonedocs_civilwar_3docsubset"


def get_url_listings() -> list[str]:
    """Return the source URLs used to build the evaluation corpus.
    """

    url_listings = [
        "https://www.archives.gov/milestone-documents/13th-amendment#transcript",
        "https://www.archives.gov/milestone-documents/14th-amendment#transcript",
        "https://www.archives.gov/milestone-documents/15th-amendment#transcript",
    ]
    return url_listings


def get_eval_records() -> list[EvalRecord]:
    """Return evaluation records for the small Civil War document subset.
    """

    eval_records = [
        {
        "inputs": {
            "query": "What did the 13th Amendment do regarding slavery?"
        },
        "expectations": {
            "expected_response": "The 13th Amendment abolished slavery and involuntary servitude in the United States except as punishment for a crime after lawful conviction."
        }
        },

        {
        "inputs": {
            "query": "What are three major constitutional principles established by the 14th Amendment?"
        },
        # "expectations": {
        #     "expected_response": [
        #         "It established birthright citizenship, prohibited states from depriving persons of due process of law, and required states to provide equal protection of the laws."
        #     ]
        # }
        "expectations": {
            "expected_facts": [
                "It established birthright citizenship.",
                "It prohibited states from depriving persons of due process of law.",
                "It required states to provide equal protection of the laws."
            ]
        }
        },
        {
        "inputs": {
            "query": "What voting rights protection is provided by the 15th Amendment?"
        },
        "expectations": {
            "expected_response": "The 15th Amendment prohibits denying or abridging the right to vote on account of race, color, or previous condition of servitude."
        }
        },
    ]
    return eval_records
