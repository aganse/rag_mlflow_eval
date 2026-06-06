"""Evaluation dataset definitions for the recent-space-news corpus."""

EvalRecord = dict[str, dict[str, str | list[str]]]


def get_dataset_name() -> str:
    """Return the MLflow dataset name for the small evaluation set.
    """
    return "space_news"


def get_url_listings() -> list[str]:
    """Return the source URLs used to build the evaluation corpus.
    These webpages hold specific/tangible space science news that happened since 2024.
    To be used in RAG experiments with the gpt-4o-mini model which has knowledge cutoff Oct 2023.
    """

    url_listings = [
        "https://en.wikipedia.org/wiki/Blue_Ghost_Mission_1",
        "https://en.wikipedia.org/wiki/Peregrine_Mission_One",
        "https://en.wikipedia.org/wiki/Vera_C._Rubin_Observatory",
        "https://www.nasa.gov/mission/artemis-ii",
        "https://www.cnn.com/2026/04/11/science/moon-mission-takeaways-artemis-2",
        "https://rubinobservatory.org/news/first-imagery-rubin",
        "https://www.astrobotic.com/astrobotics-peregrine-launches-to-the-moon",
        "https://www.astrobotic.com/update-4-for-peregrine-mission-one",
        "https://www.astrobotic.com/update-17-for-peregrine-mission-one",
        "https://www.smithsonianmag.com/smart-news/doomed-lunar-lander-will-burn-up-in-earths-atmosphere-thursday-180983609",
    ]
    return url_listings


def get_eval_records() -> list[EvalRecord]:
    """Return evaluation records for the small space news subset."""

    eval_records = [
        {
        "inputs": {
            "query": "On what date did the Firefly Blue Ghost 1 spacecraft launch?"
        },
        "expectations": {
            "expected_response": "Firefly’s Blue Ghost Mission 1 launched on January 15, 2025."
        }
        },

        {
        "inputs": {
            "query": "Did the Firefly Blue Ghost 1 spacecraft reach the Moon?  What happened and on what date?"
        },
        "expectations": {
            "expected_response": "Yes. Blue Ghost successfully landed on the Moon on March 2, 2025."
        }
        },

        {
        "inputs": {
            "query": "On what date did the Astrobotic Peregrine spacecraft launch?"
        },
        "expectations": {
            "expected_response": "Astrobotic’s Peregrine Mission One launched on January 8, 2024."
        }
        },

        {
        "inputs": {
            "query": "Did the Astrobotic Peregrine spacecraft reach the Moon?  What ultimately happened to it?"
        },
        "expectations": {
            "expected_response": "No. Shortly after launch, Peregrine suffered a propulsion-system anomaly that caused a propellant leak, making a lunar landing impossible. After several days of spacecraft operations and data collection, it was allowed to reenter Earth’s atmosphere and burn up on January 18, 2024."
        }
        },

        {
        "inputs": {
            "query": "On what date did the Vera Rubin Observatory release its first images from its full telescope?"
        },
        "expectations": {
            "expected_response": "The Vera C. Rubin Observatory released its first images on June 23, 2025."
        }
        },

        {
        "inputs": {
            "query": "What did the Vera Rubin Observatory see in its first 10 hours of test observations?"
        },
        "expectations": {
            "expected_response": "The early images from the Vera C. Rubin Observatory captured millions of galaxies, Milky Way stars, and thousands of asteroids."
        }
        },

        {
        "inputs": {
            "query": "What were the launch and splashdown dates of the NASA Artemis II spacecraft mission?"
        },
        "expectations": {
            "expected_response": "The launch date for the Artemis II mission is April 1, 2026, and the splashdown date is April 10, 2026."
        }
        },

    ]
    return eval_records
