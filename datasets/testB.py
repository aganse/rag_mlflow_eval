"""Evaluation dataset definitions for the small recent-space-news test dataset."""

EvalRecord = dict[str, dict[str, str | list[str]]]


def get_dataset_name() -> str:
    """Return the MLflow dataset name for the small evaluation set.
    """
    return "space_news_2024-2026_7docs"


def get_url_listings() -> list[str]:
    """Return the source URLs used to build the evaluation corpus.
    These (hopefully permanent) webpages hold specific/tangible space science news that happened since 2024:
    * Firefly's successful Blue Ghost 1 lunar lander, Jan-Mar 2025
    * Astrobotic's failed Peregrine lunar lander, Jan 2024
    * Blue Origin exploded New Glenn rocket and SpaceX's StarShip booster failure, May 2026
    to be used in RAG experiments with the gpt-4o-mini model which has knowledge cutoff of Oct 2023.
    """

    url_listings = [
        "https://fireflyspace.com/missions/blue-ghost-mission-1",
        "https://www.astrobotic.com/astrobotics-peregrine-launches-to-the-moon",
        "https://www.astrobotic.com/update-4-for-peregrine-mission-one",
        "https://www.astrobotic.com/update-17-for-peregrine-mission-one",
        "https://www.smithsonianmag.com/smart-news/doomed-lunar-lander-will-burn-up-in-earths-atmosphere-thursday-180983609",
        "https://www.pbs.org/newshour/show/how-a-blue-origin-rocket-explosion-could-impact-nasas-moon-mission",
        "https://www.cnbc.com/2026/06/01/blue-origin-launchpad-may-not-be-restored-until-2028-nasas-isaacman.html",
    ]
    return url_listings


def get_eval_records() -> list[EvalRecord]:
    """Return evaluation records for the small space news subset."""

    eval_records = [
        {
        "inputs": {
            "query": "When did the Blue Ghost 1 spacecraft launch?"
        },
        "expectations": {
            "expected_response": "Firefly’s Blue Ghost Mission 1 launched on January 15, 2025, aboard a SpaceX Falcon 9 rocket from Kennedy Space Center."
        }
        },

        {
        "inputs": {
            "query": "Did the Blue Ghost 1 spacecraft reach the Moon?  What happened?"
        },
        "expectations": {
            "expected_response": "Yes. Blue Ghost successfully landed on the Moon on March 2, 2025, delivered 10 NASA payloads, completed its planned surface operations, and became the first commercial company to achieve a fully successful lunar soft landing."
        }
        },

        {
        "inputs": {
            "query": "When did the Astrobotic Peregrine spacecraft launch?"
        },
        "expectations": {
            "expected_response": "Astrobotic’s Peregrine Mission One launched on January 8, 2024, on the first flight of ULA’s Vulcan rocket."
        }
        },

        {
        "inputs": {
            "query": "Did the Astrobotic Peregrine spacecraft reach the Moon?  What happened?"
        },
        "expectations": {
            "expected_response": "No. Shortly after launch, Peregrine suffered a propulsion-system anomaly that caused a propellant leak, making a lunar landing impossible. After several days of spacecraft operations and data collection, it was intentionally allowed to reenter Earth’s atmosphere and burn up on January 18, 2024."
        }
        },

        {
        "inputs": {
            "query": "When did the Blue Origin New Glenn rocket explode?  What happened?"
        },
        "expectations": {
            "expected_response": "New Glenn exploded on May 28, 2026, during a pre-launch hot-fire (engine-firing) test at Cape Canaveral. The rocket was destroyed and the launch complex suffered major damage, delaying future New Glenn missions. No injuries were reported."
        }
        },

    ]
    return eval_records
