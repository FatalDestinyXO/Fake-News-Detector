"""
Generate a small synthetic 'fake_news.csv' that mimics the Kaggle Fake News
dataset schema. This lets the pipeline run end-to-end out of the box.
Replace this CSV with the real Kaggle file for production-quality numbers.
"""

import csv
import pathlib
import random

random.seed(42)

OUT = pathlib.Path(__file__).resolve().parent / "fake_news.csv"

# ---------------------------------------------------------------------------
# Templates – chosen so vocabularies differ enough for ML to learn meaningfully.
# ---------------------------------------------------------------------------
REAL_TITLES = [
    "Federal Reserve Raises Interest Rates by 25 Basis Points",
    "NASA Confirms Successful Launch of Artemis Lunar Mission",
    "European Union Reaches Climate Agreement Ahead of Summit",
    "Tech Giant Reports Quarterly Earnings Above Expectations",
    "Researchers Publish New Findings on Alzheimer Treatment",
    "Senate Passes Bipartisan Infrastructure Bill After Long Debate",
    "Olympics Opening Ceremony Draws Record International Audience",
    "Central Bank Holds Rates Steady Amid Inflation Concerns",
    "Scientists Sequence Genome of Endangered Coral Species",
    "Major Automaker Recalls Vehicles Over Software Defect",
    "Supreme Court Hears Arguments in Landmark Privacy Case",
    "United Nations Adopts Resolution on Humanitarian Aid",
    "Global Stock Markets Close Higher Following Economic Data",
    "World Health Organization Issues Updated Vaccination Guidance",
    "University Researchers Develop Improved Solar Cell Efficiency",
    "Government Announces New Investment in Public Transportation",
    "International Space Station Crew Returns Safely to Earth",
    "Economists Predict Moderate Growth in Coming Quarter",
    "Census Bureau Releases Latest Population Demographics Report",
    "New Drug Receives FDA Approval After Clinical Trials",
]
REAL_BODIES = [
    "according to officials familiar with the matter, the announcement was made during a press briefing held earlier today and includes detailed timelines for implementation.",
    "the report, published in a peer reviewed journal, was the result of a multi year study involving researchers from several universities across three continents.",
    "analysts say the move is consistent with previous guidance and reflects ongoing efforts to balance economic growth with long term stability.",
    "the agreement was signed after months of negotiation and is expected to take effect at the beginning of the next fiscal year following ratification.",
    "spokespersons confirmed that the process followed standard regulatory procedures and that further updates will be provided through official channels.",
    "data released by the agency indicates a measurable shift in trends compared with the same period last year, prompting renewed policy discussions.",
    "the initiative builds on existing programs and is funded through a combination of public appropriations and private sector partnerships.",
]

FAKE_TITLES = [
    "SHOCKING: Aliens Spotted Negotiating Trade Deal With World Leaders",
    "Doctors HATE Him: Man Cures Diabetes With This One Weird Spice",
    "Government Secretly Replacing Pets With Robots, Whistleblower Claims",
    "Drinking Bleach Will Make You Immortal, Says Anonymous Expert",
    "Celebrity Reveals Earth Is Actually a Hologram Run by Lizards",
    "Tap Water Contains Mind Control Chemicals, Insider Confirms",
    "Vaccines Turn People Into Magnets Overnight, Mom of Three Warns",
    "Microwaving Phones Charges Them in Seconds, Engineer Quits Job",
    "Moon Landing Was Filmed in My Cousin Basement, He Has the Tapes",
    "5G Towers Cause Trees to Speak Latin at Midnight, Locals Report",
    "BREAKING: Bigfoot Elected Mayor of Small Town in Stunning Upset",
    "Eating Soap Three Times a Day Makes You Lose Weight Instantly",
    "Famous Singer Admits to Being a Time Traveler From 3024",
    "WiFi Signals Cause Plants to Grow in the Shape of Pyramids",
    "Drinking Coffee Standing on One Foot Reverses Aging in Minutes",
    "Birds Are Not Real and Are Actually Drones, Senator Confirms",
    "Eating Pizza for Breakfast Boosts IQ by 200 Points, Study Lies",
    "Secret Society Controls Weather Using Antique Microwave Ovens",
    "Chocolate Banned in 47 Countries After Turning Cats Into Dogs",
    "Man Builds Time Machine in Garage, Visits Tomorrow Three Times",
]
FAKE_BODIES = [
    "sources who refuse to be named claim this is the biggest cover up in history and the mainstream media is hiding the truth from the public.",
    "you wont believe what happened next, scientists are baffled and big pharma is reportedly furious about this incredible miracle discovery.",
    "share this article before they take it down, the elites do not want you to know about this shocking revelation that changes everything.",
    "an anonymous insider revealed everything in a leaked chat that exposes the entire conspiracy, the government has refused to comment so far.",
    "this one simple trick will blow your mind and doctors are absolutely terrified, click here to learn the secret they tried to bury for decades.",
    "the truth has finally come out and it confirms everything we suspected, wake up sheeple and spread the word before it is too late forever.",
    "experts who cannot be named for safety reasons say this is just the tip of the iceberg in a global plot involving every major institution.",
]

AUTHORS = ["A. Reporter", "Staff Writer", "Editor Desk", "J. Doe", "Anonymous", "News Bot"]


def synth(n: int = 600):
    rows = []
    for i in range(n):
        is_fake = i % 2  # roughly balanced classes
        if is_fake:
            title = random.choice(FAKE_TITLES)
            body = " ".join(random.choices(FAKE_BODIES, k=2))
        else:
            title = random.choice(REAL_TITLES)
            body = " ".join(random.choices(REAL_BODIES, k=2))
        rows.append(
            {
                "id": i,
                "title": title,
                "author": random.choice(AUTHORS),
                "text": f"{title}. {body}",
                "label": is_fake,
            }
        )
    random.shuffle(rows)
    return rows


def main() -> None:
    rows = synth(600)
    with OUT.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "title", "author", "text", "label"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT}")


if __name__ == "__main__":
    main()
