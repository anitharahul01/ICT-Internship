import requests
import pandas as pd
import json
import time
from bs4 import BeautifulSoup

BASE_URL = "https://www.coursera.org/search"

HEADERS = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"}

courses = []
QUERY = "courses"
MAX_PAGES = 60  # ~50 courses per page → 3000+

for page in range(1, MAX_PAGES + 1):
    print(f"Fetching page {page}")

    params = {"query": QUERY, "page": page}

    response = requests.get(BASE_URL, headers=HEADERS, params=params)

    if response.status_code != 200:
        print("Failed page:", page)
        break

    soup = BeautifulSoup(response.text, "html.parser")

    scripts = soup.find_all("script", type="application/ld+json")

    for script in scripts:
        try:
            data = json.loads(script.string)

            if isinstance(data, dict) and "itemListElement" in data:
                for item in data["itemListElement"]:
                    course = item.get("item", {})

                    courses.append(
                        {
                            "Course Title": course.get("name"),
                            "Platform": "Coursera",
                            "Category / Domain": course.get("about", {}).get("name"),
                            "Price": "Free / Paid",
                            "Duration": None,  # Not in metadata
                            "Difficulty Level": None,
                            "Instructor / Organization": course.get("provider", {}).get(
                                "name"
                            ),
                            "Rating": course.get("aggregateRating", {}).get(
                                "ratingValue"
                            ),
                            "Ranking": course.get("aggregateRating", {}).get(
                                "ratingValue"
                            ),
                            "Number of Learners": course.get("aggregateRating", {}).get(
                                "ratingCount"
                            ),
                        }
                    )

        except Exception:
            continue

    time.sleep(1)

# Save CSV
df = pd.DataFrame(courses)
df.to_csv("coursera_courses.csv", index=False, encoding="utf-8")

print("Saved coursera_courses.csv")
print("Total records:", len(df))
