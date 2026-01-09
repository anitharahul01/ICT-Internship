import requests
import pandas as pd
from bs4 import BeautifulSoup
import time

BASE_URL = "https://www.classcentral.com/search"

HEADERS = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"}

courses = []

MAX_PAGES = 10  # keep small for ethical use

for page in range(1, MAX_PAGES + 1):
    print(f"Fetching page {page}")

    params = {"q": "data science", "page": page}

    response = requests.get(BASE_URL, headers=HEADERS, params=params)

    if response.status_code != 200:
        print("Failed page", page)
        break

    soup = BeautifulSoup(response.text, "html.parser")

    course_cards = soup.select("li.course-list-course")

    for course in course_cards:
        title = (
            course.select_one("h2").get_text(strip=True)
            if course.select_one("h2")
            else None
        )
        provider = (
            course.select_one(".provider").get_text(strip=True)
            if course.select_one(".provider")
            else None
        )
        rating = (
            course.select_one(".rating").get_text(strip=True)
            if course.select_one(".rating")
            else None
        )

        courses.append(
            {
                "Course Title": title,
                "Platform": "Class Central",
                "Category": "Data Science",
                "Price": "Free / Paid",
                "Duration": None,
                "Instructor / Organization": provider,
                "Rating": rating,
                "Number of Learners": None,
            }
        )

    time.sleep(2)  # polite delay

df = pd.DataFrame(courses)
df.to_csv("classcentral_courses.csv", index=False)

print("Saved classcentral_courses.csv")
print("Total records:", len(df))
