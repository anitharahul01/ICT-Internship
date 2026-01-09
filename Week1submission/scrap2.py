from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time

options = Options()
options.add_argument("--start-maximized")

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()), options=options
)

driver.get("https://www.classcentral.com/search")
time.sleep(5)

courses = []

cards = driver.find_elements(By.CSS_SELECTOR, "li.course-list-course")

for card in cards:
    try:
        title = card.find_element(By.CSS_SELECTOR, "h2.course-name").text
    except:
        title = None

    try:
        platform = card.find_element(By.CSS_SELECTOR, ".course-provider").text
    except:
        platform = None

    try:
        category = card.find_element(By.CSS_SELECTOR, ".course-subject").text
    except:
        category = None

    try:
        rating = card.find_element(By.CSS_SELECTOR, ".course-rating").text
    except:
        rating = None

    try:
        reviews = card.find_element(By.CSS_SELECTOR, ".course-rating-count").text
    except:
        reviews = None

    courses.append(
        {
            "course_title": title,
            "platform": platform,
            "category": category,
            "price": "Free",
            "duration": None,
            "difficulty_level": None,
            "instructor_org": None,
            "rating": rating,
            "num_reviews": reviews,
        }
    )

driver.quit()

df = pd.DataFrame(courses)
df.to_csv("classcentral_courses.csv", index=False)
print(df.head())
