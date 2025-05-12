from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
import json
import time
from common_functions import _write_output_data
from selenium.webdriver.chrome.service import Service
import os
import subprocess


def ingest_vnexpress(
    minio_client,
    output_file: str,
    url: str = "https://vnexpress.net/",
    output_bucket: str = "landing",
):
    # Configure Chrome for stability
    options = Options()

    options.binary_location = "/usr/bin/chromium"
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    service = Service(executable_path="/usr/bin/chromedriver")

    # List to store metadata for all articles
    all_articles_metadata = []

    try:
        print("Creating new Chrome browser session...")
        driver = webdriver.Chrome(service=service, options=options)
    except Exception as e:
        raise Exception(f"Could not initialize Chrome browser: {e}")

    try:
        try:
            print("Opening the main webpage...")
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    driver.get(url)
                    time.sleep(5)
                    break
                except Exception as e:
                    if attempt < max_retries - 1:
                        print(f"Attempt {attempt + 1} failed. Retrying...")
                    else:
                        raise e

            print("Getting the page source after JavaScript execution...")
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, "html.parser")

            print("Extracting news headlines and links...")
            articles = soup.find_all(
                "h3", class_="title-news"
            )  # Adjust selector as needed
            print(f"Found {len(articles)} articles on the main page.")
            count = 0
        except Exception as e:
            raise Exception(f"Error accessing the main page: {e}")

        for article in articles:
            if count > 2:
                print("Reached the limit of 3 articles. Stopping further processing.")
                break

            # Extract headline and link
            headline = article.get_text(strip=True)
            link = article.find("a")["href"]
            print(f"Processing article {count + 1}: {headline}, Link: {link}")

            # Check if the URL is a video link before navigating
            if "video.vnexpress.net" in link:
                print(f"Skipping video link: {link}")
                continue  # Skip to the next article

            try:
                print(f"Navigating to article link: {link}...")
                driver.get(link)
                time.sleep(5)  # Wait for the page to load
            except Exception as e:
                print(f"Error accessing article: {link}, Error: {e}")
                continue

            print("Getting the article content...")
            article_page = BeautifulSoup(driver.page_source, "html.parser")

            print("Extracting metadata from <script type='application/ld+json'>...")
            metadata_script = article_page.find("script", type="application/ld+json")
            metadata_json = {}
            if metadata_script:
                metadata_json = json.loads(metadata_script.string)
                print("Metadata extracted from JSON script.")

            print("Extracting all <meta> elements...")
            meta_elements = article_page.find_all("meta")
            for meta in meta_elements:
                if meta.get("name") or meta.get("property"):
                    key = meta.get("name") or meta.get("property")
                    value = meta.get("content", "").strip()
                    metadata_json[key] = value

            print("Extracting the full article content...")
            content = article_page.find("article", class_="fck_detail")
            if content:
                print("Extracting related articles...")
                related_articles = []
                for section_class in ["box-tinlienquanv2", "box-tinlienquanv2_gocnhin"]:
                    section = content.find("div", class_=section_class)
                    if section:
                        for item in section.find_all("article", class_="item-news"):
                            title_tag = item.find(["h3", "h4"], class_="title-news")
                            link = title_tag.find("a")["href"] if title_tag else None
                            title = (
                                title_tag.get_text(strip=True) if title_tag else None
                            )
                            description = item.find("p", class_="description")
                            related_articles.append(
                                {
                                    "title": title,
                                    "description": (
                                        description.get_text(strip=True)
                                        if description
                                        else None
                                    ),
                                    "link": link,
                                }
                            )
                metadata_json["relatedArticles"] = related_articles

                print("Removing unwanted sections from the article content...")
                unwanted_sections = content.find_all(
                    ["div", "article"],
                    class_=[
                        "box-newsletter-new",
                        "box-tinlienquanv2",
                        "box-tinlienquanv2_gocnhin",
                    ],
                )
                for section in unwanted_sections:
                    section.decompose()

                print("Concatenating content with spaces...")
                full_text = " ".join(
                    [p.get_text(strip=True) for p in content.find_all(["p", "div"])]
                )
                metadata_json["articleContent"] = full_text

                print("Extracting author information...")
                author_section = article_page.find("div", class_="info-detail-tg")
                if author_section:
                    author_name_tag = author_section.find("h3", class_="title-news")
                    author_name = (
                        author_name_tag.get_text(strip=True)
                        if author_name_tag
                        else None
                    )
                    author_profile_link = (
                        author_name_tag.find("a")["href"]
                        if author_name_tag and author_name_tag.find("a")
                        else None
                    )
                    author_description_tag = author_section.find(
                        "p", class_="description"
                    )
                    author_description = (
                        author_description_tag.get_text(strip=True)
                        if author_description_tag
                        else None
                    )

                    metadata_json["author"] = {
                        "name": author_name,
                        "profile_link": author_profile_link,
                        "description": author_description,
                    }
                else:
                    print(
                        "Author section not found. Checking the last sentence of the article content..."
                    )
                    sentences = full_text.split(".")
                    if len(sentences) > 1:
                        possible_author = sentences[-1].strip()
                        metadata_json["author"] = {
                            "name": possible_author,
                            "profile_link": None,
                            "description": None,
                        }
                        full_text = ".".join(sentences[:-1]).strip()
                    else:
                        metadata_json["author"] = {
                            "name": "Unknown",
                            "profile_link": None,
                            "description": None,
                        }
            else:
                print("Content not found for the article.")
                metadata_json["articleContent"] = "Content not found"
                metadata_json["relatedArticles"] = []
                metadata_json["author"] = {
                    "name": "Unknown",
                    "profile_link": None,
                    "description": None,
                }

            if "name" in metadata_json["author"] and metadata_json["author"]["name"]:
                author_name = metadata_json["author"]["name"]
                if author_name in full_text:
                    print(
                        f"Removing author's name '{author_name}' from the article content..."
                    )
                    full_text = full_text.replace(author_name, "").strip()

            metadata_json["articleContent"] = full_text

            print("Determining the @type field dynamically...")
            metadata_json["@type"] = metadata_json.get("its_subsection", "Article")

            print(f"Adding metadata for article {count + 1} to the list...")
            all_articles_metadata.append(metadata_json)
            print(f"Metadata for article {count + 1} processed.")
            count += 1

        _write_output_data(
            minio_client,
            output_file=output_file,
            output_bucket=output_bucket,
            input_data=all_articles_metadata,
        )
        print(f"All articles' metadata saved to {output_file}")
    except Exception as e:
        print(f"Error processing article: {e}")

    finally:
        # Close the browser
        driver.quit()
