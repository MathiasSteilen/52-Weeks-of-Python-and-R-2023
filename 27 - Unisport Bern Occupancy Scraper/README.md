# Unisport Bern Occupancy Scraper

This repository contains a Python script that scrapes the occupancy level of Unisport Bern fitness rooms. The script runs at short intervals using GitHub Actions and updates a CSV file with the latest occupancy data.

## Usage

To use this script, follow these steps:

1. Clone this repository to your local machine:

    ```bash
    git clone https://github.com/your-username/unisport-bern-occupancy.git
    ```

2. Install the required dependencies using pip:

    ```bash
    pip install pandas requests beautifulsoup4
    ```

3. Update the script with your desired intervals or settings, if necessary.

4. Run the script:

    ```bash
    python occupancy_scraper.py
    ```

## Script Description

The Python script (`occupancy_scraper.py`) in this repository retrieves the occupancy level of Unisport Bern fitness rooms from the official website. It utilizes the following libraries:

- `pandas` for data manipulation
- `requests` for making HTTP requests
- `beautifulsoup4` for HTML parsing

The script performs the following actions:

1. Sends an HTTP request to the Unisport Bern website to retrieve the occupancy data.
2. Parses the HTML content to extract the occupancy numbers.
3. Appends the new occupancy data along with the current timestamp to a CSV file (`data.csv`).
4. Handles errors as exceptions and prints error messages if any occur.

## GitHub Actions

This repository is configured with GitHub Actions to run the script at short intervals automatically. The workflow file (`occupancy_scraper.yml`) defines the schedule for running the script and updating the data.

## File Structure

- `occupancy_scraper.py`: Python script for scraping Unisport Bern occupancy data.
- `data.csv`: CSV file containing historical occupancy data.
- `.github/workflows/occupancy_scraper.yml`: GitHub Actions workflow file for scheduling script execution.

## Notes

- The script's timeout for HTTP requests is set to 180 seconds (3 minutes) to handle potential delays in response.
- Ensure proper internet connectivity and access to the Unisport Bern website for successful scraping.
