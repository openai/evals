import requests


def scrape_prs(start_pr, end_pr):
    # Set the base URL of the GitHub API
    base_url = "https://api.github.com"

    # Set the repository owner and name
    owner = "openai"
    repo = "evals"

    # Set the API endpoint for listing pull requests
    endpoint = f"/repos/{owner}/{repo}/pulls"

    # Set the query parameters for retrieving merged pull requests
    params = {
        "state": "closed",
        "sort": "updated",
        "direction": "desc",
        "per_page": 100
    }

    page = 1
    while True:
        # Update the query parameters with the current page number
        params["page"] = page

        # Send a GET request to the API endpoint and parse the response
        response = requests.get(base_url + endpoint, params=params)
        pull_requests = response.json()

        # If there are no pull requests in the response, we've reached the end of the pages
        if not pull_requests:
            break

        # Loop over each pull request in the response
        for pr in pull_requests:
            pr_number = pr["number"]

            # Check if the pull request is within the specified range
            if start_pr <= pr_number <= end_pr:
                # Check if the pull request was merged
                if pr["merged_at"]:
                    # Extract the pull request title and body
                    title = pr["title"]
                    body = pr["body"]

                    # Create a filename for the current pull request
                    file_name = f"pr-{pr_number}.md"

                    # Open a file for writing
                    with open(file_name, "w") as f:
                        # Write the pull request title and body to the file
                        f.write(f"Title: {title}\n")
                        f.write(f"Body: {body}\n")

        # Increment the page number to move to the next page of pull requests
        page += 1


# Set the range of PRs to scrape
start_pr = 719
end_pr = 751

# Call the function to scrape PRs
scrape_prs(start_pr, end_pr)
