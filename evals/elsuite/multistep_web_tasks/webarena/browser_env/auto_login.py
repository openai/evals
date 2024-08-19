"""Script to automatically login each website"""

import glob
from itertools import combinations
from pathlib import Path

from beartype import beartype
from playwright.sync_api import sync_playwright

from evals.elsuite.multistep_web_tasks.webarena.browser_env.env_config import (
    ACCOUNTS,
    GITLAB,
    REDDIT,
    SHOPPING,
    SHOPPING_ADMIN,
)

HEADLESS = True
SLOW_MO = 0


@beartype
def is_expired(storage_state: Path, url: str, keyword: str, url_exact: bool = True) -> bool:
    """Test whether the cookie is expired"""
    if not storage_state.exists():
        return True

    context_manager = sync_playwright()
    playwright = context_manager.__enter__()
    browser = playwright.chromium.launch(headless=HEADLESS, slow_mo=SLOW_MO)
    context = browser.new_context(storage_state=storage_state)
    page = context.new_page()
    page.goto(url)
    d_url = page.url
    content = page.content()
    context_manager.__exit__()
    if keyword:
        return keyword not in content
    else:
        if url_exact:
            return d_url != url
        else:
            return url not in d_url


@beartype
def renew_comb(comb: list[str]) -> None:
    context_manager = sync_playwright()
    playwright = context_manager.__enter__()
    browser = playwright.chromium.launch(headless=HEADLESS)
    context = browser.new_context()
    page = context.new_page()

    if "shopping" in comb:
        username = ACCOUNTS["shopping"]["username"]
        password = ACCOUNTS["shopping"]["password"]
        page.goto(f"{SHOPPING}/customer/account/login/")
        page.get_by_label("Email", exact=True).fill(username)
        page.get_by_label("Password", exact=True).fill(password)
        page.get_by_role("button", name="Sign In").click()

    if "reddit" in comb:
        username = ACCOUNTS["reddit"]["username"]
        password = ACCOUNTS["reddit"]["password"]
        page.goto(f"{REDDIT}/login")
        page.get_by_label("Username").fill(username)
        page.get_by_label("Password").fill(password)
        page.get_by_role("button", name="Log in").click()

    if "shopping-admin" in comb:
        username = ACCOUNTS["shopping-admin"]["username"]
        password = ACCOUNTS["shopping-admin"]["password"]
        page.goto(f"{SHOPPING_ADMIN}")
        page.get_by_placeholder("user name").fill(username)
        page.get_by_placeholder("password").fill(password)
        page.get_by_role("button", name="Sign in").click()

    if "gitlab" in comb:
        username = ACCOUNTS["gitlab"]["username"]
        password = ACCOUNTS["gitlab"]["password"]
        page.goto(f"{GITLAB}/users/sign_in")
        page.get_by_test_id("username-field").click()
        page.get_by_test_id("username-field").fill(username)
        page.get_by_test_id("username-field").press("Tab")
        page.get_by_test_id("password-field").fill(password)
        page.get_by_test_id("sign-in-button").click()

    context.storage_state(path=f"./.auth/{'.'.join(comb)}_state.json")

    context_manager.__exit__()


@beartype
def main() -> None:
    sites = ["gitlab", "shopping", "shopping-admin", "reddit"]
    urls = [
        f"{GITLAB}/-/profile",
        f"{SHOPPING}/wishlist/",
        f"{SHOPPING_ADMIN}/dashboard",
        f"{REDDIT}/user/{ACCOUNTS['reddit']['username']}/account",
    ]
    exact_match = [True, True, True, True]
    keywords = ["", "", "Dashboard", "Delete"]

    pairs = list(combinations(sites, 2))
    for pair in pairs:
        # TODO[shuyanzh] auth don't work on these two sites
        if "reddit" in pair and ("shopping" in pair or "shopping-admin" in pair):
            continue
        renew_comb(list(sorted(pair)))

    for site in sites:
        renew_comb([site])

    for c_file in glob.glob("./.auth/*.json"):
        comb = c_file.split("/")[-1].rsplit("_", 1)[0].split(".")
        for cur_site in comb:
            url = urls[sites.index(cur_site)]
            keyword = keywords[sites.index(cur_site)]
            match = exact_match[sites.index(cur_site)]
            assert not is_expired(Path(c_file), url, keyword, match)


if __name__ == "__main__":
    main()
