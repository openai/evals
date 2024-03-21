"""Implements helper functions to assist evaluation cases where other evaluators are not suitable."""
import json
from typing import Any
from urllib.parse import urlparse

import requests
from beartype import beartype
from playwright.sync_api import Page

from evals.completion_fns.openai import OpenAIChatCompletionFn
from evals.elsuite.multistep_web_tasks.webarena.browser_env.env_config import ACCOUNTS, SHOPPING


@beartype
def shopping_get_auth_token() -> str:
    response = requests.post(
        url=f"{SHOPPING}/rest/default/V1/integration/admin/token",
        headers={"content-type": "application/json"},
        data=json.dumps(
            {
                "username": ACCOUNTS["shopping_site_admin"]["username"],
                "password": ACCOUNTS["shopping_site_admin"]["password"],
            }
        ),
    )
    token: str = response.json()
    return token


@beartype
def shopping_get_latest_order_url() -> str:
    """Get the latest order url from the shopping website."""

    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }

    params = {
        "searchCriteria[sortOrders][0][field]": "created_at",
        "searchCriteria[sortOrders][0][direction]": "DESC",
        "searchCriteria[pageSize]": "1",
    }

    response = requests.get(f"{SHOPPING}/rest/V1/orders", params=params, headers=header)
    assert response.status_code == 200
    response_obj = response.json()["items"][0]
    order_id = int(response_obj["increment_id"])
    order_url = f"{SHOPPING}/sales/order/view/order_id/{order_id}/"
    return order_url


@beartype
def shopping_get_sku_latest_review_author(sku: str) -> str:
    """Get the latest review for shopping admin."""
    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }
    response = requests.get(f"{SHOPPING}/rest/V1/products/{sku}/reviews", headers=header)
    assert response.status_code == 200
    response_obj = response.json()
    if len(response_obj) == 0:
        return ""
    author: str = response_obj[-1]["nickname"]
    return author


@beartype
def shopping_get_sku_latest_review_rating(sku: str) -> str:
    """Get the latest review for shopping admin."""
    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }
    response = requests.get(f"{SHOPPING}/rest/V1/products/{sku}/reviews", headers=header)
    assert response.status_code == 200
    response_obj = response.json()
    if len(response_obj) == 0:
        return ""
    assert response_obj[0]["ratings"][0]["rating_name"] == "Rating"
    rating: str = str(response_obj[-1]["ratings"][0]["percent"])
    return rating


@beartype
def reddit_get_post_url(url: str) -> str:
    """Get the post url"""
    # Url is http://domain/f/subreddit/post_id/...
    # get domain, subreddit, post_id
    domain = urlparse(url).netloc
    tok_url = urlparse(url).path.split("/")
    # not a valid post/comment url, return the url as is
    if len(tok_url) < 4:
        return url
    if tok_url[1] != "f":
        return url
    subreddit = urlparse(url).path.split("/")[2]
    post_id = urlparse(url).path.split("/")[3]
    scheme = urlparse(url).scheme
    post_url = f"{scheme}://{domain}/f/{subreddit}/{post_id}/"
    return post_url


@beartype
def gitlab_get_project_memeber_role(page: Page, account_name: str) -> str:
    # get the account index
    try:
        account_idx = page.evaluate(
            f"""(() => {{
                const elements = document.querySelectorAll("td[data-label='Account'] span.gl-avatar-labeled-sublabel");
                let index = -1;  // Default value if not found

                for(let i = 0; i < elements.length; i++) {{
                    if(elements[i].outerText === '@{account_name}') {{
                        index = i;
                        break;
                    }}
                }}

                return index;
            }})()"""
        )

        # get the role
        role: str = page.evaluate(
            f"""(() => {{
                return document.querySelectorAll("td.col-max-role span")[{account_idx}].outerText;
            }})()"""
        )
    except Exception:
        role = ""

    return role


@beartype
def llm_fuzzy_match(pred: str, reference: str, question: str) -> float:
    """Check whether the prediction matches the reference with GPT-3.5"""
    messages: list[dict[str, Any]] = []
    messages.append({"role": "system", "content": "You are a helpful assistant"})

    messages.append(
        {
            "role": "user",
            "content": f'Given the statement "{pred}", would it be correct to infer "{reference}"? Yes or No',
        }
    )

    # NOTE: untested
    completion_fn = OpenAIChatCompletionFn(
        model="gpt-3.5-turbo", extra_options={"temperature": 0, "top_p": 1, "max_tokens": 16}
    )
    response = completion_fn(messages)
    response = response.get_completions()[0]
    if "Yes" in response:
        return 1.0
    else:
        return 0.0
