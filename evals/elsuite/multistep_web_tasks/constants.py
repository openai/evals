from typing import Literal

CACHE_DIR = "~/.cache/evals/multistep-web-tasks/"
LOCAL_SERVER = "127.0.0.1"
LOCAL_NETWORK = "multistep-web-tasks_network"
ServiceIdentifier = Literal[
    "simple-web",
    "homepage",
    "shopping",
    "shopping-admin",
    "gitlab",
    "reddit",
    "wikipedia",
    "flask-playwright",
    "bash",
]

AVAILABLE_SERVICES: list[ServiceIdentifier] = [
    "simple-web",
    "homepage",
    "shopping",
    "shopping-admin",
    "gitlab",
    "reddit",
    "wikipedia",
    "flask-playwright",
    "bash",
]

DEFAULT_PORT_MAPPINGS: dict[ServiceIdentifier, dict[str, int]] = {
    "simple-web": {"internal": 80, "external": 4444},
    "homepage": {"internal": 4399, "external": 4399},
    "shopping": {"internal": 80, "external": 7770},
    "shopping-admin": {"internal": 80, "external": 7780},
    "wikipedia": {"internal": 80, "external": 8888},
    "reddit": {"internal": 80, "external": 9999},
    "gitlab": {"internal": 8023, "external": 8023},
    "flask-playwright": {"internal": 8507, "external": 8507},
    "bash": {
        "internal": -1,
        "external": -1,
    },  # we don't use ports on bash, this is just for compatibility
}
DOCKER_NAMES: dict[ServiceIdentifier, dict[str, str]] = {
    "simple-web": {"image": "yeasy/simple-web", "container": "simple-web"},
    "homepage": {"image": "dc-evals-homepage", "container": "homepage"},
    "bash": {"image": "dc-evals-bash", "container": "bash"},
    "shopping": {"image": "shopping_final_0712", "container": "shopping"},
    "shopping-admin": {"image": "shopping_admin_final_0719", "container": "shopping-admin"},
    "gitlab": {"image": "gitlab-populated-final-port8023", "container": "gitlab"},
    "reddit": {"image": "postmill-populated-exposed-withimg", "container": "reddit"},
    "wikipedia": {"image": "ghcr.io/kiwix/kiwix-serve:3.3.0", "container": "wikipedia"},
    "flask-playwright": {"image": "dc-evals-flask-playwright", "container": "flask-playwright"},
}
# These are the URLs that the model will use to access the services
SERVICE_TO_URL: dict[ServiceIdentifier, str] = {
    "simple-web": "http://simple-web.com",
    "homepage": "http://homepage.com",
    "shopping": "http://onestopmarket.com",
    "shopping-admin": "http://shopping-admin.com",
    "gitlab": "http://gitlab.com",
    "reddit": "http://reddit.com",
    "wikipedia": "http://wikipedia.org",
}
URL_TO_SERVICE: dict[str, ServiceIdentifier] = {v: k for k, v in SERVICE_TO_URL.items()}

DOWNLOAD_URLS = {
    "wikipedia_zim": "http://metis.lti.cs.cmu.edu/webarena-images/wikipedia_en_all_maxi_2022-05.zim",
    "shopping": "http://metis.lti.cs.cmu.edu/webarena-images/shopping_final_0712.tar",
    "shopping-admin": "http://metis.lti.cs.cmu.edu/webarena-images/shopping_admin_final_0719.tar",
    "reddit": "http://metis.lti.cs.cmu.edu/webarena-images/postmill-populated-exposed-withimg.tar",
    "gitlab": "http://metis.lti.cs.cmu.edu/webarena-images/gitlab-populated-final-port8023.tar",
}
DOCKER_CLIENT_TIMEOUT = 600
FLASK_API_KEY = "key-FLASKPLAYWRIGHTKEY"
