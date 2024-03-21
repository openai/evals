import logging
import time
from pathlib import Path
from typing import Optional

import docker
import requests

from evals.elsuite.multistep_web_tasks.constants import (
    AVAILABLE_SERVICES,
    CACHE_DIR,
    DEFAULT_PORT_MAPPINGS,
    DOCKER_NAMES,
    DOWNLOAD_URLS,
    LOCAL_NETWORK,
    ServiceIdentifier,
)
from evals.elsuite.multistep_web_tasks.utils import load_experiment_config_from_dict
from evals.elsuite.multistep_web_tasks.webarena.core.utils import (
    BashBrowserExperimentConfig,
    BrowserExperimentConfig,
)

logger = logging.getLogger(__name__)


class Session:
    def __init__(self, docker_client: docker.DockerClient) -> None:  # type: ignore (docker sdk)
        self.containers_to_setup: set[str] = set()
        self.docker_client = docker_client
        self.containers: Optional[dict[ServiceIdentifier, docker.models.containers.Container]] = None  # type: ignore

    def add_samples(self, samples: list[dict]) -> None:
        self.containers_to_setup |= self._get_containers_to_setup(samples)

    def __enter__(self):
        assert len(self.containers_to_setup) > 0, "No samples added to session"

        try:
            self.network = self.setup_network()
            self.containers = self.setup_docker_environments()

            logger.info("Waiting for containers to start up, could take several minutes...")
            # we use gitlab as the container to wait for, since it should take the longest
            for container_name in self.containers:
                # raises a value error if any are not ready within timeout
                # TODO: maybe clean up/parallelise this
                self._is_container_ready(container_name)
            logger.info("All containers running!")
        except Exception as e:
            logger.error("Error while setting up containers, tearing down...")
            self.__exit__()
            raise e

        return self

    def __exit__(self, *args):
        if any(isinstance(arg, Exception) for arg in args):
            logger.info("Tearing down session because we hit an error...")
        else:
            logger.info("Tearing down session...")
        for arg in args:
            if isinstance(arg, Exception):
                logger.error(f"Error in session led to shutdown:\n{arg}")
        self.teardown_docker_environments()
        self.teardown_network()
        for arg in args:
            if isinstance(arg, Exception):
                raise arg

    def get_container(self, container_name: str) -> docker.models.containers.Container:  # type: ignore
        if self.containers is None:
            raise ValueError("Session must be entered before getting containers")
        if container_name not in self.containers:
            raise ValueError(f"Container {container_name} not found in session!")
        return self.containers[container_name]

    def register_container(self, container_name: ServiceIdentifier, container: docker.models.containers.Container) -> None:  # type: ignore
        assert self.containers is not None, "Session must be entered before registering containers"
        if container_name in self.containers:
            raise ValueError(f"Container {container_name} already registered")
        self.containers[container_name] = container

    def setup_docker_environments(self) -> dict[ServiceIdentifier, docker.models.containers.Container]:  # type: ignore
        containers = dict()
        logger.info(f"Setting up containers: {self.containers_to_setup = }")
        for container_name in self.containers_to_setup:
            container = self.setup_container(container_name)
            containers[container_name] = container
        logger.info(f"Finished setting up containers: {containers = }")
        return containers

    def setup_network(self) -> docker.models.networks.Network:  # type: ignore
        """Set up a network with the network name from constants.py.
        Currently I just set up the network here and rely on the bash container to get it
        TODO: Work out if the network should be handled some other way"""
        try:
            network = self.docker_client.networks.create(
                LOCAL_NETWORK,
                driver="bridge",
                options={
                    "com.docker.network.bridge.enable_icc": "true",
                    "com.docker.network.bridge.enable_ip_masquerade": "false",
                },
            )
        except docker.errors.APIError:  # type: ignore
            raise ValueError(
                f"Couldn't create {LOCAL_NETWORK}! (maybe a previous version still exists? Try `docker network ls`)"
            )

        return network

    def setup_container(self, container_name: str) -> docker.models.containers.Container:  # type: ignore
        if container_name == "bash":
            container = self._setup_bash_environment()
            return container
        elif container_name == "homepage":
            container = self._setup_homepage_environment()
            return container
        elif container_name == "simple-web":
            container = self._setup_simpleweb_environment()
            return container
        elif container_name == "shopping":
            container = self._setup_shopping_environment()
            return container
        elif container_name == "shopping-admin":
            container = self._setup_shopping_admin_environment()
            return container
        elif container_name == "reddit":
            container = self._setup_reddit_environment()
            return container
        elif container_name == "gitlab":
            container = self._setup_gitlab_environment()
            return container
        elif container_name == "wikipedia":
            container = self._setup_wikipedia_environment()
            return container
        elif container_name == "flask-playwright":
            container = self._setup_flask_playwright_environment()
            return container
        else:
            raise ValueError(
                f"Unknown container {container_name}, known containers:\n{AVAILABLE_SERVICES}"
            )

    def _setup_bash_environment(self) -> docker.models.containers.Container:  # type: ignore
        container = self._run_container_setup(
            container_name=DOCKER_NAMES["bash"]["container"],
            image_name=DOCKER_NAMES["bash"]["image"],
            # docker sdk expects str path to dir containing Dockerfile
            docker_file=str(Path(__file__).parent / "docker/dc-evals-bash"),
            # need to set tty otherwise the container just stops
            tty=True,
        )
        return container

    def _setup_homepage_environment(self) -> docker.models.containers.Container:  # type: ignore
        container = self._run_container_setup(
            container_name=DOCKER_NAMES["homepage"]["container"],
            image_name=DOCKER_NAMES["homepage"]["image"],
            # docker sdk expects str path to dir containing Dockerfile
            docker_file=str(Path(__file__).parent / "docker/homepage"),
        )
        return container

    def _setup_flask_playwright_environment(self) -> docker.models.containers.Container:  # type: ignore
        container = self._run_container_setup(
            container_name=DOCKER_NAMES["flask-playwright"]["container"],
            image_name=DOCKER_NAMES["flask-playwright"]["image"],
            # docker sdk expects str path to dir containing Dockerfile
            docker_file=str(Path(__file__).parent / "docker/flask-playwright"),
        )
        return container

    def _setup_simpleweb_environment(self) -> docker.models.containers.Container:  # type: ignore
        container = self._run_container_setup(
            container_name=DOCKER_NAMES["simple-web"]["container"],
            image_name=DOCKER_NAMES["simple-web"]["image"],
            check_repository=True,
        )
        return container

    def _setup_shopping_environment(self) -> docker.models.containers.Container:  # type: ignore
        container_name = DOCKER_NAMES["shopping"]["container"]
        container = self._run_container_setup(
            container_name=container_name,
            image_name=DOCKER_NAMES["shopping"]["image"],
            check_repository=False,
            cache_file="shopping_final_0712.tar",
            url=DOWNLOAD_URLS["shopping"],
        )

        ports = DEFAULT_PORT_MAPPINGS["shopping-admin"]
        internal_port = ports["internal"]

        # setup commands from webarena
        logger.warning("Starting exec_runs in shopping container; may take 10s")
        # TODO: work out if there's a more flexible way to wait for redis to be running rather than sleeping 5s
        time.sleep(5)
        exec_out = container.exec_run(
            f"/var/www/magento2/bin/magento setup:store-config:set --base-url='http://{container_name}:{internal_port}'"
        )
        if exec_out.exit_code != 0:
            logger.warning(f"Error setting base url in shopping: {exec_out}")
            raise ValueError("Error setting base url in shopping")
        container.exec_run(
            f'mysql -u magentouser -pMyPassword magentodb -e  \'UPDATE core_config_data SET value="http://{container_name}:{internal_port}/" WHERE path = "web/secure/base_url";\''
        )
        container.exec_run("/var/www/magento2/bin/magento cache:flush")

        return container

    def _setup_shopping_admin_environment(self) -> docker.models.containers.Container:  # type: ignore
        ports = DEFAULT_PORT_MAPPINGS["shopping-admin"]
        internal_port = ports["internal"]
        container_name = DOCKER_NAMES["shopping-admin"]["container"]
        container = self._run_container_setup(
            container_name=container_name,
            image_name=DOCKER_NAMES["shopping-admin"]["image"],
            check_repository=False,
            cache_file="shopping_admin_final_0719.tar",
            url=DOWNLOAD_URLS["shopping-admin"],
        )

        ports = DEFAULT_PORT_MAPPINGS["shopping-admin"]
        internal_port = ports["internal"]
        # setup commands from webarena
        logger.warning("Starting exec_runs in shopping-admin container; may take 10s")
        # TODO: work out if there's a more flexible way to wait for redis to be running
        time.sleep(5)
        exec_out = container.exec_run(
            f"/var/www/magento2/bin/magento setup:store-config:set --base-url='http://{container_name}:{internal_port}'"
        )
        if exec_out.exit_code != 0:
            logger.warning(f"Error setting base url in shopping-admin: {exec_out}")
            raise ValueError("Error setting base url in shopping-admin")
        container.exec_run(
            f'mysql -u magentouser -pMyPassword magentodb -e  \'UPDATE core_config_data SET value="http://{container_name}:{internal_port}/" WHERE path = "web/secure/base_url";\''
        )
        container.exec_run("/var/www/magento2/bin/magento cache:flush")

        return container

    def _setup_reddit_environment(self) -> docker.models.containers.Container:  # type: ignore
        container = self._run_container_setup(
            container_name=DOCKER_NAMES["reddit"]["container"],
            image_name=DOCKER_NAMES["reddit"]["image"],
            check_repository=False,
            cache_file="postmill-populated-exposed-withimg.tar",
            url=DOWNLOAD_URLS["reddit"],
        )
        return container

    def _setup_gitlab_environment(self) -> docker.models.containers.Container:  # type: ignore
        entrypoint_file = str((Path(__file__).parent / "docker/gitlab/entrypoint.sh").resolve())

        container = self._run_container_setup(
            container_name=DOCKER_NAMES["gitlab"]["container"],
            image_name=DOCKER_NAMES["gitlab"]["image"],
            check_repository=False,
            cache_file="gitlab-populated-final-port8023",
            url=DOWNLOAD_URLS["gitlab"],
            volumes={entrypoint_file: {"bind": "/entrypoint.sh", "mode": "ro"}},
            command="/entrypoint.sh",
        )
        return container

    def _setup_wikipedia_environment(self) -> docker.models.containers.Container:  # type: ignore
        # make sure we have access to the wikipedia data archive
        wikipedia_path = Path(CACHE_DIR) / "wikipedia_en_all_maxi_2022-05.zim"
        if not wikipedia_path.is_file():
            logger.warning(f"wikipedia zim not found at {wikipedia_path}, downloading...")
            try:
                download_to_file(
                    DOWNLOAD_URLS["wikipedia_zim"],
                    Path(CACHE_DIR) / "wikipedia_en_all_maxi_2022-05.zim",
                )
            except Exception as e:
                logger.warning(
                    f"Error downloading wikipedia zim from {DOWNLOAD_URLS['wikipedia_zim']}: {e}"
                )
                raise ValueError(
                    "Couldn't download wikipedia zim, please see the instructions in the multistep-web-tasks README.md"
                )

        container = self._run_container_setup(
            container_name=DOCKER_NAMES["wikipedia"]["container"],
            image_name=DOCKER_NAMES["wikipedia"]["image"],
            check_repository=True,
            command="/data/wikipedia_en_all_maxi_2022-05.zim",
            volumes=[f"{CACHE_DIR}:/data"],
        )
        return container

    def _run_container_setup(
        self,
        container_name: str,
        image_name: str,
        cache_file: Optional[str] = None,
        docker_file: Optional[str] = None,
        check_repository: bool = False,
        url: Optional[str] = None,
        network: Optional[str] = LOCAL_NETWORK,
        **run_kwargs,
    ) -> docker.models.containers.Container:  # type: ignore
        # convenience function to avoid writing this out n times
        def container():
            try:
                if network is not None:
                    container = self.docker_client.containers.run(
                        name=container_name,
                        image=image_name,
                        detach=True,
                        network=network,
                        **run_kwargs,
                    )
                    return container
                else:
                    return self.docker_client.containers.run(
                        name=container_name,
                        image=image_name,
                        detach=True,
                        network_disabled=True,
                        **run_kwargs,
                    )
            except docker.errors.APIError as e:  # type: ignore
                logger.error(f"Error running container {container_name}: {e}")
                logger.error("Try running the `CLEANUP.sh` script in `reproducibility`")
                raise e

        try:
            _ = self.docker_client.images.get(image_name)
            return container()
        except docker.errors.ImageNotFound:  # type: ignore
            logger.info(f"{image_name} not found locally, attempting to build...")
            try:
                self._get_image(
                    image_name=image_name,
                    cache_file=cache_file,
                    docker_file=docker_file,
                    check_repository=check_repository,
                    url=url,
                )
                return container()
            except ValueError as e:
                logger.error(f"Error getting image {image_name}: {e}")
                raise e

    def _get_image(
        self,
        image_name: str,
        cache_file: Optional[str] = None,
        docker_file: Optional[str] = None,
        check_repository: bool = False,
        url: Optional[str] = None,
    ) -> bool:
        # optionally, check the repository
        if check_repository:
            try:
                logger.info(f"checking repository for {image_name}...")
                _ = self.docker_client.images.pull(image_name)
                return True
            except docker.errors.APIError:  # type: ignore
                logger.warning(f"{image_name} not found in repository")

        # next, optionally try to load from a cached tar
        if cache_file is not None:
            # first, try to get from local images
            cache_path = (Path(CACHE_DIR) / cache_file).expanduser()
            try:
                logger.info(f"trying to load {image_name} from cache...")
                with cache_path.open("rb") as f:
                    _ = self.docker_client.images.load(f)
                return True
            except FileNotFoundError:
                logger.warning(f"tar not found at cache path {cache_path}")
        # next, optionally build from a docker file
        if docker_file is not None:
            try:
                logger.info(f"trying to build {image_name} from Dockerfile...")
                self.build_image_from_dockerfile(docker_file, image_name)
                return True
            except Exception as e:
                logger.warning(f"couldn't build from Dockerfile: {docker_file}: {e}")

        # finally, optionally download tar from the web
        if url is not None and cache_file is not None:
            # to appease type-checking we define this again
            cache_path = (Path(CACHE_DIR) / cache_file).expanduser()
            try:
                logger.info(f"attempting to download tar from {url}...")
                download_to_file(url, cache_path)
                logger.info(f"Downloaded {image_name} tar to {cache_path}")
                with cache_path.open("rb") as f:
                    _ = self.docker_client.images.load(f)
                return True
            except Exception as e:
                logger.warning(f"Error loading from downloaded {image_name} tar from {url}: {e}")

        raise ValueError(
            f"Could not find the docker image '{image_name}' through any route (which usually means it failed to download):"
            " please see the instructions in the multistep-web-tasks README.md"
        )

    def build_image_from_dockerfile(self, dockerfile_dir: str, image_name: str) -> docker.models.images.Image:  # type: ignore
        """Build a Docker image from a Dockerfile."""
        try:
            image, build_logs = self.docker_client.images.build(  # type: ignore (returns a 2-tuple)
                path=dockerfile_dir,
                tag=image_name,
                rm=True,
            )

            for line in build_logs:
                logger.debug(line)

            return image
        except docker.errors.BuildError as e:  # type: ignore
            logger.error(f"Error building Docker image '{image_name}': {e}")

            for image in self.docker_client.images.list():
                logger.info(image.tags)  # type: ignore

            for line in e.build_log:
                logger.debug(line)

            raise e

    def _get_containers_to_setup(self, samples) -> set[str]:
        containers_to_setup = set()
        # TODO: work out if this can/should be cleaned up
        for sample in samples:
            experiment_config = load_experiment_config_from_dict(sample)
            if isinstance(experiment_config, BrowserExperimentConfig) or isinstance(
                experiment_config, BashBrowserExperimentConfig
            ):
                containers_to_setup.update(experiment_config.sites)
        return containers_to_setup

    def teardown_network(self) -> None:
        self.network.remove()  # type: ignore (network does have .remove())

    def teardown_docker_environments(self) -> None:
        """Currently stops and removes all setup containers.
        TODO: maybe allow some to stay, esp. if they're stateless?"""
        if self.containers is None:
            logger.warning(
                "No containers to remove; session must be entered before removing containers"
            )
            return

        for container_name in list(self.containers.keys()):
            logger.info(f"Removing container {container_name}: {self.containers[container_name]}")
            self.teardown_container(container_name)

    def teardown_container(self, container_name: ServiceIdentifier) -> None:  # type: ignore
        if self.containers is None:
            logger.warning(
                "No containers to remove; session must be entered before removing container"
            )
            return

        container = self.containers[container_name]
        self.network.disconnect(container)
        container.stop()
        container.remove()
        del self.containers[container_name]

    def _is_container_ready(
        self, container_name: ServiceIdentifier, path="/", timeout=300, interval=10
    ):
        """
        Polls the container's service until it's ready to serve HTTP requests or the timeout is reached.

        Parameters:
        - container_name: Name of the container in self.containers.
        - path: Path to check on the server. Default is root.
        - timeout: Total time in seconds to wait for the container to be ready.
        - interval: Time in seconds between each poll.

        Returns:
        - True if the container's service is ready, raises ValueError otherwise
        """

        assert self.containers is not None, "Session must be entered before checking containers"
        port = DEFAULT_PORT_MAPPINGS[container_name]["internal"]

        url = f"http://localhost:{port}{path}"
        end_time = time.time() + timeout

        while time.time() < end_time:
            try:
                logger.debug(f"Checking {url} for {container_name}...")
                result = self.containers[container_name].exec_run(
                    f"wget --spider --timeout={interval} --tries=1 {url}"
                )

                # If the exit code is 0, the HTTP request was successful
                if result.exit_code == 0:
                    return True

            except Exception as e:
                # If an exception occurs (e.g., the service is not yet available), just pass and try again
                logger.debug(f"While checking {url} for {container_name}, got exception: {e}")

            time.sleep(interval)

        # If the loop completes without returning, the timeout was reached
        raise ValueError(f"Timeout reached while waiting for {url} to be ready")


def download_to_file(url: str, path: Path) -> None:
    r = requests.get(url, allow_redirects=True, stream=True)
    if r.status_code == 200:
        with path.open("wb") as f:
            f.write(r.content)
    else:
        logger.warning(f"Error downloading {url}: {r.status_code}")
