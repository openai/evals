"""
This file defines utilities for connecting to and interacting with Snowflake.
Familiarity with this file should not be needed even if working with Snowflake.
"""
import logging
import os
import time
from contextlib import contextmanager
from typing import Optional

logger = logging.getLogger(__name__)


def _first_not_none(*args):
    for arg in args:
        if arg is not None:
            return arg
    return None


class SnowflakeError(Exception):
    pass


class SnowflakeConnection:
    def __init__(
        self,
        autocommit=True,
        account: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        warehouse: Optional[str] = None,
        paramstyle="pyformat",
    ):
        self.account = _first_not_none(account, os.environ.get("SNOWFLAKE_ACCOUNT"))
        self.database = _first_not_none(database, os.environ.get("SNOWFLAKE_DATABASE"))

        self.user = _first_not_none(
            user,
            os.environ.get("SNOWFLAKE_USERNAME"),
        )
        self.password = _first_not_none(
            password,
            os.environ.get("SNOWFLAKE_PASSWORD"),
        )

        if self.user is None and self.password is None:
            self.user = os.environ["USER"]
            self.use_browser_auth = True
        else:
            self.use_browser_auth = False

        self.autocommit = autocommit
        self.warehouse = warehouse
        self.paramstyle = paramstyle

        self.ctx = None

    def _ensure_connected(self):
        if self.ctx is not None:
            return
        import snowflake.connector

        logging.getLogger("snowflake").setLevel(logging.WARNING)
        # Snowflake prints a scary "Don't know how to construct
        # ResultBatches from response..." error when doing a select
        # with no results.
        logging.getLogger("snowflake.connector.result_batch").setLevel(logging.CRITICAL)

        try:
            self.ctx = snowflake.connector.connect(
                user=self.user,
                account=self.account,
                database=self.database,
                schema="public",
                password=self.password,
                authenticator="externalbrowser" if self.use_browser_auth else "snowflake",
                autocommit=self.autocommit,
                client_prefetch_thread=16,
                client_session_keep_alive=True,
                warehouse=self.warehouse,
                paramstyle=self.paramstyle,
            )
        except snowflake.connector.errors.DatabaseError as e:
            raise SnowflakeError(
                f"""Failed to connect to database: {e}
(HINT: if running on a server, you may want to set SNOWFLAKE_PASSWORD=... to use password authentication)"""
            )

    def cursor(self, *args, **kwargs):
        self._ensure_connected()
        cs = self.ctx.cursor(*args, **kwargs)
        return cs

    @contextmanager
    def __call__(self, *args, **kwargs):
        cs = self.cursor(*args, **kwargs)
        try:
            yield cs
        finally:
            cs.close()

    def query(self, *args, many=False, pandas_out=False, list_out=False, **kwargs):
        with self() as cs:
            if many:
                cs.executemany(*args, **kwargs)
            else:
                cs.execute(*args, **kwargs)
            if pandas_out:
                return cs.fetch_pandas_all()
            elif list_out:
                return cs.fetchall()

    def robust_query(self, max_trials: Optional[int] = None, *args, **kwargs):
        from snowflake.connector.errors import OperationalError, ProgrammingError

        ntrials = 0
        while True:
            try:
                return self.query(*args, **kwargs)
            except (OperationalError, ProgrammingError) as e:
                if max_trials is not None and ntrials >= max_trials:
                    raise
                logger.info(f"Snowflake insert failed, will retry in 5s {e}")
                ntrials += 1
                time.sleep(5)
