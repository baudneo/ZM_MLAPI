"""Helper class that adds file locking to a class."""
import time
from logging import getLogger
from typing import Optional
from os import getuid

from portalocker import BoundedSemaphore, AlreadyLocked

from zm_mlapi.app import locks_enabled, get_global_config

logger = getLogger("zm_mlapi")
LP: str = 'Lock:'


class FileLock:
    lock: Optional[BoundedSemaphore] = None
    is_locked: bool = False

    def create_lock(self):
        if locks_enabled():
            if self.lock:
                if isinstance(self.lock, BoundedSemaphore):
                    logger.warning(
                        f"{LP} '{self.name}' LOCK ALREADY EXISTS!!! [name: {self.lock.name}]"
                    )
                    return
                else:
                    logger.warning(
                        f"{LP} '{self.name}' LOCK ALREADY EXISTS BUT IS NOT A BoundedSemaphore!!! creating new lock"
                    )
            if locks := get_global_config().settings.lock_settings:
                self.lock_dir = locks.lock_dir
                lock = locks.get(self.processor.casefold())
                self.lock_name = f"{lock.name}-{getuid()}"
                self.lock_maximum = lock.max
                self.lock_timeout = lock.timeout
                self.lock = BoundedSemaphore(
                    maximum=self.lock_maximum,
                    name=self.lock_name,
                    timeout=self.lock_timeout,
                    directory=self.lock_dir,
                )
                logger.debug(
                    f"{LP} '{self.name}' CREATED LOCK!!! [directory: {self.lock.directory}] - "
                    f"[name: {self.lock.name}] - [max: {self.lock.maximum}] - "
                    f"[timeout: {self.lock.timeout}]"
                )

    def acquire_lock(self):
        if locks_enabled():
            if self.is_locked:
                logger.debug(f"{LP} '{self.name}' lock for '{self.lock.name}' already acquired")
                return
            try:
                if self.lock:

                    logger.debug(
                        f"{LP} '{self.name}' attempting to acquire lock for '{self.lock.name}'..."
                    )
                    timer = time.perf_counter()
                    self.lock.acquire()
                    logger.debug(
                        f"perf:{LP} '{self.name}' lock acquired for '{self.lock.name}' in "
                        f"{time.perf_counter() - timer:.5f} seconds"
                    )
                    self.is_locked = True
                else:
                    logger.debug(
                        f"{LP} {self.name} has no lock to acquire, creating one..."
                    )
                    self.create_lock()
                    # logger.debug(f"{LP} {self.name} attempting to acquire {self.processor} lock after creating one ...")
                    self.acquire_lock()
                    # self.is_locked = True

            except AlreadyLocked as already_locked_exc:
                logger.error(
                    f"{LP} {self.name} failed to acquire {self.processor} lock: {already_locked_exc}"
                )
                logger.error(
                    f"{LP} timeout waiting for '{self.lock.name}'  for {self.lock.timeout}"
                    f" seconds"
                )
                raise ValueError(
                    f"Timeout waiting for {self.lock.name} portalock for {self.lock.timeout} seconds"
                )
            except AssertionError as assertion_error:
                logger.error(
                    f"{LP} {self.name} failed to acquire {self.processor} lock: {assertion_error}"
                )
                raise ValueError(f"AssertionError: {assertion_error}")
            except Exception as e:
                logger.error(f"{LP}acquire_lock: {e}")
                raise e
        else:
            logger.debug(f"{LP} {self.name} locks are disabled")
        return self.lock

    def release_lock(self):
        if locks_enabled():
            if self.lock:
                if not self.is_locked:
                    logger.debug(
                        f"{LP} '{self.name}' already released '{self.lock.name}' -- {self.lock.lock = }"
                    )
                    return
                logger.debug(
                    f"{LP} '{self.name}' release_lock CALLED - {self.lock.name = } -- {self.is_locked = }"
                )
                self.lock.release()
                self.is_locked = False
                logger.debug(
                    f"{LP} '{self.name}' released '{self.lock.name}' -- {self.is_locked = }"
                )
