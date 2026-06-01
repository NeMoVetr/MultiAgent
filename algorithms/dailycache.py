import functools
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class DailyCache:
    """
    Декоратор: кэширует результат функции один раз в сутки (00:00 UTC)
    """

    def __init__(self, func):
        self.func = func
        self._cache = {}
        self._last_update_date = None
        functools.update_wrapper(self, func)

    def __call__(self, *args, **kwargs):
        today_utc = datetime.now(timezone.utc).date()

        if self._last_update_date != today_utc:
            logger.info(f"Daily cache reset. New UTC day: {today_utc}")
            self._cache.clear()
            self._last_update_date = today_utc

        # Ключ кэша
        cache_key = (args, tuple(sorted(kwargs.items())))

        if cache_key not in self._cache:
            logger.info(f"Cache miss → calling {self.func.__name__}")
            result = self.func(*args, **kwargs)
            self._cache[cache_key] = result
        else:
            logger.debug(f"Cache hit for {self.func.__name__}")

        return self._cache[cache_key]