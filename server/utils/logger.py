import logging
import os
import sys
import time
import datetime

# Import LOGS_DIR after config sets up sys.path
from src.config import LOGS_DIR

class SessionLogger:
    """File + stdout logger for the entire server session."""

    def __init__(self):
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(LOGS_DIR, exist_ok=True)
        self.log_path = os.path.join(LOGS_DIR, f"server_session_{ts}.log")
        self.start_time = time.time()

        self._logger = logging.getLogger("model_server")
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers.clear()

        fmt = logging.Formatter("[%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s",
                                datefmt="%H:%M:%S")

        fh = logging.FileHandler(self.log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        self._logger.addHandler(fh)

        sh = logging.StreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        self._logger.addHandler(sh)

    def info(self, msg):
        self._logger.info(msg)

    def debug(self, msg):
        self._logger.debug(msg)

    def error(self, msg):
        self._logger.error(msg)

    def warning(self, msg):
        self._logger.warning(msg)

    def stop(self):
        elapsed = time.time() - self.start_time
        mins, secs = divmod(int(elapsed), 60)
        self.info(f"Server stopped (session duration: {mins}m {secs}s)")
        for h in self._logger.handlers[:]:
            h.close()
            self._logger.removeHandler(h)

    def get_log_contents(self):
        """Read the current log file contents."""
        try:
            with open(self.log_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception:
            return ""

# Global logger instance
logger = None

def init_logger():
    global logger
    if logger is None:
        logger = SessionLogger()
    return logger

def get_logger():
    return logger
