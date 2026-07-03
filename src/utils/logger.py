import logging, sys
from pathlib import Path
from config import LOG_LEVEL

Path("logs").mkdir(exist_ok=True)

def get_logger(name: str) -> logging.Logger:
    log = logging.getLogger(name)
    if log.handlers: return log
    log.setLevel(LOG_LEVEL)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt)
    fh = logging.FileHandler("logs/platform.log"); fh.setFormatter(fmt)
    log.addHandler(sh); log.addHandler(fh)
    return log