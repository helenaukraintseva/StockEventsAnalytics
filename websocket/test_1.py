import datetime as dt
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def get_agg_timestamp(ts: float):
    """
    Печатает округлённые варианты временной метки:
      - с точностью до секунды
      - с точностью до минуты
      - округление по 5 минутам

    :param ts: Timestamp в секундах
    :return: None
    """
    dt_obj = dt.datetime.fromtimestamp(ts)
    logging.info("⏱️  Время: %s", dt_obj)
    logging.info("▶ Точность до секунды: %d", int(dt_obj.replace(microsecond=0).timestamp()))
    logging.info("▶ Точность до минуты: %d", int(dt_obj.replace(second=0, microsecond=0).timestamp()))
    minute = (dt_obj.minute // 5) * 5
    logging.info("▶ Округление по 5 минутам: %d", int(dt_obj.replace(minute=minute, second=0, microsecond=0).timestamp()))


if __name__ == "__main__":
    now = dt.datetime.now()
    timestamp = now.timestamp()
    get_agg_timestamp(timestamp)
