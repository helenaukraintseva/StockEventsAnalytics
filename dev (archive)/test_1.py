import datetime as dt


value = 1745235531

def get_agg_timestamp(ts):
    dt_obj = dt.datetime.fromtimestamp(ts)
    print(int(dt_obj.replace(microsecond=0).timestamp()))
    print(int(dt_obj.replace(second=0, microsecond=0).timestamp()))
    minute = (dt_obj.minute // 5) * 5
    print(int(dt_obj.replace(minute=minute, second=0, microsecond=0).timestamp()))

dt_obj = dt.datetime.fromtimestamp(value)

print(dt_obj)
now = dt.datetime.now()
print(now)
timestamp = now.timestamp()
print(timestamp)
get_agg_timestamp(timestamp)  #