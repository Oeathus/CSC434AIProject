from urllib.request import urlopen
import re
import os
import pandas as pd

dateFixer = re.compile(r'(\d{4}-\d{2}-\d{2})\s\d{2}:\d{2}')
spaceRemover = re.compile(r'(?<=,)\s{3}(?=,)')

roc_url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?" \
          "station=ROC&data=tmpf&data=dwpf&data=relh&data=drct&data=sknt&data=alti&data=mslp&data=p01i&data=vsby" \
          "&year1=2020&month1=1&day1=1&year2=2020&month2=12&day2=14&tz=America%2FNew_York&format=onlycomma" \
          "&latlon=no&elev=no&missing=empty&trace=empty&direct=no&report_type=1&report_type=2"
buf_url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?" \
          "station=IAG&station=BUF&data=tmpf&data=dwpf&data=relh&data=drct&data=sknt&data=alti&data=mslp&data=p01i&data=vsby" \
          "&year1=2020&month1=1&day1=1&year2=2020&month2=12&day2=14&tz=America%2FNew_York&format=onlycomma" \
          "&latlon=no&elev=no&missing=empty&trace=empty&direct=no&report_type=1&report_type=2"
syr_url = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?" \
          "station=FZY&station=SYR&data=tmpf&data=dwpf&data=relh&data=drct&data=sknt&data=alti&data=mslp&data=p01i&data=vsby" \
          "&year1=2020&month1=1&day1=1&year2=2020&month2=12&day2=14&tz=America%2FNew_York&format=onlycomma" \
          "&latlon=no&elev=no&missing=empty&trace=empty&direct=no&report_type=1&report_type=2"

roc_data = urlopen(roc_url, timeout=300).read().decode("utf-8")
buf_data = urlopen(buf_url, timeout=300).read().decode("utf-8")
syr_data = urlopen(syr_url, timeout=300).read().decode("utf-8")

roc_data = dateFixer.sub(r'\1', roc_data)
roc_data = spaceRemover.sub("", roc_data)
roc_data = re.sub("valid", "day", roc_data)

buf_data = dateFixer.sub(r'\1', buf_data)
buf_data = spaceRemover.sub("", buf_data)
buf_data = re.sub("valid", "day", buf_data)

syr_data = dateFixer.sub(r'\1', syr_data)
syr_data = spaceRemover.sub("", syr_data)
syr_data = re.sub("valid", "day", syr_data)
