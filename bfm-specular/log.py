import requests
from logging import Handler, Formatter
import logging
import datetime

TELEGRAM_TOKEN = ''
TELEGRAM_CHAT_ID = ''

def telegram_logger(msg):
	print(msg)
	return
	payload = {
		'chat_id': TELEGRAM_CHAT_ID,
		'text': msg
	}
	requests.post("https://api.telegram.org/bot{token}/sendMessage".format(token=TELEGRAM_TOKEN),
							 data=payload)


def telegram_upload_image(path):
	return
	payload = {
		'chat_id': (None, TELEGRAM_CHAT_ID),
		'photo': (path, open(path, 'rb'))
	}
	requests.post("https://api.telegram.org/bot{token}/sendPhoto".format(token=TELEGRAM_TOKEN),
							 files=payload)

