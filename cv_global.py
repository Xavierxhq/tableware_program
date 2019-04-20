Global = None
WIDTH = 300
HEIGHT = 300

def _init():
	global Global
	Global = {}

def set_value(key: str, value):
	global Global
	Global[key] = value

def get_value(key: str):
	global Global
	return Global[key]
