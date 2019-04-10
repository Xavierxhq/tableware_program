# -*- coding: UTF-8 -*-

import smtplib
from email.mime.text import MIMEText
from email.header import Header


class Emailer():
	def __init__(self, sender, receivers):
		self.sender = sender
		self.receivers = receivers
		self.smtpObj = smtplib.SMTP('localhost', 25)

	def send(self, content, subject='CV Experiment Report', source='CV Project', destination='CV Report'):
		# 三个参数：第一个为文本内容，第二个 plain 设置文本格式，第三个 utf-8 设置编码
		message = MIMEText(content, 'plain', 'utf-8')
		message['From'] = Header(source, 'utf-8')   # 发送者
		message['To'] =  Header(destination, 'utf-8')        # 接收者
		message['Subject'] = Header(subject, 'utf-8')
		self.smtpObj.sendmail(self.sender, self.receivers, message.as_string())
		print('Email sent.')
