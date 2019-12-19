from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait 
from selenium.webdriver.common.by import By
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.support.ui import Select
from selenium.webdriver.common.keys import Keys
import time
import requests
import re
from bs4 import BeautifulSoup
from csv import writer

csv_count = 0


def parsePage(webLink):
	paragraphs = []
	dataOutput = []
	response = requests.get(webLink).text
	soup = BeautifulSoup(response,'html.parser')
	posts = soup.findAll("div",{"class":"ipsType_normal ipsType_richText ipsContained"},"p")

	for post in posts:
		paragraphs.append(post.text.replace("\n", ''))

	# for i in range(0,len(paragraphs)):

	# 	print(i, " " ,paragraphs[i])
	return paragraphs, True
def seleniumInteract(webLink):
	outPut = []
	browser = webdriver.Chrome()

	browser.get(webLink)
	SCROLL_PAUSE_TIME = 30.0
	end_of_scroll = True
	while(end_of_scroll):

		parsed, parsedFlag = parsePage(str(browser.current_url))
		outPut.append(parsed)
		if(parsedFlag == True):
			try:
				browser.find_element_by_class_name("ipsPagination_next").click()
			except WebDriverException:
				end_of_scroll = False
				print ("Element is not clickable")
	browser.close()
	return outPut


def writeIntoCSV(post):
	print("fuck brian")
	final = seleniumInteract(post)
	fin = False
	global csv_count
	# outputFile = open("seleniumData.txt", "w")
	with open('seleniumData.csv', 'a') as outcsv:
		writer1 = writer(outcsv)
		for a in final:
			for b in a:
				writer1.writerow([csv_count,b])
				csv_count = csv_count + 1
		return True		

	