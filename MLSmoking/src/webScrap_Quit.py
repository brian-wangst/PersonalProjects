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
import forumScript as quit1


# def writeIntoCSV(reply,post):
#     with open('seleniumData.csv', 'a') as outcsv:
#         writer = writer(outcsv)
#         writer.writerow(['index','#reply', 'post'])
#         for a in post:
#             for b in a:
#                 print(b)
#                 writer.writerow([count,reply,b])
#                 count = count + 1   
#     return 
def get_urls(web_url_str):
    listOfURL = []
    response = requests.get(str(web_url_str))
    soup = BeautifulSoup(response.text, 'html.parser')
    pattern = re.compile("https://www.quittrain.com/topic/.*/$")
    results = soup.find_all('a', href = pattern)

    numbers = soup.find_all('span', class_="ipsDataItem_stats_number")
    t = 0
    replies =[]
    for i in range(len(numbers)):
        if((i%2) == 0):
            replies.append(numbers[i].string)
    for result in results:
        try:
            print(result['href'])
            print("replies amount:", replies[t])
            print("fuck tim")
            listOfURL.append(result['href'])
            t = t + 1
            print(t)
        except:
            print("end of the links")
    return listOfURL

def getAllURLS():
    browser = webdriver.Chrome()
    browser.get("https://www.quittrain.com/forum/2-quit-smoking-discussions/")
    SCROLL_PAUSE_TIME = 1.0
    end_of_scroll = True
    listOfURLS = []
    count = 0
    while(end_of_scroll):
        if(count >= 20):
            #browser.quit()
            break
        count = count + 1
        print("COUNT: " ,count)
        listOfURLS.append(get_urls(browser.current_url)) 
        time.sleep(SCROLL_PAUSE_TIME)
        try:
            browser.find_element_by_class_name("ipsPagination_next").click()
        except WebDriverException:
            end_of_scroll = False
            print ("Element is not clickable")
    return listOfURLS

def main():
    urls = getAllURLS()
    for a in urls:
        for b in a:
            quit1.writeIntoCSV(b)

if __name__ == "__main__":
    main()



