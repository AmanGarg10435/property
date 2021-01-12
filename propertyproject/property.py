from bs4 import BeautifulSoup
import requests
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression


def scrape():
    titles = []
    builds = []
    details = []
    land_areas = []
    built_areas = []
    prices = []
    rooms = []
    links = []


    def remove_pun(number):
        number2 = ""
        for n in number:
            if n != "$" and n != ",":
                number2 += n
        return number2

    for i in range(1,10):
        source = requests.get("https://www.stproperty.sg/search/sale/landed?modelsNotRequired=CLUSTER%20HOUSE&cdResearchSubTypes=10&maxSalePrice=50000000&selectedDistrictIds=10&page=" + str(i)).text
        soup = BeautifulSoup(source, "lxml")

        containers = soup.find_all("div", class_="listingDetailsDiv")
        for container in containers:
            title = container.find("div", class_="title-row insideListingTitleRow visible-lg").a.span.text
            build = container.find("div", class_="listingDetailType").span.text
            if len(build) <4:
                continue
            detail = container.find("div", class_="listingDetailValues").text[10:].split()
            if len(detail) != 11:
                continue
            else:
                land_area = detail[0]
                built_area = detail[6]
            price = container.find("div", class_="listingDetailPrice adjusted-line-height").text[10:]
            room = container.find("div", class_="listingDetailRoomNo").text
            if len(room) > 4 or len(room) == 3:
                continue
            if len(room) == 4:
                room = int(room[0]) + int(room[2])
            link = "stproperty.sg" + container.find("div", class_="title-row insideListingTitleRow visible-lg").a["href"]

            titles.append(title)
            builds.append(build)
            rooms.append(int(room))
            links.append(link)

            built_area = remove_pun(built_area)
            land_area = remove_pun(land_area)
            price = remove_pun(price)

            built_areas.append(int(built_area))
            land_areas.append(int(land_area))
            prices.append(int(price))


    dataframe = pd.DataFrame({
        "title": titles,
        "type": builds,
        "built_area": built_areas,
        "land_area": land_areas,
        "price": prices,
        "rooms": rooms,
        "link": links,
    })


    dataframe.to_csv("house_data2.csv")


data = pd.read_csv("house_data.csv")
data.reset_index(drop=True, inplace=True)
data.drop(data.columns[0],axis=1,inplace=True)

x = np.array(data.drop(["title", "type", "link", "price"], 1))
y = np.array(data["price"])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)


model = LinearRegression()
model.fit(x_train,y_train)
acc = model.score(x_train,y_train)
predictions = model.predict(x_test)

print(acc)
for i in range(len(predictions)):
    print(round(predictions[i],1), y_test[i])




