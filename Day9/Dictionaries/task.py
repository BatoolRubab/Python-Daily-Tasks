capitals = {
    "Pakistan" : "Islamabad",
    "Turkey" : "Istanbul",
    "France" : "Paris",
}
# nested list in a dictionary
# travel_log = {
#     "Pakistan" : ["Islamabad", "Karachi", "Lahore", "Kashmir"],
#     "Turkey" : ["Istanbul", "Okan", "Antalya", "Konya"],
#     "France" : ["Paris", "Lillie", "Lyon", "Dijon"]
# }
# print(travel_log)
# print(travel_log["Pakistan"][2])

# nested lists
# nested_list = ["A", "B",["C","D"]]
# print(nested_list[2][1])

travel_log = {
    "Pakistan" : {
        "total_visits": 8,
        "cities_visit" :["Islamabad", "Karachi", "Lahore", "Kashmir"],
    },
    "Turkey" : {
        "total_visits": 5,
        "cities_visited":["Istanbul", "Okan", "Antalya", "Konya"],
    },
    "France" :{
    "total_visits" : 3,
    "cities_visited": ["Paris", "Lillie", "Lyon", "Dijon"],
    },
}
print(travel_log["Pakistan"]["cities_visit"][3])