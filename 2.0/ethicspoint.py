import requests
import os
import configs
from dotenv import load_dotenv, find_dotenv
   
def get_client_greeting(dnis:str):  
    service_url = f"{configs.intakeUrl}ClientServices.asmx/GetClientByDnis?dnis={dnis}"
    print(f"ServiceUrl: {service_url}")
    return __get(service_url)
    
def get_guidelines(dnis:str):
    # First, we need to fetch the client by DNIS
    service_url = f"{configs.intakeUrl}ClientServices.asmx/GetClientByDnis?dnis={dnis}"
    #print(f"Client Url: {service_url}")
    return __get(service_url)
   # mainPageId = client.MainPageID

    # Second, we fetch the guideline using client.MainPageId
    #service_url = f"{configs.epUrl}MainPageServices.asmx/GetMainPageById?mainPageId={mainPageId}"
   # print(f"MainPage Url: {service_url}")
    #return __get(service_url)

def get_issue_types(clientKey:str, tierId:int, epimApiAuthKey:str, languageCode:str, locationId:int, locationDataPrivacy:bool):    
    service_url = f"{configs.epUrl}v6.0/Intake/GetMobileIssueTypesWithDefaults"
    print(f"ServiceUrl: {service_url}")
    data = {
        "clientKey": clientKey,
        "tierId": tierId,
        "epimApiAuthKey": epimApiAuthKey,
        "languageCode": languageCode,
        "locationId": locationId,
        "locationDataPrivacy": locationDataPrivacy
    } 
    return __post(service_url, data)

def get_locations(clientKey:str, epimApiAuthKey:str, platformCorrectionId:str):
    service_url =  f"{configs.epUrl}v6.0/Locations/GetLocations"
    print(f"ServiceUrl: {service_url}")
    data = {
        "clientkey":clientKey, 
        "EpimApiAuthKey":epimApiAuthKey, 
        "platformCorrectionId": platformCorrectionId
    }
    return __post(service_url, data)

def get_violation_types(clientKey:str, tierId:int, epimApiAuthKey:str, languageCode:str, locationId:int, locationDataPrivacy:bool):
    service_url = f"{configs.epUrl}v6.0/Intake/GetMobileIssueTypesWithDefaults"
    print(f"ServiceUrl: {service_url}")
    data = {
        "clientKey": clientKey,
        "tierId": tierId,
        "epimApiAuthKey": epimApiAuthKey,
        "languageCode": languageCode,
        "locationId": locationId,
        "locationDataPrivacy": locationDataPrivacy
    } 
    return __post(service_url, data)

def get_questions(clientKey:str, epimApiAuthKey:str, violationTypeId:int, dataPrivacyMode:int, languageCode:str):
    service_url = f"{configs.epUrl}v6.0/ViolationQuestion/GetViolationQuestionAndAnswersForPlatformPackages"
    print(f"ServiceUrl: {service_url}")
    data = { 
        "clientKey": clientKey, 
        "violationTypeId": violationTypeId, 
        "dataPrivacyMode": dataPrivacyMode, 
        "languageCode": languageCode,
        "epimApiAuthKey": epimApiAuthKey
    } 
    return __post(service_url, data)

def __get( url):
    # make an HTTP request to the URL
    response = requests.get(url, verify=not configs.debug_on)
    # check if the request was successful
    if response.status_code == 200:
        # parse the JSON response
        data = response.json()
        # extract the data from the JSON response
        return data
    else:
        # if the request was not successful, return an empty list
        return []

def __post(url, data):
    print(f"PostData: {data}")
    response = requests.post(url, json=data, verify=not configs.debug_on)
    if response.status_code == 200:
        data = response.json()
        return data
    else:
        return []