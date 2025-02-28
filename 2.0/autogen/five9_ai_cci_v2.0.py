from dotenv import load_dotenv
load_dotenv() 
import numpy as np
import pandas as pd
import json
import re
import os
import openai
import random
import time
from datetime import datetime
from openai import AzureOpenAI
from sentence_transformers import SentenceTransformer
import autogen
from autogen import GroupChat
from autogen import GroupChatManager
from autogen import AssistantAgent
from autogen.agentchat.contrib.retrieve_user_proxy_agent import RetrieveUserProxyAgent
from autogen import ConversableAgent, UserProxyAgent, config_list_from_json
from autogen.retrieve_utils import TEXT_FORMATS
from autogen import AssistantAgent, ConversableAgent, UserProxyAgent


llm_config = {
    "config_list": [
        {
            "model": "gpt-4o",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "api_type": "azure",
            "base_url": os.getenv("OPENAI_API_BASE"),
            "api_version": os.getenv("API_VERSION"),
        },
    ],
    "temperature": 0.0,
    "timeout": 300,
}

load_dotenv()  

api_key = os.getenv("OPENAI_API_KEY")
azure_endpoint = os.getenv("OPENAI_API_BASE")
api_version = os.getenv("API_VERSION")
model = "gpt-4o"
temprature = 0.0
top_p = 1.0

def get_response_client(template, text, temprature=temprature, top_p=top_p, model=model, azure_endpoint=azure_endpoint, api_key=api_key, api_version=api_version):
    time.sleep(1)

    client = AzureOpenAI(
        azure_endpoint = azure_endpoint, 
        api_key= api_key,  
        api_version= api_version
    )

    conversation = [
        {"role": "system", "content": template},
        {"role": "user", "content": text},
    ]

    response = client.chat.completions.create(
        model=model,
        temperature=temprature,
        top_p= top_p,
        max_tokens=4000,
        response_format={ "type": "json_object" },  #JSON mode
        messages=conversation
    )

    return response.choices[0].message.content

##########################   The model expects context of the data from API calls. ########################## 

clientKey = """Call Center Training 32 - Target Corporation"""

guidelines = """Response from /v6.0/Intake/guidelines: {"clientKey": "Call Center Training 32 - Target Corporation", "CustomText": "\nOpening\n\nThank you for calling the Call Center Training 32 - Target Corporation Hotline, this is (CS Name). Would you like to file a report or follow up on an existing report? Finished\n\nDONT READ: Factory workers may report on this line. Accept reports from factory workers who work for or with Call Center Training 32 - Target Corporation.\nREAD: Before we begin, please know that this process may take 10-15 minutes or more depending on the nature of your concern or question. To ensure proper submission of your report, you will need to remain on the line until I've read the report narrative details back to you. I will also provide you with a report key for follow-up purposes. Do you wish to proceed?\n\nDONT READ: Please accept reports from Starbucks employees that work inside Target retail stores.\nDONT READ: Please accept reports from SHIPT employees.\nDONT READ: Do not accept reports for CVS employees.\n"}"""

locations = """Response from /v6.0/Intake/locations: {"clientKey": "Call Center Training 32 - Target Corporation", "Locations": [{"Case_CompanyLocation": "Branch Office D", "Case_CompanyCity": "Austin", "Case_CompanyState": "TX", "Case_CompanyZip": "73301", "Case_CompanyCountry": "USA", "Case_LocationCustomField1": "Technical Support", "Case_LocationCustomField2": "SW", "Case_LocationCustomField3": "District 7", "Case_LocationCustomField4": "Floor 5"}, {"Case_CompanyLocation": "Branch Office E", "Case_CompanyCity": "Denver", "Case_CompanyState": "CO", "Case_CompanyZip": "80201", "Case_CompanyCountry": "USA", "Case_LocationCustomField1": "Sales", "Case_LocationCustomField2": "MW", "Case_LocationCustomField3": "District 4", "Case_LocationCustomField4": "Building 1"}, {"Case_CompanyLocation": "Branch Office F", "Case_CompanyCity": "Orlando", "Case_CompanyState": "FL", "Case_CompanyZip": "32801", "Case_CompanyCountry": "USA", "Case_LocationCustomField1": "Customer Service", "Case_LocationCustomField2": "SE", "Case_LocationCustomField3": "District 2", "Case_LocationCustomField4": "Suite 300"}]}"""

IssueTypes = """
Response from /v6.0/Intake/GetMobileIssueTypesWithDefaults: {
    "clientKey": "Call Center Training 32 - Target Corporation",
    "data": {
        "ViolationTypeList": [
            {
                "Description": "Unauthorized access to company systems or data, hacking, or other cyber threats. This includes phishing attempts, malware infections, and other cybersecurity incidents.",
                "Name": "Cybersecurity Breach",
                "ViolationTypeId": 13
            },
            {
                "Description": "Violations related to environmental regulations, including improper disposal of hazardous materials, pollution, and non-compliance with environmental laws.",
                "Name": "Environmental Violation",
                "ViolationTypeId": 14
            },
            {
                "Description": "Reports of physical violence or threats of violence in the workplace. This includes fights, assaults, and other forms of physical aggression.",
                "Name": "Workplace Violence",
                "ViolationTypeId": 15
            },
            {
                "Description": "Instances of theft, embezzlement, or other forms of financial fraud. This includes stealing company property, misappropriating funds, and falsifying financial records.",
                "Name": "Financial Fraud",
                "ViolationTypeId": 16
            },
            {
                "Description": "Violations related to health and safety regulations, including unsafe working conditions, lack of proper safety equipment, and failure to follow safety protocols.",
                "Name": "Health and Safety Violation",
                "ViolationTypeId": 17
            },
            {
                "Description": "Reports of discrimination based on age, including unfair treatment, harassment, or bias against employees or customers due to their age.",
                "Name": "Age Discrimination",
                "ViolationTypeId": 18
            },
            {
                "Description": "Instances of unethical behavior or conflicts of interest, such as accepting gifts or favors in exchange for preferential treatment or using company resources for personal gain.",
                "Name": "Ethical Violation",
                "ViolationTypeId": 19
            },
            {
                "Description": "Reports of violations related to data privacy, including unauthorized sharing of personal information, data breaches, and failure to comply with data protection regulations.",
                "Name": "Data Privacy Violation",
                "ViolationTypeId": 20
            },
            {
                "Description": "Instances of workplace bullying, including repeated mistreatment, verbal abuse, and other forms of hostile behavior that create a toxic work environment.",
                "Name": "Workplace Bullying",
                "ViolationTypeId": 21
            },
            {
                "Description": "Reports of non-compliance with legal or regulatory requirements, including violations of industry standards, government regulations, and company policies.",
                "Name": "Regulatory Non-Compliance",
                "ViolationTypeId": 22
            }
        ]
    }
}
"""

questions = """Response from /v6.0/ViolationQuestion/GetViolationQuestionAndAnswersForPlatformPackages: "{"clientKey": "Call Center Training 32 - Target Corporation", "data": [{"violationTypeId": 13, "ViolationQuestionId": 171, "Question": "Where did this happen?"}, {"violationTypeId": 13, "ViolationQuestionId": 172, "Question": "What type of cyber threat occurred?"}, {"violationTypeId": 13, "ViolationQuestionId": 173, "Question": "Who was involved in the cybersecurity breach?"}, {"violationTypeId": 14, "ViolationQuestionId": 171, "Question": "Where did this happen?"}, {"violationTypeId": 14, "ViolationQuestionId": 174, "Question": "What environmental regulation was violated?"}, {"violationTypeId": 14, "ViolationQuestionId": 175, "Question": "Who was responsible for the environmental violation?"}, {"violationTypeId": 15, "ViolationQuestionId": 171, "Question": "Where did this happen?"}, {"violationTypeId": 15, "ViolationQuestionId": 176, "Question": "What type of violence occurred?"}, {"violationTypeId": 15, "ViolationQuestionId": 177, "Question": "Who was involved in the workplace violence?"}, {"violationTypeId": 16, "ViolationQuestionId": 171, "Question": "Where did this happen?"}, {"violationTypeId": 16, "ViolationQuestionId": 178, "Question": "What type of financial fraud occurred?"}, {"violationTypeId": 16, "ViolationQuestionId": 179, "Question": "Who was involved in the financial fraud?"}, {"violationTypeId": 17, "ViolationQuestionId": 171, "Question": "Where did this happen?"}, {"violationTypeId": 17, "ViolationQuestionId": 180, "Question": "What health and safety regulation was violated?"}, {"violationTypeId": 17, "ViolationQuestionId": 181, "Question": "Who was responsible for the health and safety violation?"}, {"violationTypeId": 18, "ViolationQuestionId": 171, "Question": "Where did this happen?"}, {"violationTypeId": 18, "ViolationQuestionId": 182, "Question": "What type of age discrimination occurred?"}, {"violationTypeId": 18, "ViolationQuestionId": 183, "Question": "Who was involved in the age discrimination?"}, {"violationTypeId": 19, "ViolationQuestionId": 171, "Question": "Where did this happen?"}, {"violationTypeId": 19, "ViolationQuestionId": 184, "Question": "What type of ethical violation occurred?"}, {"violationTypeId": 19, "ViolationQuestionId": 185, "Question": "Who was involved in the ethical violation?"}, {"violationTypeId": 20, "ViolationQuestionId": 171, "Question": "Where did this happen?"}, {"violationTypeId": 20, "ViolationQuestionId": 186, "Question": "What type of data privacy violation occurred?"}, {"violationTypeId": 20, "ViolationQuestionId": 187, "Question": "Who was involved in the data privacy violation?"}, {"violationTypeId": 21, "ViolationQuestionId": 171, "Question": "Where did this happen?"}, {"violationTypeId": 21, "ViolationQuestionId": 188, "Question": "What type of workplace bullying occurred?"}, {"violationTypeId": 21, "ViolationQuestionId": 189, "Question": "Who was involved in the workplace bullying?"}, {"violationTypeId": 22, "ViolationQuestionId": 171, "Question": "Where did this happen?"}, {"violationTypeId": 22, "ViolationQuestionId": 190, "Question": "What type of regulatory non-compliance occurred?"}, {"violationTypeId": 22, "ViolationQuestionId": 191, "Question": "Who was involved in the regulatory non-compliance?"}]}"""

implicated_parties = """{"Affected Party":"","Perpetrator":"","Witness":"","Other":""}"""

#############################################################################################################

template_memory = """
Role: JSON Chat Memory: You will be given a new message and a JSON Chat Memory.
Your primary responsibility is to memorize the new message and the JSON Chat Memory.
You will take to action!

Do not wrap the output in quotation marks.
Do not wrap the output in code block delimiters (```).
"""

template_router = """
Role: You are a classification agent. 
Objective: You will be given the caller's new message, {{JSON Chat Memory}}, {{group_chat}} priorities, and applicable {{group_chat}}'s {{Termination Requirements}}. 
Your primary responsibility is to detect the the action id for the new message to route to by checking the {{JSON Chat Memory}} against the applicable {{Termination Requirements}} and determine the next {{group_chat}} only from the following {{group_chat}} options.

{
  "group_chat": 
  [
    {"id": "main_imminent_issue",
    "priority": 1, 
    "role": "This function is used when AI detects imminent issue",
    "termination_requirements": {
        "requirement_1": "presence of imminent issue event in context",
      }
    },
    {"id": "main_guidelines",
    "priority": 2, 
    "role": "This function is used when the JSON Chat Memory does not contain the customer's responses to the READ messages in guidelines/instructions.",
    "termination_requirements": {
        "requirement_1": "Answer to all READ Messages",
        "requirement_2": "Have identified the caller's intent to call, whether it is a new report or a follow-up."
      }
    },
    {"id": "main_locations",
    "priority": 3, 
    "role": "This function is used when the JSON Chat Memory does not contain the customer responses to the location indentification requirements.",
    "termination_requirements": {
        "requirement_1": "Respond to all location-related inquiries", 
        "requirement_2": "Identify the State, City, and Building details."
      }
    },
    {"id": "main_issue_questions",
    "priority": 4, 
    "role": "This function is used when input message is the transcription of thge event.",
    "termination_requirements": {
        "requirement_1": "transcription of the report is in memory"
      }
    },
    {"id": "main_issue_questions_follow_up",
    "priority": 5, 
    "role": "This function is used when the JSON Chat Memory does not contain the customer responses to the issue questions",
    "termination_requirements": {
        "requirement_1": "Answer to all questions"
      }
    },
    {"id": "main_implicated_parties",
    "priority": 6, 
    "role": "This function is used when the JSON Chat Memory does not contain the customer responses to the issue questions",
    "termination_requirements": {
        "requirement_1": "Information of implicated parties including their names",
        "requirement_2": "Information of implicated parties including their roles",
        "requirement_3": "Information of implicated parties including their job titles"
      }
    },
    {"id": "terminate_chat",
    "priority": 7, 
    "role": "This function is used when the JSON Chat Memory does contain the customer responses to the previous priorities.",
    "termination_requirements": {
        "requirement_1": "Answer to all questions"
      }
    }
  ]
}

If the {{Termination Requirements}} are satisfied, your detected action will be terminate_chat and you are Done!.

example: 

{"action":"main_issue_questions"}

Do not wrap the output in quotation marks.
Do not wrap the output in code block delimiters (```).
"""

template_imminent_issue = """
At any point during the interview, AI detects imminent issue and flags report with priority with this message: {"Final Message":"Flag report with priority"}
At any point during the interview, (If suicide ideations deteted, transfer call to CS workflow) with this message: {"Final Message":"transfer call to CS workflow"}

Examples:
Threats of Violence: Reports of threats of physical harm or violence from coworkers, supervisors, or customers.
Harassment or Assault: Incidents of sexual harassment, physical assault, or severe bullying.
Medical Emergencies: Situations involving severe health issues such as heart attacks, severe allergic reactions, or injuries.
Unsafe Working Conditions: Immediate dangers due to unsafe conditions like hazardous materials, lack of safety equipment, or structural hazards.
Fire or Explosion: Reports of fires, explosions, or similar emergencies posing immediate threats to safety.
Active Shooter or Hostage Situation: Incidents involving an active shooter or hostage situation within the workplace.
Severe Psychological Distress: Cases of severe psychological distress or suicidal thoughts/intentions.

(If suicide ideations deteted, transfer call to CS workflow) with this message: "transfer call to CS workflow"

Do not wrap the output in quotation marks.
Do not wrap the output in code block delimiters (```).
"""

########################################### ADD SESSION MEMORY ###############################################

with open("memory.json", "w") as json_file:
    json.dump([], json_file)

def load_chat_history():
    with open("memory.json", "r") as json_file:
        chat_history_data = json.load(json_file)
    return chat_history_data

chat_history = load_chat_history()

###############################################################################################################

memory_agent = ConversableAgent(
    name="memory_agent",
    system_message= f"{template_memory} + chat history: {chat_history}",
    llm_config=llm_config,
    human_input_mode= "NEVER",
)

router_agent = ConversableAgent(
    name="router_agent",
    system_message= f"{template_router} + guidelines instructions: {guidelines}",
    llm_config=llm_config,
    human_input_mode= "NEVER",
)

def state_transition_router(last_speaker, groupchat):
    messages = groupchat.messages

    if last_speaker is memory_agent:
        return router_agent
    elif last_speaker is router_agent:
        return None

groupchat_router = autogen.GroupChat(
    agents=[memory_agent,
            router_agent
           ],
    messages=[],
    max_round=10,
    speaker_selection_method=state_transition_router,
)

manager_router = autogen.GroupChatManager(groupchat=groupchat_router, llm_config=llm_config)

template_process_guidelines = """
Extract text in JSON from text or HTML elements similar to the following:

{
  "OPENING": "Text from opening HTML element",
  "READ": {
    "READ_1": "1st Text to read",
    "READ_2": "2nd Text to read",
  },
  "DONT_READ": {
    "DONT_READ_1": "1st Text not to read",
    "DONT_READ_2": "2nd Text not to read",
  },
  "Other_Instructions": {
    "new_report": "Other instructions for new report",
    "follow_up": "Other instructions for follow-up"
  }
}

Do not wrap the output in quotation marks.
Do not wrap the output in code block delimiters (```).
"""

guidelines = get_response_client(template_process_guidelines, guidelines)
guidelines = json.loads(guidelines)

user_agent = ConversableAgent(
    name="user_agent",
    system_message= f"You are an assistant",
    llm_config=llm_config,
    human_input_mode= "NEVER",
)

imminent_issue_agent = ConversableAgent(
    name="imminent_issue_agent",
    system_message= f"Imminent Issues: {template_imminent_issue}",
    llm_config=llm_config,
    human_input_mode= "NEVER",
)

def state_transition_imminent_issue(last_speaker, groupchat):
    messages = groupchat.messages

    if last_speaker is user_agent:
        return imminent_issue_agent
    elif last_speaker is imminent_issue_agent:
        return None

groupchat_imminent_issue = autogen.GroupChat(
    agents=[user_agent,
            imminent_issue_agent
           ],
    messages=[],
    max_round=20,
    speaker_selection_method=state_transition_imminent_issue,
)

manager_imminent_issue = autogen.GroupChatManager(groupchat=groupchat_imminent_issue, llm_config=llm_config)

def main_imminent_issue():
    memory = read_chat_history()
    memory.append(new_message)

    last_message = json.dumps(memory)

    user_agent.initiate_chat(recipient=manager_imminent_issue, message=last_message, clear_history=False)
    messages_json = manager_imminent_issue.messages_to_string(manager_imminent_issue.groupchat.messages)

    new_messages = json.loads(messages_json)
    memory.extend(new_messages)

    write_chat_history(memory)
    extract_and_print_questions()

template_guidelines = """
Role: Ethics and Compliance Contact Center Agent
Objective: You will be given the callers [previous_chat_history] and applicable [guidelines]. 
Your primary responsibility is to determine the call reasons "New Report"/ "Follow-Up" after gathering all answers to the [guidelines] READ messages in a valid JSON.

Don't repeat the content of the previous_chat_history.
Under no circumstances determine the call reasons if you don't have an answer to the READ messages/questions.
Your response should consist of only one question at a time retrieved from the READ messages/questions.
You will categorize the call as either a new report or a follow-up on an existing case.

Call Reasons:
New Report: The caller is calling to file a new report.
Follow-Up: The caller is calling to follow up on an existing case.

Steps to Determine Call Reason:
Always start with the "Opening" text from the [guidelines]. check if the "Opening" statement has already been used in the current session 
and then proceed to the next relevant READ messages.

Do not wrap the output in quotation marks.
Do not wrap the output in code block delimiters (```).
"""

user_agent = ConversableAgent(
    name="user_agent",
    system_message= f"You are an assistant",
    llm_config=llm_config,
    human_input_mode= "NEVER",
)

guidelines_agent = ConversableAgent(
    name="guidelines_agent",
    system_message= f"Report Datetime: {template_guidelines} + guidelines: {guidelines}",
    llm_config=llm_config,
    human_input_mode= "NEVER",
)

def state_transition_guidelines(last_speaker, groupchat):
    messages = groupchat.messages

    if last_speaker is user_agent:
        return guidelines_agent
    elif last_speaker is guidelines_agent:
        return None

groupchat_guidelines = autogen.GroupChat(
    agents=[user_agent,
            guidelines_agent
           ],
    messages=[],
    max_round=20,
    speaker_selection_method=state_transition_guidelines,
)

manager_guidelines = autogen.GroupChatManager(groupchat=groupchat_guidelines, llm_config=llm_config)

def collect_json_data():
    follow_ups = {}
    while not follow_ups:
        user_input = input('Please provide data in JSON format (e.g. {"key": "value"} or {"key1": "value1", "key2": "value2"}): ')
        try:
            user_data = json.loads(user_input)
            if isinstance(user_data, dict):
                follow_ups.update(user_data)
            else:
                print("The JSON input must be an object (key-value pairs). Please try again.")
        except json.JSONDecodeError:
            print("Invalid JSON input. Please try again.")
    return follow_ups

def extract_questions_from_memory(memory):
    last_message = memory[-1]
    content = last_message["content"]

    try:
        content_dict = json.loads(content)
        return content_dict
    except json.JSONDecodeError:
        questions = [q.strip() + "?" if not q.strip().endswith("?") else q.strip() for q in content.split("\n") if q.strip()]
        questions_dict = {q: "" for q in questions}
        # return questions_dict
        return json.dumps(questions_dict)

def read_chat_history():
    try:
        with open("memory.json", "r") as json_file:
            try:
                return json.load(json_file)
            except json.JSONDecodeError:
                return []  
    except FileNotFoundError:
        return [] 

def extract_and_print_questions():
    memory = read_chat_history()
    questions = extract_questions_from_memory(memory)
    print(questions)

def write_chat_history(chat_history_data):
    seen = set()
    unique_messages = []

    for message in chat_history_data:
        message_str = json.dumps(message, sort_keys=True, ensure_ascii=False)  
        if message_str not in seen:
            seen.add(message_str)
            unique_messages.append(message)


    with open("memory.json", "w", encoding="utf-8") as json_file:
        json.dump(unique_messages, json_file, indent=4, ensure_ascii=False) 

def main_guidelines(new_message):
    memory = read_chat_history()
    memory.append(new_message)

    last_message = json.dumps(new_message)

    user_agent.initiate_chat(recipient=manager_guidelines, message=last_message, clear_history=False)
    messages_json = manager_guidelines.messages_to_string(manager_guidelines.groupchat.messages)

    new_messages = json.loads(messages_json)
    memory.extend(new_messages)

    write_chat_history(memory)
    extract_and_print_questions()

template_locations = """
Retrieve the location of the event from [locations] by asking a few questions one by one about the state, city, etc., 
using [locations] data.

Do not repeat questions with the same context.

As soon as you detect the location from [locations], respond with a question to confirm "Case_CompanyLocation": "", 
and "Case_CompanyCity": "" , and  "Case_CompanyLocation":"" from the data package available in [locations]. 

Your final question would be to confirm the location of the event. 

Example: 

"Could you please confirm if the event took place in Austin, Texas, in Building D?"

Do not wrap the output in quotation marks.
Do not wrap the output in code block delimiters (```).
"""

locations_agent = ConversableAgent(
    name="locations",
    system_message= f"locations: {locations} + {template_locations}",
    llm_config=llm_config,
    human_input_mode= "NEVER",
)

def state_transition_locations(last_speaker, groupchat):
    messages = groupchat.messages

    if last_speaker is user_agent:
        return locations_agent
    elif last_speaker is locations_agent:
        return None

groupchat_locations = autogen.GroupChat(
    agents=[user_agent,
            locations_agent
           ],
    messages=[],
    max_round=10,
    speaker_selection_method=state_transition_locations,
)

manager_locations = autogen.GroupChatManager(groupchat=groupchat_locations, llm_config=llm_config)

def main_locations(new_message):
    memory = read_chat_history()
    memory.append(new_message)
   
    last_message = json.dumps(new_message)

    user_agent.initiate_chat(recipient=manager_locations, message=last_message, clear_history=False)
    messages_json = manager_locations.messages_to_string(manager_locations.groupchat.messages)

    new_messages = json.loads(messages_json)
    memory.extend(new_messages)

    write_chat_history(memory)
    extract_and_print_questions()

template1 = """You are an ethics and compliance analyst. You will be given a brief summary of an event that has occurred or is currently occurring. 
            Your job is to classify the event using the provided dictionary.
            Your final response should include the retrieved key-value pairs from the given dictionary.
            
            Example: 
            {
                "Reduction of Hours": "Events that are related to an employee having the number of hours they work reduced."
            }
            
            Do not wrap the output in quotation marks.
            Do not wrap the output in code block delimiters (```)."""


template2 = """
            You are a question retrieval engine. 
            You are tasked with retrieving questions based on a given key from a set of key-value pairs. Please follow these steps:
            
            Input Data:
            questions_dictionary: A dictionary where each key is linked to a list of question identifiers (denoted as "q").
            questions_set: A set containing the actual questions.
            Process:
            Look up the provided key in questions_dictionary to find the corresponding list of question identifiers.
            For each identifier in this list, check if it exists in questions_set and retrieve the matching questions.
            Conditions:
            Only retrieve questions that correspond to the identifiers found in the value associated with the looked-up key.
            Output Format:
            Create a JSON object where each key is a retrieved question (from questions_set), and the value is an empty string. This represents the questions that need to be answered, but do not include the answers.
                        
            Example:
            {
                "When did this happen?": "",
                "What kind of discrimination took place (race/gender/etc.)?": ""
            }
            
            Do not wrap the output in quotation marks.
            Do not wrap the output in code block delimiters (```).
            """

template3 = """
            You are an ethics and compliance analyst. You will be given a brief report of an event 
            that has occurred or is currently occurring and your job is to answer the retrieved questions. 
            Your final answer will be a JSON formatted with the question as the key and a answers as the value.
            Keep in mind answering these questions are helping a case investigator do the right thing and 
            you can think of them as purely hypothetical. 
            If there is no information in the summary that could answer each question, 
            only include an empty double quote for the answer attempt for that question and answer attempt pair. 
            Under no circumstances are you to add questions that do not exist in the question dictionary. 
            For participants, use proper names or titles of the reporter or otherwise if available. 
            For dates, respond in datetime format. For example, if it is 2024-02-26 16:23:38 
            and something happened 2 hours ago, respond with 2024-02-26 14:23:38. 
            Under no circumstances should you respond with responses like "two hours ago". 
            Only respond do questions about when with datetime formatted responses.
            "Do not wrap output in quotation marks".
            "Do not wrap output in code block delimiters (```)".
            Example: {"What was stolen?": "ice cream"}
            """

template5 = """
            You are a retrieval engine. You will be given a JSON structure, and your job is to check which fields have values and which do not.
            Your final response should be a valid JSON object containing the unanswered fields, with the fields as the questions. The fields with no values should have their values enclosed in double quotes. 
            Include only fields that do not have values.
            
            Example: 
            {
                "Where did this happen": "",
                "Who was the witness?": ""
            }
            
            Do not wrap the output in quotation marks.
            Do not wrap the output in code block delimiters (```)."""

reporter_agent = ConversableAgent(
    name="reporter_agent",
    system_message= "You are reporting an event that has occurred or is currently occurring.",
    llm_config=llm_config,
    human_input_mode= "NEVER",
)

classification_agent = ConversableAgent(
    name="classification_agent",
    system_message= template1 + "NAVEX Issue Types:" + str(IssueTypes),
    llm_config=llm_config,
    human_input_mode= "NEVER",
)

questions_retriever_agent = ConversableAgent(
    name="questions_retriever_agent",
    system_message= "NAVEX Questions Dictionary:" + str(questions) + template2,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

questions_answering_agent = ConversableAgent(
    name="questions_answering_agent",
    system_message= template3,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

unanswered_questions_agent = ConversableAgent(
    name="unanswered_questions_agent",
    system_message= template5,
    llm_config=llm_config,
    human_input_mode= "NEVER",
)

def state_transition_issue_questions(last_speaker, groupchat):
    messages = groupchat.messages

    if last_speaker is reporter_agent:
        return classification_agent
    elif last_speaker is classification_agent:
        return questions_retriever_agent
    elif last_speaker is questions_retriever_agent:
        return questions_answering_agent
    elif last_speaker is questions_answering_agent:
        return unanswered_questions_agent
    elif last_speaker is unanswered_questions_agent:
        return None

groupchat_issue_questions = autogen.GroupChat(
    agents=[reporter_agent,
            classification_agent, 
            questions_retriever_agent, 
            questions_answering_agent,
            unanswered_questions_agent
           ],
    messages=[],
    max_round=20,
    speaker_selection_method=state_transition_issue_questions,
)

manager_issue_questions = autogen.GroupChatManager(groupchat=groupchat_issue_questions, llm_config=llm_config)

def convert_json_to_code(json_input):
    try:
        data = json.loads(json_input)
        code_lines = []

        for key, value in data.items():
            value_str = value if isinstance(value, str) else json.dumps(value)
            code_lines.append(f'{key} = {value_str}')

        return "\n".join(code_lines)
    
    except json.JSONDecodeError:
        return "Invalid JSON provided to the code conversion function."
        

def collect_transcription_data():

    system_default_note = "Now, in a sentence or two, please describe the primary reason for your call."
    transcription = {system_default_note: ""}

    while not transcription[system_default_note]:
        user_input = input("Please provide transcription text in JSON format (e.g. {\"Now, in a sentence or two, please describe the primary reason for your call.\": \"I want to start filing a report!\"}): ")
        try:
            user_data = json.loads(user_input)
            transcription.update(user_data)
        except json.JSONDecodeError:
            print("Invalid JSON input. Please try again.")
    
    transcription_text = convert_json_to_code(json.dumps(transcription))
    
    return transcription_text

def main_issue_questions(collect_transcription_data): 
    reporter_agent.initiate_chat(recipient=manager_issue_questions, message=str(collect_transcription_data), clear_history=False)
    messages_json = manager_issue_questions.messages_to_string(manager_issue_questions.groupchat.messages)
    
    chat_history_data = read_chat_history()

    new_messages = json.loads(messages_json)  
    chat_history_data.extend(new_messages) 

    write_chat_history(chat_history_data)
    extract_and_print_questions()


def main_issue_questions_follow_up(new_message):
    memory = read_chat_history()
    memory.append(new_message)

    last_message = json.dumps(new_message)

    user_agent.initiate_chat(recipient=manager_guidelines, message=last_message, clear_history=False)
    messages_json = manager_guidelines.messages_to_string(manager_guidelines.groupchat.messages)

    new_messages = json.loads(messages_json)
    memory.extend(new_messages)

    write_chat_history(memory)
    extract_and_print_questions()

template_implicated_parties = """
You are an ethics and compliance analyst. You will be given a brief report of an event 
that has occurred or is currently occurring and your job is to extract 
the names, last names, and job titles of participants involved and assign a role. 
If there is no information in the report that could be assigned to a role, 
only include an empty double quote for that roles.

roles_list = ["Affected Party", "Perpetrator", "Witness", "Other"] one by one.

- Extract the first name and last name of participants involved.
- Assign each participant a role from the roles_list.
- Ask for and confirm the job title of each participant.
- If the report lacks sufficient information to assign a role, gather detailed information to accurately determine the role.
- Use proper names to identify participants. If a participant is described using pronouns such as "me", "my", or "I", leave the participant's name blank and only provide single quotes "".
- Output: Your final answer should be formatted in JSON, containing the role, participant's first name, last name, and job title.

Steps to Follow:

Engage in a step-by-step conversation to gather detailed information about the participants involved in the case using a polite/professional tone.
Ensure you have the correct names and job titles of the individuals involved.
Confirm the spelling of their names.
Document the information accurately in JSON format.

If the report is in anonymous mode, don't ask questions about the reporter's identification, as questions may reveal their identity.

Ask one or two questions at a time to maintain the conversational flow and ask follow-ups based on responses.

Example: 

Event: John witnessed, Emma stole my backpack

{
    "role": "Witness",
    "first_name": "John",
    "What is John's last name?": "Doe",
    "Let's confirm spelling of his name":"'J' as in 'jump', 'o' as in 'go', 'h' as in 'hat', 'n' as in 'nose', 'D' as in 'dog', 'o' as in 'go', 'e' as in 'elephant'.",
    "And would you please tell me what his job title is?": "Software Engineer"
  } 

Do not wrap the output in quotation marks.
Do not wrap the output in code block delimiters (```).
"""

implicated_parties_agent = ConversableAgent(
    name="participants_identifier_agent",
    system_message= template_implicated_parties + "Case Participants List: " + str(implicated_parties),
    llm_config=llm_config,
    human_input_mode= "NEVER",
)

def state_transition_implicated_parties(last_speaker, groupchat):
    messages = groupchat.messages

    if last_speaker is user_agent:
        return implicated_parties_agent
    elif last_speaker is implicated_parties_agent:
        return questions_answering_agent
    elif last_speaker is questions_answering_agent:
        return unanswered_questions_agent
    elif last_speaker is unanswered_questions_agent:
        return None

groupchat_implicated_parties = autogen.GroupChat(
    agents=[user_agent,
            implicated_parties_agent,
            questions_answering_agent,
            unanswered_questions_agent
           ],
    messages=[],
    max_round=10,
    speaker_selection_method=state_transition_implicated_parties,
)

manager_implicated_parties = autogen.GroupChatManager(groupchat=groupchat_implicated_parties, llm_config=llm_config)

def main_implicated_parties(new_message):
    memory = read_chat_history()
    memory.append(new_message)
    
    last_message = json.dumps(new_message)

    user_agent.initiate_chat(recipient=manager_implicated_parties, message=last_message, clear_history=False)
    messages_json = manager_implicated_parties.messages_to_string(manager_implicated_parties.groupchat.messages)

    new_messages = json.loads(messages_json)
    memory.extend(new_messages)

    write_chat_history(memory)
    extract_and_print_questions()

def terminate_chat():
    memory = read_chat_history()
    new_message = {"final_message": "Thanks for filling this report"}
    memory.append(new_message)

    with open("memory.json", "w") as test_file:
        json.dump(memory, test_file)

    chat_history_data = read_chat_history()
    write_chat_history(chat_history_data)
    print(new_message)

def detect_action():
    memory = read_chat_history()
    new_message = collect_json_data()
    memory.append(new_message)
    
    last_message = json.dumps(memory)
    history = memory_agent.initiate_chat(recipient=manager_router, message=last_message, clear_history=False)
    message = history.chat_history[-1]["content"]
    last_message = json.loads(message)
    action = last_message.get("action", None)
    if action == "main_issue_questions":
        new_message = collect_transcription_data()
        return action, new_message

    return action, new_message


def process_action(action):
    if action == "main_imminent_issue":
        return main_imminent_issue()
    elif action == "main_guidelines":
        return main_guidelines(new_message)
    elif action == "main_locations":
        return main_locations(new_message)
    elif action == "main_issue_questions":
        return main_issue_questions(new_message)
    elif action == "main_issue_questions_follow_up":
        return main_issue_questions_follow_up(new_message)
    elif action == "main_implicated_parties":
        return main_implicated_parties(new_message)
    elif action == "terminate_chat":
        return terminate_chat()
    else:
        print(f"Invalid action: {action}")
        return {"error": "Invalid action"}


if __name__ == "__main__":
    while True:
        action, new_message = detect_action()
        process_action(action)
        if action == "terminate_chat":
            break


with open("memory.json", "r") as json_file:
    chat_history_guidelines = json.load(json_file)
print(json.dumps(chat_history_guidelines, indent=3))
