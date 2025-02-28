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

clientKey = """DNIS"""

guidelines ="""
{
  "CUSTOMER": "DNIS",
  "OPENING": "Thank you for contacting the call center. This is AICCI, the AI-powered representative. Would you please tell me if you are calling for a new report or follow-up?",
  "Other_Instructions": {
    "new_report": {
      "instruction": "Follow other instructions for new reports",
      "READ": {
        "READ_1": "Before starting this report, please understand this process may take 10-15 minutes. Would you like to proceed?",
        "READ_2": ""
      },
      "DONT_READ": {
        "DONT_READ_1": "",
        "DONT_READ_2": ""
      },
      "Affiliation": {
        "instruction": "Check for direct affiliation with DNIS",
        "READ": {
          "READ_1": "Please confirm if you are directly affiliated with DNIS.",
          "READ_2": "If you are not directly affiliated, please provide the name of the organization you are representing."
        },
        "DONT_READ": {
          "DONT_READ_1": "",
          "DONT_READ_2": ""
        }
      }
    },
    "follow_up": {
      "instruction": "Follow other instructions for follow-ups",
      "READ": {
        "READ_1": "Would you please provide me with the follow-up key that we have prepared for you?",
        "READ_2": ""
      },
      "DONT_READ": {
        "DONT_READ_1": "",
        "DONT_READ_2": ""
      }
    }
  }
}
"""

locations = """Response from /v6.0/Intake/locations: {"clientKey": "Call Center Training 32 - Target Corporation", "Locations": [{"Case_CompanyLocation": "Branch Office D", "Case_CompanyCity": "Austin", "Case_CompanyState": "TX", "Case_CompanyZip": "73301", "Case_CompanyCountry": "USA", "Case_LocationCustomField1": "Technical Support", "Case_LocationCustomField2": "SW", "Case_LocationCustomField3": "District 7", "Case_LocationCustomField4": "Floor 5"}, {"Case_CompanyLocation": "Branch Office E", "Case_CompanyCity": "Denver", "Case_CompanyState": "CO", "Case_CompanyZip": "80201", "Case_CompanyCountry": "USA", "Case_LocationCustomField1": "Sales", "Case_LocationCustomField2": "MW", "Case_LocationCustomField3": "District 4", "Case_LocationCustomField4": "Building 1"}, {"Case_CompanyLocation": "Branch Office F", "Case_CompanyCity": "Orlando", "Case_CompanyState": "FL", "Case_CompanyZip": "32801", "Case_CompanyCountry": "USA", "Case_LocationCustomField1": "Customer Service", "Case_LocationCustomField2": "SE", "Case_LocationCustomField3": "District 2", "Case_LocationCustomField4": "Suite 300"}]}"""

IssueTypes = """Response from /v6.0/Intake/GetMobileIssueTypesWithDefaults: "{\"clientKey\": \"Call Center Training 32 - Target Corporation\", \"data\": {\"ViolationTypeList\": [{\"Description\": \"Unauthorized access to company systems or data, hacking, or other cyber threats. This includes phishing attempts, malware infections, and other cybersecurity incidents.\", \"Name\": \"Cybersecurity Breach\", \"ViolationTypeId\": 13}, {\"Description\": \"Violations related to environmental regulations, including improper disposal of hazardous materials, pollution, and non-compliance with environmental laws.\", \"Name\": \"Environmental Violation\", \"ViolationTypeId\": 14}, {\"Description\": \"Reports of physical violence or threats of violence in the workplace. This includes fights, assaults, and other forms of physical aggression.\", \"Name\": \"Workplace Violence\", \"ViolationTypeId\": 15}, {\"Description\": \"Instances of theft, embezzlement, or other forms of financial fraud. This includes stealing company property, misappropriating funds, and falsifying financial records.\", \"Name\": \"Financial Fraud\", \"ViolationTypeId\": 16}, {\"Description\": \"Violations related to health and safety regulations, including unsafe working conditions, lack of proper safety equipment, and failure to follow safety protocols.\", \"Name\": \"Health and Safety Violation\", \"ViolationTypeId\": 17}, {\"Description\": \"Reports of discrimination based on age, including unfair treatment, harassment, or bias against employees or customers due to their age.\", \"Name\": \"Racial Discrimination\", \"ViolationTypeId\": 18}, {\"Description\": \"Instances of unethical behavior or conflicts of interest, such as accepting gifts or favors in exchange for preferential treatment or using company resources for personal gain.\", \"Name\": \"Ethical Violation\", \"ViolationTypeId\": 19}, {\"Description\": \"Reports of violations related to data privacy, including unauthorized sharing of personal information, data breaches, and failure to comply with data protection regulations.\", \"Name\": \"Data Privacy Violation\", \"ViolationTypeId\": 20}, {\"Description\": \"Instances of workplace bullying, including repeated mistreatment, verbal abuse, and other forms of hostile behavior that create a toxic work environment.\", \"Name\": \"Workplace Bullying\", \"ViolationTypeId\": 21}, {\"Description\": \"Reports of non-compliance with legal or regulatory requirements, including violations of industry standards, government regulations, and company policies.\", \"Name\": \"Regulatory Non-Compliance\", \"ViolationTypeId\": 22}]}}"""

questions = """Response from /v6.0/ViolationQuestion/GetViolationQuestionAndAnswersForPlatformPackages: "{\"clientKey\": \"Call Center Training 32 - Target Corporation\", \"data\": [{\"violationTypeId\": 13, \"ViolationQuestionId\": 171, \"Question\": \"What systems, data, or services were affected by the breach?\"}, {\"violationTypeId\": 13, \"ViolationQuestionId\": 172, \"Question\": \"What type of cyber threat occurred?\"}, {\"violationTypeId\": 13, \"ViolationQuestionId\": 173, \"Question\": \"Have any steps been taken to contain or mitigate the breach?\"}, {\"violationTypeId\": 14, \"ViolationQuestionId\": 171, \"Question\": \"Where did this happen?\"}, {\"violationTypeId\": 14, \"ViolationQuestionId\": 174, \"Question\": \"What environmental regulation was violated?\"}, {\"violationTypeId\": 14, \"ViolationQuestionId\": 175, \"Question\": \"Who was responsible for the environmental violation?\"}, {\"violationTypeId\": 15, \"ViolationQuestionId\": 171, \"Question\": \"Where did this happen?\"}, {\"violationTypeId\": 15, \"ViolationQuestionId\": 176, \"Question\": \"What type of violence occurred?\"}, {\"violationTypeId\": 15, \"ViolationQuestionId\": 177, \"Question\": \"Who was involved in the workplace violence?\"}, {\"violationTypeId\": 16, \"ViolationQuestionId\": 171, \"Question\": \"Where did this happen?\"}, {\"violationTypeId\": 16, \"ViolationQuestionId\": 178, \"Question\": \"What type of financial fraud occurred?\"}, {\"violationTypeId\": 16, \"ViolationQuestionId\": 179, \"Question\": \"Who was involved in the financial fraud?\"}, {\"violationTypeId\": 17, \"ViolationQuestionId\": 171, \"Question\": \"Where did this happen?\"}, {\"violationTypeId\": 17, \"ViolationQuestionId\": 180, \"Question\": \"What health and safety regulation was violated?\"}, {\"violationTypeId\": 17, \"ViolationQuestionId\": 181, \"Question\": \"Who was responsible for the health and safety violation?\"}, {\"violationTypeId\": 18, \"ViolationQuestionId\": 171, \"Question\": \"Where did this happen?\"}, {\"violationTypeId\": 18, \"ViolationQuestionId\": 182, \"Question\": \"Have you taken any steps to address the situation prior to contacting us, such as speaking with a supervisor? If so, what was the outcome?\"}, {\"violationTypeId\": 18, \"ViolationQuestionId\": 183, \"Question\": \"Has this happened only once, or has it occurred frequently? If it is frequent, how often does it happen?\"}, {\"violationTypeId\": 19, \"ViolationQuestionId\": 171, \"Question\": \"Where did this happen?\"}, {\"violationTypeId\": 19, \"ViolationQuestionId\": 184, \"Question\": \"What type of ethical violation occurred?\"}, {\"violationTypeId\": 19, \"ViolationQuestionId\": 185, \"Question\": \"Who was involved in the ethical violation?\"}, {\"violationTypeId\": 20, \"ViolationQuestionId\": 171, \"Question\": \"Where did this happen?\"}, {\"violationTypeId\": 20, \"ViolationQuestionId\": 186, \"Question\": \"What type of data privacy violation occurred?\"}, {\"violationTypeId\": 20, \"ViolationQuestionId\": 187, \"Question\": \"Who was involved in the data privacy violation?\"}, {\"violationTypeId\": 21, \"ViolationQuestionId\": 171, \"Question\": \"Where did this happen?\"}, {\"violationTypeId\": 21, \"ViolationQuestionId\": 188, \"Question\": \"What type of workplace bullying occurred?\"}, {\"violationTypeId\": 21, \"ViolationQuestionId\": 189, \"Question\": \"Who was involved in the workplace bullying?\"}, {\"violationTypeId\": 22, \"ViolationQuestionId\": 171, \"Question\": \"Where did this happen?\"}, {\"violationTypeId\": 22, \"ViolationQuestionId\": 190, \"Question\": \"What type of regulatory non-compliance occurred?\"}, {\"violationTypeId\": 22, \"ViolationQuestionId\": 191, \"Question\": \"Who was involved in the regulatory non-compliance?\"}]}"""

implicated_parties = """{"Affected Party":"","Perpetrator":"","Witness":"","Other":""}"""

#############################################################################################################

template_memory = """
Role: You are the JSON Chat Memory Agent.
Objective: Your primary responsibility is to memorize the new message and the {{JSON Chat Memory}}.
You will take to action!

Do not wrap the output in quotation marks.
Do not wrap the output in code block delimiters (```).
"""

template_router = """
Role: You are a classification agent responsible for managing the routing of new messages based on predefined rules and priorities.
Objective: You will analyze the incoming message, evaluate the associated {{JSON Chat Memory}}, consider the relevant {{group_chat}} priorities, 
and apply the applicable {{termination_requirements}}. Your task is to:
Detect the correct action ID: Based on the new message, refer to the {{JSON Chat Memory}} to determine the appropriate action ID that aligns with the {{termination_requirements}}.
Route the message to the appropriate group: After determining the action id, select the next appropriate {{group_chat}} for the message from the available options.

{
  "group_chat": 
  [
    {"id": "main_imminent_issue",
    "priority": 1, 
    "role": "This function is used when application detects imminent issue",
    "termination_requirements": {
        "requirement_1": "The presence of an imminent issue or event in the context.",
      }
    },
    {"id": "main_guidelines",
    "priority": 2, 
    "role": "This function is used when the JSON Chat Memory does not contain the customer's responses to the READ messages in guidelines/instructions.",
    "termination_requirements": {
        "requirement_1": "Have answers to all associated READ messages.",
        "requirement_2": "Have identified Affiliation of the caller.",
        "requirement_3": "Have followed instructions for DONT READ messages"
      }
    },
    {"id": "main_locations",
    "priority": 3, 
    "role": "This function is used when the JSON Chat Memory does not contain the customer responses to the location indentification requirements.",
    "termination_requirements": {
        "requirement_1": "Respond to all location-related inquiries", 
        "requirement_2": "Identify the State, and City",
        "requirement_4": "Identify building name and address.",
        "requirement_5": "Identify whether the event's location is onsite or offsite."
      }
    },
    {"id": "main_checker_point",
    "priority": 4, 
    "role": "This function is used when the JSON Chat Memory does not contain the responses to report_mode whether the report is anonymous or not.",
    "termination_requirements": {
        "requirement_1": "The status of report_mode is being collected and stored in memory.",
        "requirement_2": "The status of file_type is being collected and stored in memory.",
        "requirement_3": "The status of location_type is being collected and stored in memory."
      }
    },
    {"id": "main_issue_questions",
    "priority": 5, 
    "role": "This function is used when input message is the transcription of the event.",
    "termination_requirements": {
        "requirement_1": "transcription of the report which is the primary reason for the call is in memory"
      }
    },
    {"id": "main_issue_questions_follow_up",
    "priority": 6, 
    "role": "This function is used when the JSON Chat Memory does not contain the customer responses to the issue questions",
    "termination_requirements": {
        "requirement_1": "Answer to all retrieved violation type questions."
      }
    },
    {"id": "main_implicated_parties",
    "priority": 7, 
    "role": "This function is used when the JSON Chat Memory does not contain the customer responses to the issue questions",
    "termination_requirements": {
        "requirement_1": "Information of implicated parties including their names",
        "requirement_2": "Information of implicated parties including their roles",
        "requirement_3": "Information of implicated parties including their job titles"
      }
    },
    {"id": "main_final_allegation",
    "priority": 8, 
    "role": "This function is used when the JSON Chat Memory does not contain a reevaluation of the ViolationTypeId after the initial one.",
    "termination_requirements": {
        "requirement_1": "Information of the presence of ViolationTypeId for the second time in memory after implicated parties",
      }
    },
    {"id": "main_follow_up",
    "priority": 9, 
    "role": "This function is used when the JSON Chat Memory does not contain additional follow up questions",
    "termination_requirements": {
        "requirement_1": "Information of the presence of follow up questions in memory after detection of the final allegation",
      }
    },
    {"id": "main_report_review",
    "priority": 10, 
    "role": "This function is used when the JSON Chat Memory does not contain the report summary read by the reporter.",
    "termination_requirements": {
        "requirement_1": "Information of reviewing the summary of the report.",
      }
    },  
    {"id": "terminate_chat",
    "priority": 11, 
    "role": "This function is used when the JSON Chat Memory does contain the customer responses to the previous priorities.",
    "termination_requirements": {
        "requirement_1": "Answer to all questions"
      }
    }
  ]
}

If the {{termination_requirements}} are satisfied, your detected action will be terminate_chat and you are Done!.

example: 

{"action":"main_issue_questions"}

Do not wrap the output in quotation marks.
Do not wrap the output in code block delimiters (```).
"""

template_imminent_issue = """
Role: Ethics and Compliance Contact Center Agent 
Objective: You will be given the callers conversation records. Your primary responsibility is to return a "Final Message" in certain situations:

At any point during the interview, if imminent issue detect, flag report with priority with this message: {"Final Message":"Flag report with priority"}
At any point during the interview, If suicide ideations deteted, transfer call to CS workflow with this message: {"Final Message":"transfer call to CS workflow"}

Situation Examples:
Threats of Violence: Reports of threats of physical harm or violence from coworkers, supervisors, or customers.
Harassment or Assault: Incidents of sexual harassment, physical assault, or severe bullying.
Medical Emergencies: Situations involving severe health issues such as heart attacks, severe allergic reactions, or injuries.
Unsafe Working Conditions: Immediate dangers due to unsafe conditions like hazardous materials, lack of safety equipment, or structural hazards.
Fire or Explosion: Reports of fires, explosions, or similar emergencies posing immediate threats to safety.
Active Shooter or Hostage Situation: Incidents involving an active shooter or hostage situation within the workplace.
Severe Psychological Distress: Cases of severe psychological distress or suicidal thoughts/intentions.
(If suicide ideations deteted, transfer call to CS workflow) with this message: {"Final Message":"transfer call to CS workflow"}

example: 
{"Final Message":"transfer call to CS workflow"} For a scenario like: {"Tell me what has happened": "Jack Jones has a gun in the workplace."}

Do not wrap the output in quotation marks.
Do not wrap the output in code block delimiters (```).
"""



########################################### ADD SESSION MEMORY ###############################################

with open("memory.json", "w") as json_file:
    json.dump([{}], json_file)

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
Role: Assistant to extract text in JSON.
Objective: Extract text in JSON from text or HTML elements similar to the following format in a valid JSON. 
You may have one, none, or multiple READ and DONT_READ messages:

{
  "CUSTOMER": "DNIS",
  "OPENING": "Text from opening HTML element",
  "READ": {
    "READ_1": "1st Text to read",
    "READ_2": "2nd Text to read"
  },
  "DONT_READ": {
    "DONT_READ_1": "1st Text not to read",
    "DONT_READ_2": "2nd Text not to read"
  },
  "Other_Instructions": {
    "new_report": {
      "instruction": "Follow other instructions for new reports",
      "READ": {
        "READ_1": "1st Text to read",
        "READ_2": "2nd Text to read"
      },
      "DONT_READ": {
        "DONT_READ_1": "1st Text not to read",
        "DONT_READ_2": "2nd Text not to read"
      },
      "Affiliation": {
        "instruction": "Check for direct affiliation with DNIS",
        "DONT_READ": {
          "DONT_READ_1": "1st Text not to read",
          "DONT_READ_2": "2nd Text not to read"
        },
        "READ": {
          "READ_1": "1st Text to read",
          "READ_2": "2nd Text to read"
        }
      }
    },
    "follow_up": {
      "instruction": "Follow other instructions for follow-ups",
      "DONT_READ": {
        "DONT_READ_1": "1st Text not to read",
        "DONT_READ_2": "2nd Text not to read"
      },
      "READ": {
        "READ_1": "1st Text to read",
        "READ_2": "2nd Text to read"
      }
    }
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
Role: Ethics and Compliance Contact Center Analyst
Objective: Handle calls by following a structured process using provided guidelines and previous chat history.

Steps:

Start with the "OPENING" statement from the guidelines.
Sequentially address each "READ" message: 
a. If not addressed, respond with the "READ" message. 
b. If addressed, move to the next "READ" message.
Determine if the call is a "Follow-Up" or "New Report": 
a. For "Follow-Up", follow "Other_Instructions" if available and respond with the "READ" message is "affiliation"  
b. For "New Report", follow "Other_Instructions" if available.
Continue addressing each "READ" message in sequence.

Example 1: New Report

"OPENING": "Text from opening HTML element"
"READ_1": "1st Text to read"
"READ_2": "2nd Text to read"

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
    system_message= f"guidelines: {guidelines} + {template_guidelines}",
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
    max_round=10,
    speaker_selection_method=state_transition_guidelines,
)

manager_guidelines = autogen.GroupChatManager(groupchat=groupchat_guidelines, llm_config=llm_config)

def collect_json_data():
    follow_ups = {"questions": []}
    while not follow_ups["questions"]:
        user_input = input('Please provide data in JSON format (e.g. {"questions": [{"question": "your question", "answer": "your answer"}]}): ')
        try:
            user_data = json.loads(user_input)
            if isinstance(user_data, dict) and "questions" in user_data and isinstance(user_data["questions"], list):
                valid = all(isinstance(item, dict) and "question" in item and "answer" in item for item in user_data["questions"])
                if valid:
                    follow_ups.update(user_data)
                else:
                    print("Each item in the questions list must be an object with question and answer keys. Please try again.")
            else:
                print("The JSON input must be an object with a questions key containing a list of question-answer objects. Please try again.")
        except json.JSONDecodeError:
            print("Invalid JSON input. Please try again.")
    return follow_ups


def extract_questions_from_memory(memory):
    last_message = memory[-1]
    content = last_message["content"]
    try:
        content_dict = json.loads(content)
        if "questions" in content_dict and isinstance(content_dict["questions"], list):
            return content_dict
        else:
            raise json.JSONDecodeError("Invalid format", content, 0)
    except json.JSONDecodeError:
        questions = [q.strip() + "?" if not q.strip().endswith("?") else q.strip() for q in content.split("\n") if q.strip()]
        questions_dict = {"questions": [{"question": q, "answer": ""} for q in questions]}
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
        json.dump(unique_messages, json_file, ensure_ascii=False)
        

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


template_report_type_checker= """
Role: classsification agent to determine the call reason.
Objective: determine the call reason, whether it is a "Follow-Up" or a "New Report" in a valid JSON.

Example1:
{"report_type":"New Report"}

Example2:
{"report_type":"Follow-Up"}

Do not wrap the output in quotation marks.
Do not wrap the output in code block delimiters (```).
"""

report_type_checker_agent = ConversableAgent(
    name="report_type_checker_agent",
    system_message= template_report_type_checker,
    llm_config=llm_config,
    human_input_mode= "NEVER",
)

def state_transition_report_type_checker(last_speaker, groupchat):
    messages = groupchat.messages

    if last_speaker is user_agent:
        return report_type_checker_agent
    elif last_speaker is report_type_checker_agent:
        return None

groupchat_report_type_checker = autogen.GroupChat(
    agents=[user_agent,
            report_type_checker_agent
           ],
    messages=[],
    max_round=10,
    speaker_selection_method=state_transition_report_type_checker,
)

manager_report_type_checker = autogen.GroupChatManager(groupchat=groupchat_report_type_checker, llm_config=llm_config)

def main_report_type_checker():
    memory = read_chat_history()
    last_message = json.dumps(memory)

    user_agent.initiate_chat(recipient=manager_report_type_checker, message=last_message, clear_history=False)
    messages_json = manager_report_type_checker.messages_to_string(manager_report_type_checker.groupchat.messages)

    new_messages = json.loads(messages_json)
    report_type_checker = new_messages[-1]
    memory.append(report_type_checker)
    
    write_chat_history(memory)


template_locations = """
Role: Ethics and Compliance Contact Center Analyst for finding locations. 
Objective: Retrieve the location of the event from [locations] by asking a few questions one by one about the state, city, etc., using [locations] data. 
If the location is not available in the data, it is likely an offsite location. You will be given a series of sequential steps. 
Your task is to complete each step in order, one at a time, and move to the next step only after the current one is completed successfully. 
Begin execution from first Step. Once first step is successfully completed, proceed to the second step, and so on, until all steps are complete.

Steps:
- Ask for State and City
- Ask for Building Address and Name of Building and chech if "Case_CompanyLocation", Case_CompanyCity", "Case_CompanyState" is available.
- Identify whether the event has happened at the company's onsite location or somewhere offsite. Your final question would be to confirm the location of the event.

Examples:
"Could you please provide the state and city where the event took place?"
[Wait for the response and proceed to Step 2]
Could you now provide the building address and the name of the building where the event occurred?"
[Wait for the response and proceed to Step 3]
"Based on the information provided, I need to confirm whether the event happened at the company's onsite location or somewhere offsite. 
Could you please confirm if the event took place onsite at [Building Name], [City], [State]?"

[If the location is offsite, modify the question accordingly: "Could you please confirm if the event took place offsite at [Building Name/Location], [City], [State]?"]

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


template_location_checker= """
Role: classsification agent to determine the location of the event.
Objective: determine the location of the event, whether it is a "onsite" or "offsite", 
and map it to the available locations data in a valid JSON.

Example 1:

{
  "location_type": "onsite",
  "Case_CompanyLocation": "Branch Office C",
  "Case_CompanyCity": "Albany",
  "Case_CompanyState": "NY",
  "Case_CompanyZip": "12207",
  "Case_CompanyCountry": "USA",
  "Case_LocationCustomField1": "Customer Service",
  "Case_LocationCustomField2": "SE",
  "Case_LocationCustomField3": "District 3",
  "Case_LocationCustomField4": "Suite 200"
}

Example 2:

{
  "location": "offsite",
  "Case_CompanyLocation": "HG's Restaurant",
  "Case_CompanyCity": "Houston",
  "Case_CompanyState": "TX"
}

Do not wrap the output in quotation marks.
Do not wrap the output in code block delimiters (```).
"""

location_checker_agent = ConversableAgent(
    name="location_checker_agent",
    system_message= f"locations: {locations} + {template_location_checker}",
    llm_config=llm_config,
    human_input_mode= "NEVER",
)

def state_transition_location_checker(last_speaker, groupchat):
    messages = groupchat.messages

    if last_speaker is user_agent:
        return location_checker_agent
    elif last_speaker is location_checker_agent:
        return None

groupchat_location_checker = autogen.GroupChat(
    agents=[user_agent,
            location_checker_agent
           ],
    messages=[],
    max_round=10,
    speaker_selection_method=state_transition_location_checker,
)

manager_location_checker = autogen.GroupChatManager(groupchat=groupchat_location_checker, llm_config=llm_config)

def main_location_checker():
    memory = read_chat_history()
    last_message = json.dumps(memory)

    user_agent.initiate_chat(recipient=manager_location_checker, message=last_message, clear_history=False)
    messages_json = manager_location_checker.messages_to_string(manager_location_checker.groupchat.messages)

    new_messages = json.loads(messages_json)
    location_checker = new_messages[-1]
    memory.append(location_checker)
    
    write_chat_history(memory)


template_report_mode = """
Role:Classification agnet to detect the report mode
Objective: Detect the report mode based on the answer you receive. Your answer should be 
a valid JSON with "Report Mode" as the key and the detected report mode, either "Anonymous" or "Non-Anonymous," as the value.

Example:
1. {"report-mode":"Anonymous"}
2. {"report-mode":"Non-Anonymous"}

Do not wrap the output in quotation marks.
Do not wrap the output in code block delimiters (```).
"""

template_anonymous_mode = """You are reviewing reports submitted by employees at a company who wish to submit their report anonymously. Please read through the report and
only remove any of the person writing the reports personal information, such as their phone number, email address, name, location (such as
office number or desk/cubicle/work location which would identify the reporter), race, job title, length of employment, disabilities or gender.
Information about people they witnessed should remain. The report should retain as much of the original text as possible without giving identifying
details about the person reporting it.

Example1: "I am the only female/engineer in the office/location" should be removed.
Example2: "The colleague from Marketing, who often sits two desks away from me" should be changed to "The colleague from Marketing".
Do not wrap the output in quotation marks.
Do not wrap the output in code block delimiters (```).
"""

anonymous_mode_agent = ConversableAgent(
    name="anonymous_mode",
    system_message= template_report_mode,
    llm_config=llm_config,
    human_input_mode= "NEVER",
)


def state_transition_anonymous_mode(last_speaker, groupchat):
    messages = groupchat.messages

    if last_speaker is user_agent:
        return anonymous_mode_agent
    elif last_speaker is anonymous_mode_agent:
        return None
groupchat_anonymous_mode = autogen.GroupChat(
    agents=[user_agent,
            anonymous_mode_agent
           ],
    messages=[],
    max_round=10,
    speaker_selection_method=state_transition_anonymous_mode,
)

manager_anonymous_mode = autogen.GroupChatManager(groupchat=groupchat_anonymous_mode, llm_config=llm_config)


def collect_anonymous_mode():
    system_default_note_1 = "Do you wish to provide your name today, or would you rather remain anonymous?"
    system_default_note_2 = "Okay. And what is your relationship to your company?"
    anonymous_mode = {"questions": [
        {"question": system_default_note_1, "answer": ""},
        {"question": system_default_note_2, "answer": ""}
    ]}

    while any(item["answer"] == "" for item in anonymous_mode["questions"]):
        print(json.dumps(anonymous_mode))
        user_input = input('Please provide your answers in JSON format (e.g. {"questions": [{"question": "your question", "answer": "your answer"}]}): ')
        try:
            user_data = json.loads(user_input)
            if isinstance(user_data, dict) and "questions" in user_data:
                for user_item in user_data["questions"]:
                    if "question" in user_item and "answer" in user_item:
                        for item in anonymous_mode["questions"]:
                            if item["question"] == user_item["question"]:
                                item["answer"] = user_item["answer"]
                                break
                    else:
                        print("Each question object must have 'question' and 'answer' keys. Please try again.")
            else:
                print("The JSON input must be an object with a 'questions' key containing a list of question objects. Please try again.")
        except json.JSONDecodeError:
            print("Invalid JSON input. Please try again.")

    anonymous_mode_text = json.dumps(anonymous_mode)

    return anonymous_mode_text
    

def main_anonymous_mode(collect_anonymous_mode): 
    user_agent.initiate_chat(recipient=manager_anonymous_mode, message=str(collect_anonymous_mode), clear_history=False)
    messages_json = manager_anonymous_mode.messages_to_string(manager_anonymous_mode.groupchat.messages)
    
    chat_history_data = read_chat_history()

    new_messages = json.loads(messages_json)  
    chat_history_data.extend(new_messages) 

    write_chat_history(chat_history_data)


template_misfile_checker= """
Role: classsification agent to process misfile.
Objective: determine misfile and handle to process misfile in a valid JSON. determine either "missfile" or "correct_file" as the value.

How to determine misfile:
1. Check "DNIS". If the caller provides a different company name than the hotline name
3. Check "location_type". If the caller cannot confirm the location 
4. Check "location_type". If the location is not in the database
5. Check "Guidelines". If the parent company or affiliation cannot be established
6. Check "Guidelines" for RED Flags
7. If there is a misalignment between what has been reported and the type of business

What is the process to handle misfile on a hotline intake?
1. Talk to a live agent
2. Use the online website at XYZ.COM/REPORT

Example1:
{"file_type":"misfile",
 "system_message":"Talk to a live agent"}

Example2:
{"file_type":"correct_file",
 "system_message":"proceeds with taking the report"}

Do not wrap the output in quotation marks.
Do not wrap the output in code block delimiters (```).
"""

misfile_checker_agent = ConversableAgent(
    name="misfile_checker_agent",
    system_message= template_misfile_checker,
    llm_config=llm_config,
    human_input_mode= "NEVER",
)

def state_transition_misfile_checker(last_speaker, groupchat):
    messages = groupchat.messages

    if last_speaker is user_agent:
        return misfile_checker_agent
    elif last_speaker is misfile_checker_agent:
        return None

groupchat_misfile_checker = autogen.GroupChat(
    agents=[user_agent,
            misfile_checker_agent
           ],
    messages=[],
    max_round=10,
    speaker_selection_method=state_transition_misfile_checker,
)

manager_misfile_checker = autogen.GroupChatManager(groupchat=groupchat_misfile_checker, llm_config=llm_config)

def main_misfile_checker():
    memory = read_chat_history()
    last_message = json.dumps(memory)

    user_agent.initiate_chat(recipient=manager_misfile_checker, message=last_message, clear_history=False)
    messages_json = manager_misfile_checker.messages_to_string(manager_misfile_checker.groupchat.messages)

    new_messages = json.loads(messages_json)
    misfile_checker = new_messages[-1]
    memory.append(misfile_checker)
    
    write_chat_history(memory)


classification_template = """
Role: Ethics and Compliance Contact Center Analyst 
Objective: You will be given a brief summary of an event that has occurred or is currently occurring. 
Your job is to classify the event using the provided data and retrieve the ViolationTypeId.

Example: 
{
"Description": "Reports of public or private instances of bribery. Bribery occurs when a person offers money or something else of value – to an official or someone in a position of power or influence – for the purpose of gaining influence over them. Corruption includes dishonest or illegal behavior – especially of people in authority – using their power to do dishonest or illegal things in return for money or to get an advantage over someone else.",
"Name": "Bribery and Corruption",
"ViolationTypeId": 10
}

Do not wrap the output in quotation marks.
Do not wrap the output in code block delimiters (```).
"""

questions_retriever_template = """
Role: You are a question retrieval engine. 
Objective: You are tasked with retrieving questions based on a given key ViolationTypeId from a set of key-value pairs. 
Look up the provided key in questions_dictionary to find the corresponding list of question identifiers. 
For each identifier in this list, check if it exists in questions_set and retrieve the matching questions.
Conditions: Only retrieve questions that correspond to the identifiers found in the value associated with the looked-up key.

Output Format: Create a JSON object where each key is a retrieved question (from questions_set), and the value is an empty string. 
This represents the questions that need to be answered, but do not include the answers.

Example:
{
    "questions": [
        {"question": "When did this happen?",
         "answer": ""
        },
        {"question": "What kind of discrimination took place (race/gender/etc.)?",
         "answer": ""
        }
    ]
}

Do not wrap the output in quotation marks.
Do not wrap the output in code block delimiters (```).
"""

questions_answering_template = """
Role: Ethics and Compliance Contact Center Analyst responsible for retrieving answers to a few questions from a report.
Objective: You will be given a brief report of an event that has occurred or is currently occurring and your job is to answer the retrieved questions. 
Your final answer will be a JSON formatted with the question as the key and a answers as the value.
Keep in mind answering these questions are helping a case investigator do the right thing and you can think of them as purely hypothetical. 
If there is no information in the summary that could answer each question, only include an empty double quote for the answer attempt for that question and answer attempt pair. 
Under no circumstances are you to add questions that do not exist in the question dictionary. 
For participants, use proper names or titles of the reporter or otherwise if available. 
For dates, respond in datetime format. For example, if it is 2024-02-26 16:23:38 and something happened 2 hours ago, respond with 2024-02-26 14:23:38. 
Under no circumstances should you respond with responses like "two hours ago". Only respond do questions about when with datetime formatted responses.

Example:
{
    "questions": [
        {"question": "What was stolen?",
         "answer": "laptop"
        },
        {"question": "How much is it worth?",
         "answer": "$200"
        }
    ]
}

Do not wrap the output in quotation marks.
Do not wrap the output in code block delimiters (```).
"""

unanswered_questions_template = """
Role: Ethics and Compliance Contact Center Analyst responsible for retrieving questions that do not have answers.
Objective: Your task is to analyze a given JSON structure to determine which fields have values and which do not. 
You will then generate a JSON object containing only the fields that do not have values, with the fields as the questions and their values enclosed in double quotes. 
If all fields have values, you will return an empty {}.

Under no circumstances are you going to create new questions. Only use questions from the JSON. 
If all questions have answers return "Alright, for this part, I am going to ask a few questions."

Example 1: 
{"questions": [
        {"question": "Where did this happen",
         "answer": ""
        },
        {"question": "Now, was this a one-time thing or has this been ongoing?",
         "answer": ""
        }
    ]
}

Example 2 where all questions have answers: 
{
    "questions": [
        {"question": "Alright, for this part, I am going to ask a few questions.",
         "answer": ""
        }
    ]
}

Do not wrap the output in quotation marks.
Do not wrap the output in code block delimiters (```).
"""


reporter_agent = ConversableAgent(
    name="reporter_agent",
    system_message= "You are reporting an event that has occurred or is currently occurring." + "If the report mode is anonymous report:" + template_anonymous_mode + "Report Datetime: " + str(datetime.now()),
    llm_config=llm_config,
    human_input_mode= "NEVER",
)

classification_agent = ConversableAgent(
    name="classification_agent",
    system_message= classification_template + "NAVEX Issue Types:" + str(IssueTypes),
    llm_config=llm_config,
    human_input_mode= "NEVER",
)

questions_retriever_agent = ConversableAgent(
    name="questions_retriever_agent",
    system_message= "Questions Dictionary:" + str(questions) + questions_retriever_template + "If the report mode is anonymous report:" + template_anonymous_mode,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

questions_answering_agent = ConversableAgent(
    name="questions_answering_agent",
    system_message= questions_answering_template,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

unanswered_questions_agent = ConversableAgent(
    name="unanswered_questions_agent",
    system_message= unanswered_questions_template,
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
    transcription = {"questions": [{"question": system_default_note, "answer": ""}]}

    while not transcription["questions"][0]["answer"]:
        print(json.dumps(transcription))
        user_input = input('Please provide transcription text in JSON format (e.g. {"questions": [{"question": "Now, in a sentence or two, please describe the primary reason for your call.", "answer": ""}]}): ')
        try:
            user_data = json.loads(user_input)
            if isinstance(user_data, dict) and "questions" in user_data and isinstance(user_data["questions"], list):
                valid = all(isinstance(item, dict) and "question" in item and "answer" in item for item in user_data["questions"])
                if valid:
                    transcription.update(user_data)
                else:
                    print("Each item in the questions list must be an object with question and answer keys. Please try again.")
            else:
                print("The JSON input must be an object with a questions key containing a list of question-answer objects. Please try again.")
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

    user_agent.initiate_chat(recipient=manager_issue_questions, message=last_message, clear_history=False)
    messages_json = manager_issue_questions.messages_to_string(manager_issue_questions.groupchat.messages)

    new_messages = json.loads(messages_json)
    memory.extend(new_messages)

    write_chat_history(memory)
    extract_and_print_questions()


template_implicated_parties = """
Role: Ethics and Compliance Contact Center Analyst responsible for extracting the names of participants involved and assigning them a role.
Objective: You will be given a brief report of an event that has occurred or is currently occurring. Your job is to extract the names of participants involved and assign a role using only the roles listed in the "Case Participants List."

If no information in the report can be assigned to a role, provide an empty double quote ("") for that role. 
Do not infer roles beyond what is explicitly stated.

Your final answer should be in JSON format, where each key represents a role, and its value is the corresponding participant’s name. 
Always prioritize using proper names. If the report uses pronouns like "me," "my," or "I" instead of a name, leave the value as an empty string ("").

Additionally, generate conversational questions for each role in the "Case Participants List." 
Use a natural language format and ensure each question explicitly asks for the participant’s full name and role within the organization.

Formatting Examples:
Given "Case Participants List:" [witness, perpetrator]

Questions list:

{
"What is the first and last name of the perpetrator?",
"What is the perpetrator's role within the organization?",
"What is the first and last name of the witness?",
"What is the witness's role within the organization?"
}

You will ask maximum of two questions at a time.

Example JSON Output for the 1st two questions:

{
    "questions": [
        {
            "question": "What is the first and last name of the perpetrator?",
            "answer": ""
        },
        {
            "question": "What is the perpetrator's role within the organization?",
            "answer": ""
        }
    ]
}

Example JSON Output for the 2nd two questions if you have answers for the first two:

{
    "questions": [
        {
            "question": "What is the first and last name of the witness?",
            "answer": ""
        },
        {
            "question": "What is the witness's role within the organization?",
            "answer": ""
        }
    ]
}
Do not wrap the output in quotation marks.
Do not wrap the output in code block delimiters (```).
"""

implicated_parties_agent = ConversableAgent(
    name="participants_identifier_agent",
    system_message= "Case Participants List: " + str(implicated_parties) + template_implicated_parties + "If the report mode is anonymous report:" + template_anonymous_mode,
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


def state_transition_final_allegation(last_speaker, groupchat):
    messages = groupchat.messages

    if last_speaker is reporter_agent:
        return classification_agent
    elif last_speaker is classification_agent:
        return None

groupchat_final_allegation = autogen.GroupChat(
    agents=[reporter_agent,
            classification_agent
           ],
    messages=[],
    max_round=20,
    speaker_selection_method=state_transition_final_allegation,
)

manager_final_allegation = autogen.GroupChatManager(groupchat=groupchat_final_allegation, llm_config=llm_config)

def main_final_allegation():
    memory = read_chat_history()
    last_message = json.dumps(memory)

    reporter_agent.initiate_chat(recipient=manager_final_allegation, message=last_message, clear_history=False)
    messages_json = manager_final_allegation.messages_to_string(manager_final_allegation.groupchat.messages)

    new_messages = json.loads(messages_json)
    final_allegation = new_messages[-1]
    memory.append(final_allegation)
    
    write_chat_history(memory)


template_follow_up = """
Role: Ethics and Compliance Contact Center Analyst (CCA).
Objective: Check the memory and ask up to three follow-up questions so that the case investigator will be able 
to have clear event details recorded.

Example output:

{
    "questions": [
        {"question": "Okay. Thank you. All right. Now, do you know why Sue agreed to give Craig this money?",
         "answer": ""
        }
    ]
}

Do not wrap the output in quotation marks.
Do not wrap the output in code block delimiters (```).
"""

follow_up_agent = ConversableAgent(
    name="follow_up_agent",
    system_message= template_follow_up + template_anonymous_mode,
    llm_config=llm_config,
    human_input_mode= "NEVER",
)

def state_transition_follow_up(last_speaker, groupchat):
    messages = groupchat.messages

    if last_speaker is user_agent:
        return follow_up_agent
    elif last_speaker is follow_up_agent:
        return None

groupchat_follow_up = autogen.GroupChat(
    agents=[user_agent,
            follow_up_agent
           ],
    messages=[],
    max_round=10,
    speaker_selection_method=state_transition_follow_up,
)

manager_follow_up = autogen.GroupChatManager(groupchat=groupchat_follow_up, llm_config=llm_config)

def main_follow_up(new_message):
    memory = read_chat_history()
    memory.append(new_message)
    
    last_message = json.dumps(new_message)

    user_agent.initiate_chat(recipient=manager_follow_up, message=last_message, clear_history=False)
    messages_json = manager_follow_up.messages_to_string(manager_follow_up.groupchat.messages)

    new_messages = json.loads(messages_json)
    memory.extend(new_messages)

    write_chat_history(memory)
    extract_and_print_questions()

template_report_review = """
Role: Ethics and Compliance Contact Center Analyst 
Objective: Summarize the report in detail for the reporter with a professional tone. 

Summary Requirements:
Provide a detailed summary of the actual events.
Include the primary reason for the call.
Include the content of the questions and answers.
Mention when the event happened.
Mention who was involved.
Do not make up any information.
Do not mention violation types or classifications. 

Example:

{
    "questions": [
        {
            "question": "Kindly allow me a few moments to review the summary of the report for you.
                        So at 2 p.m. on December 24th, 2015, the caller overheard Craig and Sue discussing Craig's personal financial woes in Sue's room, room 216, 
                        and his need for money to purchase a home. Shortly after the conversation, the caller saw Sue heading towards Craig's office with a check. 
                        When the caller inquired about what Sue held, Sue confirmed that she was holding a Christmas gift for someone. 
                        The caller observed Sue hand Craig the check in his office. Craig embraced Sue and cried. 
                        The caller does not know if anyone else overheard Craig in Sue's conversation or witnessed Sue providing Craig the check. 
                        Now, after work, the caller saw that Craig posted on Facebook, Christmas miracles do happen, received $2,000 for my home today. 
                        Although the name of the issuer of the check was concealed, the caller believes this was the check that Sue gave Craig. Now, per company policy, 
                        employees cannot accept gifts of any amount from residents under any circumstances.",
            "answer": ""
        }
    ]
}

Do not wrap the output in quotation marks.
Do not wrap the output in code block delimiters (```).
"""


report_review_agent = ConversableAgent(
    name="report_review_agent",
    system_message= template_report_review + template_anonymous_mode,
    llm_config=llm_config,
    human_input_mode= "NEVER",
)

def state_transition_report_review(last_speaker, groupchat):
    messages = groupchat.messages

    if last_speaker is user_agent:
        return report_review_agent
    elif last_speaker is report_review_agent:
        return None

groupchat_report_review = autogen.GroupChat(
    agents=[user_agent,
            report_review_agent
           ],
    messages=[],
    max_round=10,
    speaker_selection_method=state_transition_report_review,
)

manager_report_review = autogen.GroupChatManager(groupchat=groupchat_report_review, llm_config=llm_config)

def main_report_review():
    memory = read_chat_history()
    
    user_agent.initiate_chat(recipient=manager_report_review, message=str(memory), clear_history=False)
    messages_json = manager_report_review.messages_to_string(manager_report_review.groupchat.messages)

    new_messages = json.loads(messages_json)
    final_review = new_messages[-1]
    memory.append(final_review)

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

    if action == "main_checker_point":
        new_message = collect_anonymous_mode()
        return action, new_message

    elif action == "main_issue_questions":
        new_message = collect_transcription_data()
        return action, new_message

    return action, new_message 

def process_action(action, new_message):
    if action == "main_imminent_issue":
        return main_imminent_issue()
    elif action == "main_guidelines":
        return main_guidelines(new_message)
    elif action == "main_locations":
        main_report_type_checker()
        return main_locations(new_message)
    elif action == "main_checker_point":
        main_anonymous_mode(new_message)
        main_misfile_checker()
        main_location_checker()
        new_message = collect_transcription_data()
        return main_issue_questions(new_message)
    elif action == "main_issue_questions":
        return main_issue_questions(new_message)
    elif action == "main_issue_questions_follow_up":
        return main_issue_questions_follow_up(new_message)
    elif action == "main_implicated_parties":
        return main_implicated_parties(new_message)
    elif action == "main_final_allegation":
        main_final_allegation()
        return main_follow_up(new_message)
    elif action == "main_follow_up":
        return main_follow_up(new_message)
    elif action == "main_report_review":
        return main_report_review()
    elif action == "terminate_chat":
        return terminate_chat()
    else:
        print(f"Invalid action: {action}")
        return {"error": "Invalid action"}

if __name__ == "__main__":
    while True:
        action, new_message = detect_action()
        process_action(action, new_message)
        if action == "terminate_chat":
            break



