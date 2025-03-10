# Five9_ContactCenterIntake-Approach
# Author: Kyle Zarifsadr - NOV 2024
# After installing the dependencies and setting up the .env file to access the Azure OpenAI endpoint, you can run this program locally in the terminal. The goal is to demonstrate programmatically how to send and receive JSON data between the Fivetran and NAVEX containers.
# This code requires a few modifications to function properly in a single container.

import numpy as np
import json
import pandas as pd
import os
import autogen
import sys
from datetime import datetime
from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent
from autogen import ConversableAgent
from autogen import GroupChat
from autogen import GroupChatManager
from openai import AzureOpenAI
from flask import Flask, request, jsonify

app = Flask(__name__)


load_dotenv() 

########################################################################################################

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
            Do not wrap the output in code block delimiters (```)."""

template3 = """You are an ethics and compliance analyst. You will be given a brief report of an event 
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

template4 = """You are an ethics and compliance analyst. You will be given a brief report of an event 
               that has occurred or is currently occurring and your job is to extract 
               the names of participants involved and assign a role. 
               If there is no information in the report that could be assigned to a role, 
               only include an empty double quote for that roles.
               Only use the roles "Affected Party", "Perpetrator", "Witness", and "Other". 
               Your final answer will be a JSON formatted with the role as the key and a participant as the value.
               Whenever possible, use proper names to identify participants. 
               If the description of a participant is an article such as me, my, or I, leave the participant blank and only provide single quotes "".
               "Do not wrap output in quotation marks".
               "Do not wrap output in code block delimiters (```)". 
               Example: {"Affected Party":, "Perpetrator":, "Witness":, "Other":} """

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

classification_dictionary = {
            "Substance Abuse": "Events related to impairment resulting from use of substances (drugs/alcohol - legal or illegal) impacting the workplace or violating a policy - can be on or off-duty and on- or off-premises including at company events.",
            "Discrimination": "Events of discrimination or concerns relating to accommodation requests. Discrimination generally occurs when there is a negative employment action impacting a term or condition of employment, that action is taken by the employer (which can include managers as well as others who have control over terms or conditions of work such as team leads), or the action was taken because of protected characteristic. Discrimination Events include workplace accommodation requests to adjust something relating to work linked to either a religious practice/belief or a disability. This includes allegations or Events related to religious practices or beliefs or speaks to a workplace modification or leave request linked to a medical condition or disability.",
            "Harassment": "Events of harassment that are linked to a protected characteristic (such as race, gender, sex, religion, disability, age, etc.) and includes allegations of unwelcome behavior that is offensive to a reasonable person, and is related to, or done because of, a protected characteristic.",
            "Patient Abuse": "Events of mistreatment, harm, or exploitation of a patient under the care of a healthcare professional or caregiver. This can include physical, emotional, sexual, or financial abuse, as well as neglect or abandonment. Patient abuse can occur in various settings, such as hospitals, nursing homes, or in-home care, and can have severe consequences for the victims health, well-being, and dignity.",
            "Patient Neglect": "Patient neglect refers to the failure of a healthcare professional, caregiver, or healthcare facility to provide the necessary care, attention, and support required to meet a patients basic needs, resulting in harm or potential harm to the patient. This can include physical, emotional, or medical neglect. Examples of patient neglect may involve: Failing to provide adequate food, water, or hygiene, Ignoring or not responding to a patients requests for assistance or medical attention, Failing to administer medications or treatments as prescribed, Not providing proper supervision or safety measures, leading to accidents or injuries, Failing to address or report signs of abuse or mistreatment, Not providing emotional support or social interaction necessary for the patients well-being. Patient neglect can occur in various settings, such as hospitals, nursing homes, or in-home care, and can have serious consequences for the patients health, well-being, and dignity.",
            "Theft": "Events that the organizations assets are being wasted, inappropriately used, abused, or not properly protected. This Question Class can include a wide array of assets such as property, tools, money/credit cards, facilities, company vehicles, or other tangible company assets.",
            "Time Abuse": "Events that the organizations employees are inappropriately using time while at work such as time card fraud (overstating hours to receive pay that was not actually earned).",
            "Reduction of Hours": "Events that are related to an employee having the number of hours they work reduced.",
            "Physical Violence": "Events that are related to physical violence, any intentional act of aggression or force that causes bodily harm, injury, or pain to another person. This may include behaviors such as hitting, punching, slapping, kicking, choking, pushing, or using weapons to inflict harm. Physical violence can occur between individuals, within families, or in broader societal contexts, and it often has significant negative consequences for the well-being and safety of those involved.",
            "Threats": "Events that are related to threats, expressions or indications of an intention to cause harm, damage, or negative consequences to someone or something. They can be verbal threats, written threats, or threats implied through actions or behaviors. These events can be directed towards individuals, groups, organizations, or even entire nations. They can involve physical harm, emotional or psychological harm, damage to property or reputation, or any other form of harm that can create fear, anxiety, or a sense of insecurity in the targeted person or entity.",
            "Personal Conflict of Interest": "Events about a conflict of interest, either a self-report or a report involving the behavior of others. A conflict of interest can arise in any situation where an employee’s financial or personal interest could potentially or actually interfere, or even appear to interfere, with their business judgement or the interests of the organization.",
            "Wrongful Termination": "Events about wrongful termination, also known as wrongful dismissal or wrongful discharge, refers to the illegal termination of an employees contract by an employer. This occurs when an employee is fired or let go for reasons that are not legally justified or in violation of the employees rights. Examples of wrongful termination include: Firing an employee based on their race, gender, age, religion, sexual orientation, disability, or other protected characteristics,  Terminating an employee for exercising their legal rights, such as filing a complaint about workplace harassment, discrimination, or unsafe working conditions, Firing an employee for refusing to engage in unethical practices.",
            "Workplace Civility": "Events between coworkers that are problematic and constitute unwelcome behavior but are note related to a protected class and/or are not related to violence or threatening behavior.",
            "Other": "Events not described in the other provided classifications.",
            "Accounting": "Events related to accounting, financial reporting or auditing. Examples include the unethical or improper recording and analysis of the business and financial transactions associated with generally accepted accounting practices. Examples include: misstatement of revenues, misstatement of expenses, misstatement of assets, misapplications of GAAP principles, and wrongful transactions."
}

questions_set = {
    "q1": "When did this happen?",
    "q2": "What kind of substance?",
    "q3": "What kind of discrimination took place (race/gender/etc.)?",
    "q4": "What kind of harassment took place (race/gender/etc.)?",
    "q5": "Where did this happen?",
    "q6": "What was stolen?",
    "q7": "What is the approximate value?",
    "q8": "What employee committed the time abuse?",
    "q9": "How many hours were scheduled prior?",
    "q10": "How many hours are scheduled now?",
    "q11": "Was anyone hospitalized?",
    "q12": "Describe the assault.",
    "q13": "Who communicated the threat?",
    "q14": "Describe the threat.",
    "q15": "Who has a conflict of interest?",
    "q16": "What is the nature of the conflict of interest?",
    "q17": "Why do you believe the termination was illegal?",
    "q18": "Who was involved in the incident?",
    "q19": "Did anyone else witness this?",
    "q20": "What kind of records are involved?"
    
}

questions_dictionary = {
    "Substance Abuse": ["q1", "q2"],
    "Discrimination": ["q1", "q3"],
    "Harassment": ["q1", "q4"],
    "Patient Abuse": ["q1", "q5"],
    "Patient Neglect": ["q1", "q5"],
    "Theft": ["q1", "q6", "q7"],
    "Time Abuse": ["q1", "q8"],
    "Reduction of Hours": ["q1", "q9", "q10"],
    "Physical Violence": ["q1", "q11", "q12"],
    "Threats": ["q1", "q13", "q14"],
    "Personal Conflict of Interest": ["q1", "q15", "q16"],
    "Wrongful Termination": ["q1", "q17"],
    "Workplace Civility": ["q1", "q18"],
    "Other": ["q1", "q19"],
    "Accounting": ["q1", "q20"]
}

participants_dictionary={"Affected Party":"","Perpetrator":"","Witness":"","Other":""}

########################################################################################################

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
    "temperature": 0,
    "timeout": 300,
}

########################################################################################################

user_agent = ConversableAgent(
    name="user_agent",
    system_message= "You are reporting an event that has occurred or is currently occurring.",
    llm_config=llm_config,
    human_input_mode= "NEVER",
)

classification_agent = ConversableAgent(
    name="classification_agent",
    system_message= template1 + "NAVEX Issue Types:" + str(classification_dictionary),
    llm_config=llm_config,
    human_input_mode= "NEVER",
)

questions_retriever_agent = ConversableAgent(
    name="questions_retriever_agent",
    system_message= "NAVEX Questions Dictionary:" + str(questions_dictionary) + "NAVEX Questions Set:" + str(questions_set) + template2,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

questions_answering_agent = ConversableAgent(
    name="questions_answering_agent",
    system_message= template3,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

participants_identifier_agent = ConversableAgent(
    name="participants_identifier_agent",
    system_message= template4 + "NAVEX Case Participants List: " + str(participants_dictionary),
    llm_config=llm_config,
    human_input_mode= "NEVER",
)

unanswered_questions_agent = ConversableAgent(
    name="unanswered_questions_agent",
    system_message= template5,
    llm_config=llm_config,
    human_input_mode= "NEVER",
)

########################################################################################################

def state_transition(last_speaker, groupchat):
    messages = groupchat.messages

    if last_speaker is user_agent:
        return classification_agent
    elif last_speaker is classification_agent:
        return questions_retriever_agent
    elif last_speaker is questions_retriever_agent:
        return questions_answering_agent
    elif last_speaker is questions_answering_agent:
        return unanswered_questions_agent
    elif last_speaker is unanswered_questions_agent:
        return None

groupchat = autogen.GroupChat(
    agents=[user_agent,
            classification_agent, 
            questions_retriever_agent, 
            questions_answering_agent,
            participants_identifier_agent,
            unanswered_questions_agent
           ],
    messages=[],
    max_round=20,
    speaker_selection_method=state_transition,
)

manager = autogen.GroupChatManager(groupchat=groupchat, llm_config=llm_config)

########################################################################################################

def initiate_report_chat(transcription_text, manager, user_agent):
    initial_questions = {}
    message = f"Report Datetime: {datetime.now()} Report: {transcription_text}"
    first_response = user_agent.initiate_chat(
        manager,
        message=message
    )
    last_message = first_response.chat_history[-1]
    content = last_message["content"]
    try:
        content_dict = json.loads(content)
        print("Parsed JSON content:", content_dict)
        initial_questions.update(content_dict)
    except json.JSONDecodeError:
        questions = [q.strip() + '?' if not q.strip().endswith('?') else q.strip() for q in content.split('\n') if q.strip()]
        initial_questions = {q: "" for q in questions}
        print("Parsed questions:", initial_questions)
    #return initial_questions
    result = {
            "initial_questions": initial_questions,
            "chat_history": first_response.chat_history
        }
    
    #return json.dumps(result, indent=4)
    return result

########################################################################################################

def convert_json_to_code(json_input):
    data = json.loads(json_input)  
    transcription = data.get("transcription", "")  
    transcription_text = f'transcription = "{transcription}"'  
    return transcription_text

def collect_transcription_data():
    transcription = {"transcription": ""}

    while not transcription["transcription"]:
        user_input = input("Please provide transcription text in JSON format (e.g. {\"transcription\": \"I want to start filing a report!\"}): ")
        try:
            user_data = json.loads(user_input)
            transcription.update(user_data)
        except json.JSONDecodeError:
            print("Invalid JSON input. Please try again.")
    
    transcription_text = convert_json_to_code(json.dumps(transcription))
    
    return transcription_text

########################################################################################################

def collect_answers_data(questions):
    collected_answers = {}

    while True:
        print(json.dumps(questions, indent=4))

        user_input = input("Please provide the answers in JSON format: ")

        try:
            user_data = json.loads(user_input)

            for field, default_value in questions.items():
                if field in user_data:
                    collected_answers[field] = user_data[field]
                elif default_value:
                    collected_answers[field] = default_value

            if len(collected_answers) == len(questions):
                break
            else:
                print("Some required fields are missing in your input. Please try again.")
        except json.JSONDecodeError:
            print("Invalid JSON input. Please try again.")

    return collected_answers
    

########################################################################################################

def second_state_transition(last_speaker, groupchat):
    messages = groupchat.messages

    if last_speaker is user_agent:
        return  participants_identifier_agent
    elif last_speaker is participants_identifier_agent:
        return questions_answering_agent
    elif last_speaker is questions_answering_agent:
        return unanswered_questions_agent
    elif last_speaker is unanswered_questions_agent:
        return None

second_groupchat = autogen.GroupChat(
    agents=[user_agent,
            participants_identifier_agent,
            questions_answering_agent,
            unanswered_questions_agent
           ],
    messages=[],
    max_round=20,
    speaker_selection_method=second_state_transition,
)

second_manager = autogen.GroupChatManager(groupchat=second_groupchat, llm_config=llm_config)

########################################################################################################

def second_initiate_report_chat(transcription_text, second_manager, user_agent):
    second_questions = {}
    message = f"Report Datetime: {datetime.now()} Report: {transcription_text}"
    second_response = user_agent.initiate_chat(
        second_manager,
        message=message
    )
    last_message = second_response.chat_history[-1]
    content = last_message["content"]
    try:
        content_dict = json.loads(content)
        print("Parsed JSON content:", content_dict)
        second_questions.update(content_dict)
    except json.JSONDecodeError:
        questions = [q.strip() + '?' if not q.strip().endswith('?') else q.strip() for q in content.split('\n') if q.strip()]
        second_questions = {q: "" for q in questions}
        print("Parsed questions:", second_questions)
    #return second_questions

    result = {
            "second_questions": second_questions,
            "chat_history": second_response.chat_history
        }
    
    #return json.dumps(result, indent=4)
    return result

########################################################################################################

dnis = ""
client_name = ""

@ app.route("/interaction", methods=["GET"])
def interaction():
    json = request.get_json()

    if json is None:
            return jsonify({"error": "Invalid JSON"}), 400
    
    # read the dnis and clientName from the request
    dnis = json.get("dnis")
    client_name = json.get("clientName")

    if dnis is not None and client_name is not None:
        # fetch the greeting and send back to Five9
        
    else:
        # pass the json to the process
        main(json)
    


def main(json=None):

    
    # Step 1:
    if json is not None:
        transcription_text = convert_json_to_code(json)
    else:
       transcription_text = collect_transcription_data()
    
    # Step 2:
    initial_chat_result = initiate_report_chat(transcription_text, manager, user_agent)
    initial_chat_history = initial_chat_result["chat_history"]
    initial_questions = initial_chat_result["initial_questions"]
    initial_answers = collect_answers_data(initial_questions)
    for field, answer in initial_answers.items():
        if field in initial_questions:
            initial_questions[field] = answer
    
    # Step 3
    second_chat_result = second_initiate_report_chat(transcription_text, second_manager, user_agent)
    second_chat_history = second_chat_result["chat_history"]
    second_questions = second_chat_result["second_questions"]
    second_answers = collect_answers_data(second_questions)
    for field, answer in second_answers.items():
        if field in second_questions:
            second_questions[field] = answer
    
    final_data = {
        "transcription_text": transcription_text, 
        "initial_questions": initial_questions,
        "second_questions": second_questions,
        "initial_chat_history":initial_chat_history,
        "second_chat_history":second_chat_history,
    }
    
    print("Final collected data:")
    print(json.dumps(final_data, indent=2))

if __name__ == "__main__":
    main()


# Test 1.

# Five9 Collects and SENDS:

# {"transcription": "I observed that Sarah clocked in early but didn’t start her work until hours later. This delay disrupts our team's workflow and raises concerns about accountability."}

# {"When did this happen?": "11/11/2024"}

# {"Affected Party": "Peter JJ", "Witness": "Moe YY", "Other": "Austin ZZ"}












