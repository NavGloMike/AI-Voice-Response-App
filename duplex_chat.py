# Five9_ContactCenterIntake-Approach
# Author: Kyle Zarifsadr - NOV 2024
# After installing the dependencies and setting up the .env file to access the Azure OpenAI endpoint, you can run this program locally in the terminal. The goal is to demonstrate programmatically how to send and receive JSON data between the Fivetran and NAVEX containers.
# This code requires a few modifications to function properly in a single container.

import numpy as np
import json
import pandas as pd
# import os
import autogen
import requests
import sys
import templates
import configs
from five9_api import Five9Api
from flask import Flask, jsonify
from datetime import datetime
from dotenv import load_dotenv
from autogen import AssistantAgent, UserProxyAgent
from autogen import ConversableAgent
from autogen import GroupChat
from autogen import GroupChatManager
from openai import AzureOpenAI

load_dotenv()
gpt40_configs = configs.Gpt4o()
five9Api = Five9Api()

########################################################################################################

llm_config = {
    "config_list": [
        {
            "model": gpt40_configs.gpt_model,
            "api_key": gpt40_configs.api_key,
            "api_type": "azure",
            "base_url": gpt40_configs.api_base,
            "api_version": gpt40_configs.api_version,
        },
    ],
    "temperature": configs.temperature,
    "timeout": configs.timeout
}

########################################################################################################

# region Agents

user_agent = ConversableAgent(
    name="user_agent",
    system_message= "You are reporting an event that has occurred or is currently occurring.",
    llm_config=llm_config,
    human_input_mode= "NEVER",
)

classification_agent = ConversableAgent(
    name="classification_agent",
    system_message= templates.template1 + "NAVEX Issue Types:" + str(templates.classification_dictionary),
    llm_config=llm_config,
    human_input_mode= "NEVER",
)

questions_retriever_agent = ConversableAgent(
    name="questions_retriever_agent",
    system_message= "NAVEX Questions Dictionary:" + str(templates.questions_dictionary) + "NAVEX Questions Set:" + str(templates.questions_set) + templates.template2,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

questions_answering_agent = ConversableAgent(
    name="questions_answering_agent",
    system_message= templates.template3,
    llm_config=llm_config,
    human_input_mode="NEVER",
)

participants_identifier_agent = ConversableAgent(
    name="participants_identifier_agent",
    system_message= templates.template4 + "NAVEX Case Participants List: " + str(templates.participants_dictionary),
    llm_config=llm_config,
    human_input_mode= "NEVER",
)

unanswered_questions_agent = ConversableAgent(
    name="unanswered_questions_agent",
    system_message= templates.template5,
    llm_config=llm_config,
    human_input_mode= "NEVER",
)

# endregion

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

    

    # Send a request to the caller asking for a transcription

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

    # Send questions to Five9 and wait for answers to be returned
    epim_config = configs.Epim()
    
    
    response = five9Api.get_caller_response(questions)

    if response:
        collected_answers = response
    else:
        print("Failed to receive a response.")

    # while True:
    #     print(json.dumps(questions, indent=4))

    #     user_input = input("Please provide the answers in JSON format: ")

    #     try:
    #         user_data = json.loads(user_input)

    #         for field, default_value in questions.items():
    #             if field in user_data:
    #                 collected_answers[field] = user_data[field]
    #             elif default_value:
    #                 collected_answers[field] = default_value

    #         if len(collected_answers) == len(questions):
    #             break
    #         else:
    #             print("Some required fields are missing in your input. Please try again.")
    #     except json.JSONDecodeError:
    #         print("Invalid JSON input. Please try again.")

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

def begin_chat():
    # Step 1:
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


# Test 1.

# Five9 Collects and SENDS:

# {"transcription": "I observed that Sarah clocked in early but didn’t start her work until hours later. This delay disrupts our team's workflow and raises concerns about accountability."}

# {"When did this happen?": "11/11/2024"}

# {"Affected Party": "Peter JJ", "Witness": "Moe YY", "Other": "Austin ZZ"}











