import pandas as pd
import os
import app
from dotenv import load_dotenv
from openai import AzureOpenAI
from datetime import datetime
import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import ffmpeg
import time
import json
from flask import Flask, request, jsonify
import requests

load_dotenv()  
print(os.getcwd())

model=os.getenv("GPT_MODEL")
azure_endpoint=os.getenv("GPT_ENDPOINT")
api_key=os.getenv("GPT_API_KEY")
api_version=os.getenv("API_VERSION")
record_audio=os.getenv("RECORD_AUDIO")

app = Flask(__name__)

# region Azure OpenAI Whisper - Speech to Text

def transcribe_audio(audio_test_file):
    client = AzureOpenAI(
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),  
        api_version="2024-02-01",
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
    )

    deployment_id = "whisper"

    with open(audio_test_file, "rb") as audio_file:
        result = client.audio.transcriptions.create(
            file=audio_file,
            model=deployment_id
        )
        
    return result.text
# endregion

# region Record audio using the default microphone

# audio_test_file = "./recording.wav"
# transcription_result = transcribe_audio(audio_test_file)
# print(transcription_result)

def record_audio(duration, filename):
    sample_rate = 44100  

    print("Recording...")

    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  
    print("Recording finished.")

    wav.write(filename, sample_rate, audio_data)


# endregion

# region Azure OpenAI

load_dotenv()  

api_key = os.getenv("OPENAI_API_KEY")
azure_endpoint = os.getenv("OPENAI_API_BASE")
api_version = os.getenv("API_VERSION")
model = "gpt-4o"

# audio_test_file = "./recording.wav"
# record_audio(20, audio_test_file)
# transcription_text = transcribe_audio(audio_test_file)
# print("Transcription:", transcription_text)


def get_response_client(template, text, temprature, top_p, model=model, azure_endpoint=azure_endpoint, api_key=api_key, api_version=api_version):
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
        top_p=top_p,
        max_tokens=4000,
        messages=conversation
    )

    return response.choices[0].message.content

# endregion

# region REST handler for accepting text and sending the response

@app.route('/getResponse', methods=['POST'])
def get_response():
    data = request.get_json()
    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    
    return process_text(data['text'])

# endregion

def process_text(text):
    transcription_text = text

    # Classification
    classification_template = f"{template1}, Classification Dictionary: {classification_dictionary}"
    detected_class = get_response_client(classification_template, transcription_text, 1.0, 0.0)
    
    print("Detected Class:", detected_class)
    
    for key, value in json.loads(detected_class).items():
        report = f"It looks like your report has to do with {key}, which is defined as {value}"
    
    # Retrieving Questions
    # retrieval_template = f"{template2}, Questions Dictionary: {questions_dictionary}, Questions Set: {questions_set} Detected Class: {detected_class}"
    # retrieved_questions = get_response_client(retrieval_template, 'please retrieve questions:', 1.0, 0.0)

    # # Answering Questions
    # questions_answering_template = f"{template3}, Retrieved Questions: {retrieved_questions},  Report Datetime: {datetime.now()}, Report: {transcription_text}"
    # answered_questions = get_response_client(questions_answering_template, 'please answer questions:', 1.0, 0.0)

    # # Identifying Participants
    # participants_identifier_template = f"{template4}, Report Datetime: {datetime.now()}, Report: {transcription_text}"
    # participant_info = get_response_client(participants_identifier_template, 'please retrieve participants:', 1.0, 0.0)

    # # Final Report
    # report_dict = {
    #     "Transcription": transcription_text,
    #     "Detected_Issue_Type": detected_class,
    #     "Retrieved_Questions": retrieved_questions,
    #     "Answered_Questions": answered_questions,
    #     "Participant_Information": participant_info
    # }
    #return jsonify(report_dict)

    
    return report

  

# region Templates

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

classification_dictionary = {
    0:"Harassment: Reports of harassment that are linked to a protected characteristic (such as race, gender, sex, religion, disability, age, etc.) and includes allegations of unwelcome behavior that is offensive to a reasonable person, and is related to, or done because of, a protected characteristic.",
    1:"Discrimination: Reports of discrimination or concerns relating to accommodation requests. Discrimination generally occurs when there is a negative employment action impacting a term or condition of employment, that action is taken by the employer (which can include managers as well as others who have control over terms or conditions of work such as team leads), and the action was taken because of a protected characteristic. A workplace accommodation involves a request to adjust something relating to work linked to either a religious practice or belief or a disability. This includes allegations or reports related to religious practices or beliefs or speaks to a workplace modification or leave request linked to a medical condition or disability.",
    2:"Substance Abuse: Reports related to impairment resulting from use of substances (such as drugs or alcohol, whether legal or illegal) that impact the workplace or violate a policy. The activity can include on- or off-duty and on- or off-premises conduct.",
    3:"Compensation and Benefits: Reports related to matters of compensation, pay, insurance, time-off, retirement benefits, leaves of absence (paternity, maternity, other medical) and other common employee benefits. Examples include incorrect paycheck or inaccurate recording of vacation, time-off and sick time.",
    4:"Workplace Civility: Reports related to abusive or disrespectful behavior connected to work that are not harassment or discrimination.",
    5:"Other Human Resources: Reports that cannot be categorized elsewhere and likely involve Human Resources. Examples include performance management, discipline, immigration, labor relations, grievances, job eliminations, arrests and convictions, and the sale or distribution of drugs.",
    6:"Retaliation: Reports of retaliation (including claims of reprisal or victimization) of any kind against an employee including claims of any action taken to punish or dissuade an employee from making a report or participating in an investigation either internally or externally. Retaliation claims most often involve allegations against a manager, supervisor or some other person with control and power over the reporting person. However, retaliation can also involve conduct by a coworker.",
    7:"Conflicts of Interest: Reports about a conflict of interest, either a self-report or a report involving the behavior of others. A conflict of interest can arise in any situation where an employee’s financial or personal interest could potentially or actually interfere, or even appear to interfere, with their business judgement or the interests of the organization.",
    8:"Confidential and Proprietary Information: Reports related to confidential and proprietary information or intellectual property. Confidential information is any non-public information that is not intended or permitted to be shared beyond those with a genuine business need to know the information. Confidential information can include information about people or companies and specifically includes business plans, trade secret information, customer lists, sales and marketing strategies, pricing, product development plans, and any notes or documentation of the foregoing. Intellectual property refers to an original, intangible creation of human intellect that is legally protected from unauthorized use. Intellectual property includes patents, trademarks and copyrighted works of authorship, like photographs, music, literary works, graphic design, source code, and audio and audiovisual recordings.",
    9:"Data Privacy and Protection: Reports related to the rights and responsibilities relating to data held or processed by an organization. This data can include information about employees, customers, consumers, or others. Examples include allegations of data misuse, loss or theft of data, breaches or attempted breaches, or requests by an individual relating to their own data.",
    10:"Free and Fair Competition: Reports involving activities that undermine free and fair competition in the marketplace. These activities frequently involve any agreement with a competitor to fix prices or otherwise limit competition. Even the appearance of such agreement is problematic.",
    11:"Bribery and Corruption: Reports of public or private instances of bribery. Bribery occurs when a person offers money or something else of value – to an official or someone in a position of power or influence – for the purpose of gaining influence over them. Corruption includes dishonest or illegal behavior – especially of people in authority – using their power to do dishonest or illegal things in return for money or to get an advantage over someone else.",
    12:"Insider Trading: Reports that a person is buying or selling any company’s (employer’s or any other company’s) securities and/or stock based on non-public information as well as passing (tipping) this information on to someone else who then buys or sells stock.",
    13:"Global Trade: Reports related to the import and export of goods and services globally. It can include imports (bringing goods or services into a country) or exports (sending goods or services - including software - from one country to another). This category also includes reports relating to sanctions such as trade sanctions, which make it unlawful to do business with sanctioned people or countries.",
    14:"Political Activity: Reports of improper use of employer resources (time, assets, brand, etc.) for political activity (by an individual or an organization) such as using work time for political activities, pressuring colleagues to give money or time to a PAC or associating organization name with a political candidate, official, or group. It can also include misuse of company funds for political activities, using company resources to create or distribute political messages and violations of lobbying regulations and restrictions.",
    15:"Human Rights: Reports related to human rights which generally refer to the basic rights and freedoms of individuals. Examples include reports relating to human trafficking or modern slavery that involve the use of force, fraud or coercion to obtain labor or sex for money, drugs or other goods.",
    16:"Product Quality and Safety: Reports about quality and safety issues related to products. Examples include allegations that a product is not safe for intended use, is putting others at risk of harm, or that it fails to meet industry standards.",
    17:"Other Business Integrity: Reports related to business integrity that cannot be categorized elsewhere. Examples include industry specific policies, regulations or laws.",
    18:"Accounting, Auditing and Financial Reporting: Reports related to accounting, financial reporting or auditing. Examples include the unethical or improper recording and analysis of the business and financial transactions associated with generally accepted accounting practices. Examples include: misstatement of revenues, misstatement of expenses, misstatement of assets, misapplications of GAAP principles, and wrongful transactions.",
    19:"Misuse or Misappropriation of Assets: Reports that the organization’s assets are being wasted, inappropriately used, abused, or not properly protected. This category can include a wide array of assets such as property, tools, money, credit cards, facilities, company vehicles, employee time, and even abuse of employer provided benefits.",
    20:"Imminent Threat to a Person, Animals or Property: Reports of imminent or immediate threat of harm to a person or people, animals or property. Reports may or may not involve a weapon and generally are the kind of incident where authorities (such as police or fire) are called to assist.",
    21:"Environmental: Reports about impact to the environment. This could include intentional, negligent or accidental acts or omissions that harm the environment or violate policy or regulatory or legal requirements. It can also include acts or omissions that otherwise present a risk to the climate. Examples can include such things as spills, mismanaged wastewater or resources, release into the atmosphere of harmful materials or substances, or improper disposal of hazardous waste.",
    22:"Health and Safety: Reports about workplace safety. This can include employee safety and facilities or equipment. Each employee is responsible for maintaining a safe and healthy workplace for all employees by following safety and health rules and practices and reporting accidents, injuries and unsafe equipment, practices or conditions. Reports about physical security in a facility.",
    23:"Other: Reports that do not fit any of the other categories listed.",
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
    0: ["q1", "q4"],  # Harassment
    1: ["q1", "q3"],  # Discrimination
    2: ["q1", "q2"],  # Substance Abuse
    3: ["q1", "q18"],  # Compensation and Benefits
    4: ["q1", "q18"],  # Workplace Civility
    5: ["q1", "q18"],  # Other Human Resources
    6: ["q1", "q18", "q19"],  # Retaliation
    7: ["q1", "q15", "q16"],  # Conflicts of Interest
    8: ["q1", "q20"],  # Confidential and Proprietary Information
    9: ["q1", "q20"],  # Data Privacy and Protection
    10: ["q1", "q18"],  # Free and Fair Competition
    11: ["q1", "q18"],  # Bribery and Corruption
    12: ["q1", "q20"],  # Insider Trading
    13: ["q1", "q18"],  # Global Trade
    14: ["q1", "q18"],  # Political Activity
    15: ["q1", "q18"],  # Human Rights
    16: ["q1", "q18"],  # Product Quality and Safety
    17: ["q1", "q18"],  # Other Business Integrity
    18: ["q1", "q20"],  # Accounting, Auditing and Financial Reporting
    19: ["q1", "q6", "q7"],  # Misuse or Misappropriation of Assets
    20: ["q1", "q13", "q14"],  # Imminent Threat to a Person, Animals or Property
    21: ["q1", "q18"],  # Environmental
    22: ["q1", "q18"],  # Health and Safety
    23: ["q1", "q19"],  # Other
}

participants_dictionary={0: ["Affected Party","Perpetrator","Witness","Other"]}

classification_dictionary=str(classification_dictionary)

# endregion

# def record_audio():
#     # region Start recording:
#     audio_test_file = "./recording.mp3"
#     record_audio(20, audio_test_file)
#     transcription_text = transcribe_audio(audio_test_file)
#     print("Transcription:", transcription_text)
#     # endregion

#     # region Classification
#     # template
#     classification_template = f"{template1}, Classification Dictionary: {classification_dictionary}"
#     detected_class = get_response_client(classification_template, transcription_text,1.0,0.0)
#     print(detected_class)
#     # endregion

#     # region Retrieving Questions
#     # template
#     retrieval_template = f"{template2}, Questions Dictionary: {questions_dictionary}, Questions Set: {questions_set} Detected Class: {detected_class}"

#     retrieved_questions = get_response_client(retrieval_template, 'please retrieve questions:', 1.0, 0.0)
#     print(retrieved_questions)
#     # endregion

#     # region Attempt to answer questions
#     # template
#     questions_answering_template = f"{template3}, Retrieved Questions: {retrieved_questions},  Report Datetime: {datetime.now()}, Report: {transcription_text}"

#     answered_questions = get_response_client(questions_answering_template, 'please answer questions:', 1.0,0.0)
#     print(answered_questions) 
#     # endregion

#     # region Identify Participants
#     # template
#     participants_identifier_template = f"{template4}, Report Datetime: {datetime.now()}, Report: {transcription_text}"

#     participant_info = get_response_client(participants_identifier_template, 'please retrieve participants:', 1.0,0.0)
#     print(participant_info) 
#     # endregion

#     # region Final Report
#     report_dict = {
#         "Transcription": transcription_text,
#         "Detected_Issue_Type": detected_class,
#         "Retrieved_Questions": retrieved_questions,
#         "Answered_Questions": answered_questions,
#         "Participant_Information": participant_info
#     }

#     report_json = json.dumps(report_dict, indent=4)

#     print(report_json)
    # endregion

# if record_audio == "True":
#     record_audio()

# else:
load_dotenv()  
debug = os.getenv("DEBUG")
if __name__ == '__main__':
    app.run(debug == 'True')
