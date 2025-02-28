import os
from openai import AzureOpenAI
import sounddevice as sd
import scipy.io.wavfile as wav
from dotenv import load_dotenv

# region Azure OpenAI Whisper - Speech to Text


load_dotenv()  

api_key = os.getenv("OPENAI_API_KEY")
azure_endpoint = os.getenv("OPENAI_API_BASE")
api_version = os.getenv("API_VERSION")
model = os.getenv("GPT_MODEL")

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

