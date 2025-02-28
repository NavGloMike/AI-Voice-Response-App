from openai import AzureOpenAI
import time
import json

template1 = """You are an ethics and compliance analyst. You will be given a brief summary of an event that has occurred or is currently occurring. 
            Your job is to classify the event using the provided dictionary.
            Your final response should include the retrieved key-value pairs from the given dictionary.
            
            Example: 
            {
                "Reduction of Hours": "Events that are related to an employee having the number of hours they work reduced."
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


def get_response_client(endpoint, template, text, temperature, top_p):
    
    model=model, 
    azure_endpoint=endpoint, 
    api_key=api_key, 
    api_version=api_version,
    max_tokens=max_tokens

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
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        messages=conversation
    )

    print(response.choices[0])
    return response.choices[0].message.content

def process_text(text):
    transcription_text = text

    # Classification
    classification_template = f"{template1}, Classification Dictionary: {classification_dictionary}"
    detected_class = get_response_client(classification_template, transcription_text, 1.0, 0.0)
    
    print("Detected Class:", detected_class)
    report = json.loads(detected_class)

    desc = ""
    name = ""
    
    for key, value in json.loads(detected_class).items():
         desc = value
         name = key

    expected_json = {
        "caseDetails": transcription_text,
        "Detected_Risk_Type": {
            name: desc
        }
    }
    # Convert to JSON string

    report = json.dumps(expected_json, indent=4)

    return report