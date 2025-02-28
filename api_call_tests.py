import ethicsPoint_apis as EpApi


def get_guidelines_test():
    return EpApi.get_guidelines(dnis="zz_000000")

def get_issue_types_test():
    return EpApi.get_issue_types(
        clientKey="trial12", 
        tierId=1, 
        epimApiAuthKey="U4K@!ugrqQqcrWCV8!IV", 
        languageCode="en-US", 
        locationId=151127, 
        locationDataPrivacy="false"
    )

def get_locations_test():
    return EpApi.get_locations(
        clientKey="trial12", 
        epimApiAuthKey="U4K@!ugrqQqcrWCV8!IV", 
        platformCorrectionId="40599ugm2as9dvw"
    )

def get_violation_types_test():
    return EpApi.get_violation_types(
        clientKey="trial12", 
        tierId=1, 
        epimApiAuthKey="U4K@!ugrqQqcrWCV8!IV", 
        languageCode="en-US", 
        locationId=151127, 
        locationDataPrivacy="false"
    )

def get_questions_test():
    return EpApi.get_questions(
        clientKey="trial11", 
        epimApiAuthKey="U4K@!ugrqQqcrWCV8!IV", 
        violationTypeId=102869, 
        dataPrivacyMode=0, 
        languageCode="en-US"
    )
