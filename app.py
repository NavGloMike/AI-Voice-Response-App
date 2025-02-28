import os
from flask import Flask, request, jsonify, session
from dotenv import load_dotenv
import ethicsPoint_apis as EthicsPoint  # Import the EthicsPoint class
from custom_print import printc

#load_dotenv()

app = Flask(__name__)

@app.route("/callStart", methods=["Get"])
def call_start():
    dnis = request.args.get("dnis")
    if dnis:
        session["dnis"] = dnis
    guidelines = EthicsPoint.get_guidelines(dnis)
    return jsonify(guidelines)


def main():
    # Initialize the EthicsPoint class
    dnis = "4277"
    guidelines = EthicsPoint.get_guidelines(dnis)
    printc(guidelines)
    dnis = guidelines["DNIS"]
    clientName = guidelines["ClientName"]
    clientKey = guidelines["ReDirective"]
    printc(f"DNIS: {dnis}, ClientName: {clientName}, ClientKey: {clientKey}")

if __name__ == '__main__':
   main()
   app.run(debug=True)
   