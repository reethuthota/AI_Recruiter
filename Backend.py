from flask import Flask, jsonify, request
from flask_cors import CORS
from pymongo import MongoClient
from bson import ObjectId
import os
import csv
from langchain_community.callbacks import get_openai_callback
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.schema.document import Document
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.schema.document import Document
from PyPDF2 import PdfReader
import io
import base64
from email.mime.text import MIMEText
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from requests import HTTPError
import gridfs 

app = Flask(__name__)
CORS(app, supports_credentials=True)


os.environ['OPENAI_API_KEY'] = ""
llm = OpenAI()
chain = load_qa_chain(llm, chain_type="stuff", verbose=True)
# MongoDB connection
#client = MongoClient('mongodb+srv://shreyasrajiv327:X9d3KlbihyKvVa9P@broadrangeairec.4pp66rt.mongodb.net/BroadrangeAI')
client = MongoClient('mongodb+srv://ree:ree123@ai-recruiter.robdv1v.mongodb.net/AI-Recruiter')
db = client['AI-Recruiter']

applicants_collection = db['applicants']
recruiters_collection = db['recruiters']
fs = gridfs.GridFS(db)
# Login route
@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    user_type = data.get('userType')

    if user_type == 'applicant':
        user = applicants_collection.find_one({'email': email, 'password': password})
        if user:
            # Convert ObjectId to string for serialization
            user['_id'] = str(user['_id'])
            user['userType']= 'applicant'
            return jsonify({'success': True, 'user': user})
        else:
            return jsonify({'success': False, 'message': 'Invalid email or password'}), 401

    elif user_type == 'recruiter':
        user = recruiters_collection.find_one({'email': email, 'password': password})
        if user:
            # Convert ObjectId to string for serialization
            user['_id'] = str(user['_id'])
            user['userType'] = 'recruiter'
            return jsonify({'success': True, 'user': user})
        else:
            return jsonify({'success': False, 'message': 'Invalid email or password'}), 401

    return jsonify({'success': False, 'message': 'Invalid user type'}), 400
# Signup route
@app.route('/api/signup', methods=['POST'])
def signup():
    data = request.get_json()
    name = data.get('name')
    print(name)
    email = data.get('email')
    password = data.get('password')
    user_type = data.get('userType')

    if user_type == 'applicant':
        existing_user = applicants_collection.find_one({'email': email})
        if existing_user:
            return jsonify({'success': False, 'message': 'Email already exists'}), 400

        new_user = {'name': name, 'email': email, 'password': password}
        applicants_collection.insert_one(new_user)
        return jsonify({'success': True, 'message': 'Applicant registered successfully'}), 201

    elif user_type == 'recruiter':
        existing_user = recruiters_collection.find_one({'email': email})
        if existing_user:
            return jsonify({'success': False, 'message': 'Email already exists'}), 400

        new_user = {'name': name, 'email': email, 'password': password}
        print(new_user)
        recruiters_collection.insert_one(new_user)
        return jsonify({'success': True, 'message': 'Recruiter registered successfully'}), 201

    return jsonify({'success': False, 'message': 'Invalid user type'}), 400
@app.route('/api/uploadJobDescription', methods=['POST'])
def upload_job_description():
    try:
        # Get the recruiter's email, company name, job title, and PDF file from the request
        recruiter_email = request.form['recruiterEmail']
        print(f"Recruiter Email: {recruiter_email}")
        company_name = request.form['companyName']
        print(f"Company Name: {company_name}")
        job_title = request.form['jobTitle']
        print(f"Job Title: {job_title}")
        file = request.files['file']
        print(f"File: {file.filename}")
        csv_file = request.files['csvFile']
        print(f"Skills.txt file : {csv_file.filename}")
        file_contents = csv_file.read().decode('utf-8')
        skills = file_contents.split('\n')
        
        # Read the PDF file and extract job description text
        job_description = ""
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            job_description += page.extract_text()
            
        # Read the CSV file and get the skills
        skills_string = ', '.join(skills)
        print("Skills:")
        print(skills_string)
        # Create a new collection for the job title
        collection_name = f'job_{job_title.lower().replace(" ", "_")}'
        db.create_collection(collection_name)

        # Store the job description details in the "jobs" collection
        job_data = {
            'recruiterEmail': recruiter_email,
            'companyName': company_name,
            'jobTitle': job_title,
            'table_name': collection_name,
            'jobDescription': job_description,
            'skills': skills
        }
        db['jobs'].insert_one(job_data)
        print(job_data)
        return jsonify({'success': True, 'message': 'Job description uploaded successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/recruiterJobs/<string:recruiter_email>', methods=['GET'])
def get_recruiter_jobs(recruiter_email):
    try:
        # Fetch job openings created by the recruiter
        jobs = list(db['jobs'].find({'recruiterEmail': recruiter_email}))
        print(jobs)
        # Convert ObjectId fields to strings in each job document
        jobs_data = [{'jobTitle': job['jobTitle'], 'table_name': job['table_name']} for job in jobs]

            
        return jsonify(jobs_data), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/jobDetails/<string:table_name>', methods=['GET'])
def get_job_details(table_name):
    try:
        job_details_collection = db[table_name]
        applicants = list(job_details_collection.find())
        
        # Extract name and email from each applicant
        formatted_applicants = []
        for applicant in applicants:
            name = applicant.get('name', '')
            email = applicant.get('email', '')
            formatted_applicant = {
                'name': name,
                'email': email
            }
            formatted_applicants.append(formatted_applicant)
        print(formatted_applicants)
        return jsonify(formatted_applicants), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

#@app.route('/api/jobDetails/<string:table_name>', methods=['GET'])
@app.route('/api/jobs', methods=['GET'])
def get_all_jobs():
    try:
        # Fetch all jobs from the jobs collection
        jobs = db['jobs'].find()  # Get cursor for all jobs
        jobs_list = []  # Initialize an empty list to store job details
        for job in jobs:
            job_details = {
        'companyName': job['companyName'],
        'jobTitle': job['jobTitle'],
        'table_name': job['table_name']
          }
            jobs_list.append(job_details)
        print(jobs_list)
        return jsonify(jobs_list), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
import json

@app.route('/api/apply/<job_title>', methods=['POST'])
def apply_for_job(job_title):
    name = request.form['name']
    email = request.form['email']
    file = request.files['file']

    # Extract information from the file
    extracted_info,skills = extract_resume_info(file)
    print("PRINTING THE EXTRACTED INFORMATION")
    print(extracted_info)

    # Convert extracted_info to JSON format
    extracted_info_json = json.dumps(extracted_info)
    print("PRINTING THE EXTRACTED INFORMATION")
    print(extracted_info_json)
    # Store user data in a collection corresponding to the job title
    db[job_title].insert_one({
        'name': name,
        'email': email,
        'extracted_info': extracted_info_json,
        'skills':skills
    })
    print("Data added to MongoDB")

    return jsonify({'message': 'Data uploaded successfully', 'extracted_info': extracted_info_json})

def extract_resume_info(file):
    # Create a PdfReader object from the file-like object
    pdf_reader = PdfReader(file)

    # Initialize an empty string to store text
    text = ""
    
    # Iterate through the pages of the PDF and extract text
    for page in pdf_reader.pages:
        text += page.extract_text()
    
    # Create a Document object with the extracted text
    docs = [Document(page_content=text)]
    
    # Define the prompt template
    prompt_template = """
        Extract the following information from the resume:
        
        Name: {}
        Email: {}
        Contact Info: {}
        Education: {}
        Skills: {}
        Experience: {}
        Designation: {}
    """
    
    # Initialize the language model and chain
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff", verbose=True)
    
    # Run the chain with the document and prompt template
    response = chain.run(input_documents=docs, question=prompt_template)
    
    # Split the response into lines
    parts = response.split('\n')
    
    # Initialize a dictionary to store extracted information
    extracted_info = {}
    for part in parts:
        part = part.strip()
        if part:
            key_value = part.split(':')
            if len(key_value) == 2:
                key = key_value[0].strip() 
                value = key_value[1].strip()  
                extracted_info[key] = value
    
    skills = extracted_info.get('Skills', '')
    print(skills)
    # Iterate through the lines and extract key-value pairs
    return text,skills

@app.route('/api/fetchJD/<string:jobTitle>', methods=['GET'])
def fetch_job_description(jobTitle):
    try:
        emails=[]
        # Fetch extracted info for the specified job title
        print("JOB TITLE", jobTitle)
        collection_name = f'job_{jobTitle.lower().replace(" ", "_")}'
        print("collection name:", collection_name)
        
        # Check if the collection exists
        if collection_name not in db.list_collection_names():
            return jsonify({'error': f'Collection {collection_name} does not exist'}), 404
        
        # Retrieve documents from the collection
        extracted_info_docs = db[collection_name].find()

        # Iterate over retrieved documents
        for doc in extracted_info_docs:
            print("Entered Loop")
            print("Name of the Applicant:", doc.get('name', 'N/A'))
            
            # Retrieve job skills and description
            job_doc = db['jobs'].find_one({'table_name': collection_name}, {'_id': 0, 'jobDescription': 1})
            skills = db['jobs'].find_one({'table_name': collection_name}, {'_id': 0, 'skills': 1})
            if skills:
                skills_array = skills.get("skills", [])
                skills_string = ", ".join(skills_array)
                print("Skills:", skills_string)
                # Calculate scores and update documents
                score = jobDescription_matching(doc.get('extracted_info', ""), job_doc)
                skillsScore = skills_matching(doc.get('skills', []), skills_string)
                print("Calculated scores")
                db[collection_name].update_one({'_id': doc['_id']}, {'$set': {'JDscore': score}})
                db[collection_name].update_one({'_id': doc['_id']}, {'$set': {'Skillsscore': skillsScore}})
            else:
                print("No job document found for", collection_name)
            JDscore = int(doc.get('JDscore', 0))
            print("JDScore :", JDscore)
            if JDscore > 65:
                email = doc.get('email', None)
            if email:
                emails.append(email)
        
       
        print(emails)
        for e in emails:
            subject = "Selection for jobs!"
            body = "congratulations"
            send_email(e, subject, body)
        return jsonify({'message': f'Scores calculated and stored successfully fordocuments'}), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def jobDescription_matching(resume_response, job_description):
    llm = ChatOpenAI(temperature=0.2)
    prompt = f"Give the job fit as a percentage for the job description : {job_description} and the given resume : {resume_response}."
    messages = [
    ("system", "Answer the following question with a percentage as an answer. Do not give any further explanations. Output the percentage without the % sign. If you do not know the answer, say 0"),
    ("human", prompt)]
    jobDescription_matching_score = llm.invoke(messages)
    print("Job Description Matching Score :" , jobDescription_matching_score.content)
    return jobDescription_matching_score.content

def skills_matching(resume_skills, skills_df):
    llm = ChatOpenAI(temperature=0.2)
    with get_openai_callback() as cb:
        # Calculate similarity scores for primary skills
        prompt1 = f"Find the intersection between set 1 : {', '.join(resume_skills)} and set 2 : {skills_df}. Give me a percentage as (intersection/number of items in set 2)*100."  
        messages = [
            ("system", "Answer the following question with a percentage as an answer. Do not give any further explanations. Output the percentage without the % sign. If you do not know the answer, say 0"),
            ("human", prompt1)]
        primary_similarity_scores = llm.invoke(messages)
        print("Primary Skills Match :" , primary_similarity_scores.content)
        print(cb)
    return primary_similarity_scores.content

def send_email(email, subject, body):
    SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
    flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
    creds = flow.run_local_server(port=0)
    service = build('gmail', 'v1', credentials=creds)

    message = MIMEText(body)
    message['to'] = email
    message['subject'] = subject
    create_message = {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}

    try:
        message = (service.users().messages().send(userId="me", body=create_message).execute())
        print(F'sent message to {message} Message Id: {message["id"]}')
    except HTTPError as error:
        print(F'An error occurred: {error}')
        message = None
if __name__ == '__main__':
    app.run(debug=True, port=8000)