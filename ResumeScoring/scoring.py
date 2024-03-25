import os
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.schema.document import Document
from PyPDF2 import PdfReader
import pandas as pd

from langchain.prompts.example_selector import SemanticSimilarityExampleSelector

from secret_key import openapi_key
os.environ['OPENAI_API_KEY'] = openapi_key

def extract_resume_info(resume_path):
    # Create a PdfReader object
    text = " "
    docs = []
    pdf_reader = PdfReader(resume_path)
    for page in pdf_reader.pages:
        text += page.extract_text() 
        docs.append(Document(page_content=text))
    
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
    
    llm = OpenAI()
    chain = load_qa_chain(llm, chain_type="stuff",verbose=True)
    response = chain.run(input_documents=docs, question=prompt_template)
    
    parts = response.split('\n')
    
    extracted_info = {}
    for part in parts:
        part = part.strip()
        if part:
            key_value = part.split(':')
            if len(key_value) == 2:
                key = key_value[0].strip() 
                value = key_value[1].strip()  
                extracted_info[key] = value

    df = pd.DataFrame([extracted_info])
    print(df)
    
    skills = extracted_info.get('Skills', '')
    print(skills)
    return skills
    
    # name = extracted_info.get('Name', '')
    # email = extracted_info.get('Email', '')
    # contact_info = extracted_info.get('Contact Info', '')
    # education = extracted_info.get('Education', '')
    # skills = extracted_info.get('Skills', '')
    # experience = extracted_info.get('Experience', '')
    # designation = extracted_info.get('Designation', '')
    
    # # Print the extracted information
    # print("Name:", name)
    # print("Email:", email)
    # print("Contact Info:", contact_info)
    # print("Education:", education)
    # print("Skills:", skills)
    # print("Experience:", experience)
    # print("Designation:", designation)


def skills_matching(resume_skills, skills_df):
    # Extract primary and secondary skills from skills_df
    primary_skills = set(skills_df['Primary'].str.split(',').explode().str.strip())
    secondary_skills = set(skills_df['Secondary'].str.split(',').explode().str.strip())

    # Split resume skills into a set
    resume_skills_set = set(resume_skills.split(','))
    print("\n", primary_skills)
    print("\n", secondary_skills)
    print("\n", resume_skills_set)

    # Calculate the percentage match for primary skills
    primary_match = len(resume_skills_set.intersection(primary_skills)) / len(primary_skills) * 100

    # Calculate the percentage match for secondary skills
    secondary_match = len(resume_skills_set.intersection(secondary_skills)) / len(secondary_skills) * 100

    print("Percentage match of primary skills with resume:", round(primary_match, 2), "%")
    print("Percentage match of secondary skills with resume:", round(secondary_match, 2), "%")


resume_path = '/Users/reethu/coding/Projects/AI_Recruiter/ResumeScoring/sample/Dhaval_Thakkar_Resume.pdf'
resume_skills = extract_resume_info(resume_path)

skills_df = pd.read_csv('/Users/reethu/coding/Projects/AI_Recruiter/ResumeScoring/skills.csv')
skills_matching(resume_skills, skills_df)
