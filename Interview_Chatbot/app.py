from flask import Flask, request, render_template
import os
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.schema.document import Document
from PyPDF2 import PdfReader
import pandas as pd

from secret_key import openapi_key
os.environ['OPENAI_API_KEY'] = openapi_key

app = Flask(__name__)

class InterviewChatbot:
    def __init__(self):
        self.llm = OpenAI()
        self.conversation_history = []
        self.interview_started = False
        self.question_count = 0 
    
    def extract_resume_info(self, resume_path, jd_path):
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
        
        chain = load_qa_chain(self.llm, chain_type="stuff", verbose=True)
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
    
        self.conversation_history.append(response)
        df = pd.DataFrame([extracted_info])
        
        # cover_letter_content = " "
        # pdf_reader = PdfReader(cover_letter_path)
        # for page in pdf_reader.pages:
        #     cover_letter_content += page.extract_text() 
        #     docs.append(Document(page_content=cover_letter_content))
    
        # df["Cover Letter"] = cover_letter_content
        
        jd_content = " "
        pdf_reader = PdfReader(jd_path)
        for page in pdf_reader.pages:
            jd_content += page.extract_text() 
            docs.append(Document(page_content=jd_content))
        
        df["Job Description"] = jd_content
        
        return docs
    
    def interview_scoring(self, conversation_history) :
        criteria = {
            "Communication": 0,
            "Relevant Experience": 0,
            "Problem Solving": 0,
        }
        
        chat_content = '\n'.join(self.conversation_history)
        doc_chat_content = Document(page_content=chat_content)

        # Analyze conversation history for each criterion
        for criterion in criteria.keys():
            chain = load_qa_chain(self.llm, chain_type="stuff", verbose=True)
            question = f"Evaluate the applicant's {criterion} based on the conversation history. Give me a score between 1 to 10."
            answer = chain.run(input_documents=[doc_chat_content], question=question)
            score = float(answer.strip())  # Assuming the score is provided as text
            criteria[criterion] = score

        total_score = sum(criteria.values())
        print(total_score)
    
    def interview(self, user_message, docs):
        if not self.interview_started:
            if user_message.lower() == 'hello':
                # Ask the default question if the conversation hasn't started yet
                default_question = "Let's start the interview. Please tell me about your experience."
                self.conversation_history.append(default_question)
                self.interview_started = True
                # return {'question': default_question, 'bot_response': ''}
                return default_question
            else:
                # Return a message prompting the user to start the interview
                # return {'question': '', 'bot_response': "Type 'hello' to start the interview."}
                return "Type 'hello' to start the interview."
        
        else:
            if self.question_count >= 4:  # Stop the interview after 5 questions
                self.interview_scoring(self.conversation_history)
                return "Thank you for participating in the interview. It has been completed."
            
            # Combine resume and cover letter into a single document
            combined_document_content = '\n'.join(self.conversation_history)
            for doc in docs:
                combined_document_content += '\n' + doc.page_content
            combined_document = Document(page_content=combined_document_content)
            
            # Generate a question using OpenAI
            question_generation_prompt = "Generate an interview question based on the conversation history along with given resume personalising the questions required based on the given job description. Do not repeat any questions or ask similar questions and cover a variety of domains, projects and skills mentioned in the resume. Ask only one question at a time and ask questions to assess the candidate as best as possible. \n\n"
            chain = load_qa_chain(self.llm, chain_type="stuff", verbose=True)
            question = chain.run(input_documents=[combined_document], question=question_generation_prompt)
            
            self.conversation_history.append(user_message)
            self.question_count += 1  # Increment the question count
            return question
        

chatbot = InterviewChatbot()
resume_path = '/Users/reethu/coding/Projects/AI_Recruiter/ResumeScoring/sample/Reethu_Resume.pdf' 
jd_path = '/Users/reethu/coding/Projects/AI_Recruiter/ResumeScoring/Job_description.pdf'

docs = chatbot.extract_resume_info(resume_path, jd_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['user_message']
    bot_response = chatbot.interview(user_message, docs)
    return {'bot_response': bot_response}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)