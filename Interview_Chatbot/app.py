from flask import Flask, request, render_template
import os
from langchain_openai import ChatOpenAI
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
        self.llm = ChatOpenAI(temperature=0.2)
        self.conversation_history = []
        self.interview_started = False
        self.question_count = 0 
    
    def extract_resume_info(self, resume_path, jd_path):
        resumeText = " "
        pdf_reader = PdfReader(resume_path)
        for page in pdf_reader.pages:
            resumeText += page.extract_text() 
        
        
        prompt = f"Extract the following information from the resume given below: Name, Email, Contact Info,  Website links, Education, Skills, Experience, Projects, Additional Info. Resume : {resumeText}"
        
        messages = [
        ("system", " Answer the following question with the given information. If you do not know the answer, say null"),
        ("human", prompt)]
        
        resumeResponse = self.llm.invoke(messages)
        print(resumeResponse.content)
        
        parts = resumeResponse.content.split('\n')
        

        jd_content = " "
        pdf_reader = PdfReader(jd_path)
        for page in pdf_reader.pages:
            jd_content += page.extract_text() 
      
        return resumeResponse.content, jd_content
    
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
        
 
    def interview(self, user_message, resume_content, jd_content):
        if not self.interview_started:
            if user_message.lower() == 'hello':
                default_question = "Let's start the interview. Please tell me about your experience."
                self.conversation_history.append(("bot", default_question))
                self.interview_started = True
                return default_question
            else:
                return "Type 'hello' to start the interview."
        
        else:
            self.conversation_history.append(("human", user_message))
            
            if self.question_count >= 4:
                return "Thank you for participating in the interview. It has been completed."        
        
            prompt = f"You are an interviewer tasked with assessing a candidate. Based on the conversation history provided along with the resume and job description given below, ask the candidate a question. Ensure it's unique. Based on the coversation history, make sure the question is not of the same topic as covered before. Only one question should be asked at a time. Resume : {resume_content}\n Job Description : {jd_content}\n Chat History : {self.conversation_history}"
            
            messages = [
                ("system", " Act as an interviewer and complete the task given. Do not ask questions on the same topics or repeat similar questions as covered in coversation history. Return only the final generated interview question and nothing else."),
                ("human", prompt)]
            
            question = self.llm.invoke(messages) 
            self.conversation_history.append(("bot", question.content))
            
            print(self.conversation_history)
            
            self.question_count += 1
            return question.content
        

chatbot = InterviewChatbot()

resume_path = '/Users/reethu/coding/Projects/AI_Recruiter/ResumeScoring/sample/Reethu_Resume.pdf' 
jd_path = '/Users/reethu/coding/Projects/AI_Recruiter/ResumeScoring/Job_description.pdf'

resume_content, jd_content = chatbot.extract_resume_info(resume_path, jd_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.form['user_message']
    bot_response = chatbot.interview(user_message, resume_content, jd_content)
    return {'bot_response': bot_response}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5002)