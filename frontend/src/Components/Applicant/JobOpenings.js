import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './JobOpenings.css'; // Import CSS file for styling
import ApplicantChatbot from './ApplicantChatbot';

function JobOpenings() {
  const [jobs, setJobs] = useState([]);
  const [redirectToChatbot, setRedirectToChatbot] = useState(false);

  useEffect(() => {
    // Fetch jobs from MongoDB when component mounts
    const fetchJobs = async () => {
      try {
        const response = await axios.get('http://127.0.0.1:8000/api/jobs');
        setJobs(response.data);
      } catch (error) {
        console.error('Error fetching jobs:', error);
      }
    };

    fetchJobs();
  }, []); // Empty dependency array ensures this effect runs only once when component mounts

  const handleClick = (table_name) => {
    // Navigate to the next page with the selected table_name
    setRedirectToChatbot(true);
  };

  if (redirectToChatbot) {
    return <ApplicantChatbot />;
  }

  return (
    <div className="jobOpeningsContainer">
      <h2>Job Openings</h2>
      <div className="jobList">
        {jobs.map(job => (
          <div key={job.table_name} className="jobItem" onClick={() => handleClick(job.table_name)}>
            <h3>{job.jobTitle}</h3>
            <p>{job.companyName}</p>
            {/* Add more job details as needed */}
          </div>
        ))}
      </div>
      <button className="apply-btn" onClick={() => setRedirectToChatbot(true)}>Apply for Jobs</button>
    </div>
  );
}

export default JobOpenings;
