import React, { useState, useEffect } from 'react';
import axios from 'axios';
import './ViewApplicants.css'; // Import the CSS file

function ViewApplicants() {
    const [jobs, setJobs] = useState([]);
    const [selectedJob, setSelectedJob] = useState(null);
    const [recruiterEmail, setRecruiterEmail] = useState('');
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');

    useEffect(() => {
        const fetchData = async () => {
            setLoading(true);
            setError('');

            try {
                if (recruiterEmail) {
                    const response = await axios.get(`http://127.0.0.1:8000/api/recruiterJobs/${recruiterEmail}`);
                    setJobs(response.data);
                }
            } catch (error) {
                setError('Error fetching jobs');
            }

            setLoading(false);
        };

        fetchData();
    }, [recruiterEmail]);

    const handleJobClick = async (tableName, jobTitle) => {
        try {
            const response = await axios.get(`http://127.0.0.1:8000/api/fetchJD/${jobTitle}`);
            setSelectedJob(response.data);
        } catch (error) {
            console.error('Error fetching job details:', error);
        }
    };

    const handleEmailChange = (event) => {
        setRecruiterEmail(event.target.value);
    };

    return (
        <div className="view-applicants-container">
            <h2>Job Openings</h2>
            <div className="email-input">
                <input type="email" value={recruiterEmail} onChange={handleEmailChange} placeholder="Enter Recruiter Email" />
                <button onClick={() => setRecruiterEmail(recruiterEmail)}>Submit</button>
            </div>
            <div className="job-openings">
                {loading ? (
                    <p>Loading...</p>
                ) : error ? (
                    <p>{error}</p>
                ) : (
                    <ul>
                        <h2>Jobs Created</h2>
                        {jobs.map((job) => (
                            <li key={job._id} onClick={() => handleJobClick(job.table_name, job.jobTitle)} className="job-item">
                                {job.jobTitle} - {job.companyName}
                            </li>
                        ))}
                    </ul>
                )}
            </div>

            {selectedJob && (
                <div className="selected-job">
                    <h2>Applicants</h2>
                    {selectedJob.length > 0 ? (
                        <ul className="applicant-list">
                            {selectedJob.map((applicant, index) => (
                                <li key={index} className="applicant-item">
                                    <p><strong>Name:</strong> {applicant.name}</p>
                                    <p><strong>Email:</strong> {applicant.email}</p>
                                    <p><strong>JD Score:</strong> {applicant.Jdscore}</p>
                                    <p><strong>Skills Score:</strong> {applicant.skillsScore}</p>
                                </li>
                            ))}
                        </ul>
                    ) : (
                        <p>No applicants have applied yet.</p>
                    )}
                </div>
            )}
        </div>
    );
}

export default ViewApplicants;
