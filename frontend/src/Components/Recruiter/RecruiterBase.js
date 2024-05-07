import React, { useState } from 'react';
import JobDescription from './JobDescription';
import ViewApplicants from './ViewApplicants'; // Import ViewApplicants component here
import './RecruiterBase.css'; // Import CSS file for styling

const RecruiterBase = ({ recruiterEmail }) => {
  const [currentPage, setCurrentPage] = useState(''); // Change initial state value

  const handleNavigation = (page) => {
    setCurrentPage(page);
  };

  const renderPage = () => {
    switch (currentPage) {
      case 'uploadJobDescription':
        return <JobDescription recruiterEmail={recruiterEmail} />;
      case 'viewApplicants':
        return <ViewApplicants recruiterEmail={recruiterEmail} />;
      default:
        return (
          <div className="recruiter-base-container">
            <h2 className="recruiter-dashboard-heading">Recruiter Dashboard</h2>
            <div className="recruiter-navigation">
              <button className="action-JD" onClick={() => handleNavigation('uploadJobDescription')}>
                Upload Job Description
              </button>
              <button className="action-VA" onClick={() => handleNavigation('viewApplicants')}>
                View Applicants
              </button>
            </div>
          </div>
        );
    }
  };

  return <div>{renderPage()}</div>;
};

export default RecruiterBase;
