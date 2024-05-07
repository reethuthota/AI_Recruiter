import React, { useState } from 'react';
import JobOpenings from './JobOpenings';
import './ApplicantBase.css'; // Import CSS file for styling

function ApplicantBase({ onViewJobs, onSeeResults }) {
  const [currentPage, setCurrentPage] = useState('');

  const handleNavigation = (page) => {
    setCurrentPage(page);
  };

  const renderPage = () => {
    switch (currentPage) {
      case 'viewJobs':
        return <JobOpenings />;
      case 'seeResults':
        // Add logic for rendering the results page
        break;
      default:
        return (
          <div className="applicant-base">
            <h2>Welcome, Applicant!</h2>
            <div className="buttons">
              <button className="view-jobs-btn" onClick={() => handleNavigation('viewJobs')}>View Job Openings</button>
              <button className="see-results-btn" onClick={() => handleNavigation('seeResults')}>See Results</button>
            </div>
          </div>
        );
    }
  };

  return <div>{renderPage()}</div>;
}

export default ApplicantBase;
