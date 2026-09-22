import pandas as pd
import random
from faker import Faker

# Initialize Faker
fake = Faker()

# -------------------------------------------------------------
# STEP 0: Reference Mappings & Configuration
# -------------------------------------------------------------

# Mandated cross-company courses (Business Improvement #3)
COMPLIANCE_COURSES = [
    "Workplace Ethics", 
    "Cyber Security Awareness", 
    "Code of Conduct", 
    "Information Security"
]

# Decoupled dictionaries from previous step
course_categories = {
    "Commercial Leadership & Strategy": [
        "Advanced Sales", "Advanced CRM", "Basics of management", 
        "Strategic Account Management", "Sales Forecasting & Analytics", 
        "B2B Negotiation Strategies", "Executive Leadership", "Pipeline Management"
    ],
    "Frontline Sales & Client Acquisition": [
        "Sales Fundamentals", "Basics of CRM", "Negotiation", 
        "Cold Calling Techniques", "Product Knowledge Mastery", 
        "Objection Handling", "Effective Communication", "Time Management for Sales"
    ],
    "Data Science & Core R&D": [
        "Experimental Design", "Python", "Statistics", "Research Ethics", 
        "Data Visualization with R", "Machine Learning Foundations", 
        "Scientific Manuscript Writing", "Grant Proposal Writing"
    ],
    "R&D Strategy & Governance": [
        "R&D Strategic Planning", "Lab Portfolio Management", "Intellectual Property Law", 
        "Budgeting & Financial Control", "Advanced Team Leadership", 
        "Regulatory Compliance (FDA/ISO)", "Innovation Strategy", "Cross-Functional Collaboration"
    ],
    "Medical Commerce & Compliance": [
        "Medical Ethics & Compliance", "Pharmacology Basics", "Healthcare Industry Dynamics", 
        "Clinical Trial Interpretation", "Medical Sales Techniques", "HIPAA Regulations", 
        "Hospital Procurement Processes", "Patient-Centric Communication"
    ],
    "People Operations & Labour Law": [
        "Recruitment", "Labour Law", "Performance Management", 
        "Diversity, Equity & Inclusion (DEI)", "Conflict Resolution", 
        "HR Analytics", "Employee Onboarding & Retention", "Compensation & Benefits Design"
    ],
    "Laboratory Operations & Bio-Safety": [
        "Lab Safety Protocols (OSHA)", "Chemical Hazard Management", 
        "Equipment Calibration & Maintenance", "Sample Preparation Techniques", 
        "Quality Control Basics", "Microbiology Fundamentals", 
        "Electronic Lab Notebooks (ELN)", "Inventory & Supply Management"
    ],
    "Corporate Leadership & Team Operations": [
        "Foundations of Leadership", "Strategic Decision Making", "Change Management", 
        "Performance Coaching", "Project Management Foundations", 
        "Financial Literacy for Managers", "Effective Delegation", "Emotional Intelligence at Work"
    ],
    "Industrial Operations & Supply Chain": [
        "Lean Manufacturing & Six Sigma", "Supply Chain Optimization", "Operations Strategy", 
        "Factory Automation & Industry 4.0", "EHS (Environmental Health & Safety)", 
        "Quality Management Systems", "Capital Expenditure (CapEx) Planning", "Inventory Logistics"
    ]
}

job_role_to_category = {
    "Sales Executive": "Commercial Leadership & Strategy",
    "Sales Representative": "Frontline Sales & Client Acquisition",
    "Research Scientist": "Data Science & Core R&D",
    "Research Director": "R&D Strategy & Governance",
    "Healthcare Representative": "Medical Commerce & Compliance",
    "Human Resources": "People Operations & Labour Law",
    "Laboratory Technician": "Laboratory Operations & Bio-Safety",
    "Manager": "Corporate Leadership & Team Operations",
    "Manufacturing Director": "Industrial Operations & Supply Chain"
}

# -------------------------------------------------------------
# PIPELINE EXECUTION
# -------------------------------------------------------------

# STEP 2 — Read Dataset
df = pd.read_csv(r"D:\Shin\Programming\ADS_project\raw_data\employee_master.csv")

# STEP 3 — Create Empty List
training_records = []

# STEP 4 — Training ID
training_id = 1

# STEP 5 — Loop Through Employees
for _, employee in df.iterrows():
    
    # STEP 6 — Extract Required Fields
    employee_number = employee["EmployeeNumber"]
    job_role = employee["JobRole"]
    job_level = int(employee["JobLevel"])
    training_count = int(employee["TrainingTimesLastYear"])
    
    # Skip generation if employee had no training
    if training_count == 0:
        continue
        
    # STEP 7 — Find Correct Category and Domain Courses
    category = job_role_to_category.get(job_role, "Corporate Leadership & Team Operations")
    domain_courses = course_categories.get(category, [])
    
    # Business Improvement #3: Inject mandatory compliance into the available choices pool
    # Ensures a clean blend of core track training and workplace compliance rules
    pool_of_available_courses = domain_courses + COMPLIANCE_COURSES
    
    # Business Improvement #1: Select unique courses without replacement to prevent repetitions
    # If training count exceeds pool, fall back to choice with replacement
    if training_count <= len(pool_of_available_courses):
        selected_courses = random.sample(pool_of_available_courses, training_count)
    else:
        selected_courses = random.choices(pool_of_available_courses, k=training_count)
        
    # Business Improvement #2: Define Difficulty and Hours Mapping Based on JobLevel
    # Senior roles get progressively longer/harder tasks
    level_metrics = {
        1: {"difficulty": "Beginner", "min_hr": 2, "max_hr": 4},
        2: {"difficulty": "Intermediate", "min_hr": 3, "max_hr": 5},
        3: {"difficulty": "Intermediate", "min_hr": 4, "max_hr": 6},
        4: {"difficulty": "Advanced", "min_hr": 5, "max_hr": 7},
        5: {"difficulty": "Executive", "min_hr": 6, "max_hr": 8}
    }
    # Fallback bounds if level range falls out of standard bounds
    metrics = level_metrics.get(job_level, {"difficulty": "Intermediate", "min_hr": 3, "max_hr": 6})
    
    # STEP 8 — Repeat Based on TrainingTimesLastYear
    for course in selected_courses:
        
        # Override metadata if course picked belongs to global compliance category
        is_compliance = course in COMPLIANCE_COURSES
        record_category = "Corporate Compliance" if is_compliance else category
        record_difficulty = "Universal" if is_compliance else metrics["difficulty"]
        
        # STEP 10 — Generate Hours (JobLevel and type dependent)
        training_hours = random.randint(1, 3) if is_compliance else random.randint(metrics["min_hr"], metrics["max_hr"])
        
        # STEP 11 — Generate Completion Date
        completion_date = fake.date_between(start_date="-1y", end_date="today")
        
        # Business Improvement #4: Make scores role-dependent
        if job_level <= 2:
            score = random.randint(65, 95)
        else:
            score = random.randint(75, 100)
            
        # STEP 13 — Certification Outcomes (Weighted)
        certification = random.choices(["Pass", "Fail"], weights=[95, 5])[0]
        if certification == "Fail":
            score = random.randint(30, 64) # Force down score dynamically if failed
            
        # STEP 15 — Store Row
        training_records.append({
            "TrainingID": training_id,
            "EmployeeNumber": employee_number,
            "JobRole": job_role,
            "JobLevel": job_level,
            "CourseName": course,
            "CourseCategory": record_category,
            "Difficulty": record_difficulty,
            "TrainingHours": training_hours,
            "CompletionDate": completion_date,
            "AssessmentScore": score,
            "Status": certification
        })
        
        # STEP 16 — Increase TrainingID
        training_id += 1

# STEP 17 — Convert to DataFrame
training_df = pd.DataFrame(training_records)

# STEP 18 — Save CSV
training_df.to_csv(r"D:\Shin\Programming\ADS_project\raw_data\training.csv", index=False)
print(f"Successfully generated {len(training_df)} training history rows.")
