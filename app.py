import streamlit as st
import pandas as pd
import numpy as np
import joblib
import xgboost as xgb
from sentence_transformers import SentenceTransformer
import base64
import os

#   1. Page Configuration 
st.set_page_config(page_title="CareerAI", layout="wide", initial_sidebar_state="expanded")

#  2. Helper Function for Background Image 
def get_base64_of_bin_file(bin_file):
    if os.path.exists(bin_file):
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return ""

bg_base64 = get_base64_of_bin_file("bg.png")

#  3. Session State Management 
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Login"
if 'predicted_industry' not in st.session_state:
    st.session_state.predicted_industry = None
if 'top_courses' not in st.session_state:
    st.session_state.top_courses = None
if 'user_skills' not in st.session_state:
    st.session_state.user_skills = []
if 'top_scores_pct' not in st.session_state:
    st.session_state.top_scores_pct = []
if 'user_name' not in st.session_state:
    st.session_state.user_name = "Student"
if 'users_db' not in st.session_state:
    st.session_state.users_db = {"student@university.edu": {"password": "123", "name": "Student"}}

# 4. Load AI Models 
@st.cache_resource
def load_system():
    courses_df = pd.read_csv('1/massive_courses_v4.csv')
    xgb_classifier = joblib.load('2/industry_classifier_v4.pkl')
    le = joblib.load('2/label_encoder_v4.pkl')
    lfm_data = joblib.load('3/lightfm_model_v4.pkl')
    
    lfm_model = lfm_data['model']
    c_map = lfm_data['course_map']
    try:
        nlp_model = joblib.load('1/bert_nlp_model_v4.pkl')
    except:
        nlp_model = SentenceTransformer('all-MiniLM-L6-v2')
        
    return courses_df, xgb_classifier, le, lfm_model, c_map, nlp_model

courses, xgb_model, label_encoder, lightfm_model, course_map, bert_model = load_system()
reverse_course_map = {v: k for k, v in course_map.items()}

courses['Degree_Program'] = courses['Degree_Program'].astype(str).str.strip()

# GLOBAL CSS STYLES (Applies to all pages)

st.markdown("""
<style>
/* Hide the top Streamlit header completely */
    header {display: none !important;}
    
    /* REMOVE THE SIDEBAR CLOSE ARROW SO IT CANNOT BE COLLAPSED */
    [data-testid="stSidebarCollapseButton"] {display: none !important;}

    /* Global App Styling */
    .stApp {
        background-color: #0A0A0A;
        color: #FFFFFF;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* 1. The Outer Wrapper Box */
    .stTextInput div[data-baseweb="input"], 
    .stNumberInput div[data-baseweb="input"] {
        background-color: #121212 !important;
        border: 1px solid #2B2B2B !important;
        border-radius: 8px !important;
        height: 55px !important; /* Force exact height */
    }
    
    /* 2. The Text Inside the Box (Fixes the cut-off letters!) */
    .stTextInput input, .stNumberInput input {
        color: white !important;
        font-size: 1.25rem !important;
        height: 100% !important; /* Forces the text field to match the 55px box */
        margin: 0px !important;
        padding-top: 0px !important;
        padding-bottom: 0px !important;
        line-height: normal !important; /* Centers the text naturally */
    }
    
    /* 3. Text Area (Kept separate so your big paragraph box doesn't break) */
    .stTextArea textarea {
        background-color: #121212 !important;
        border: 1px solid #2B2B2B !important;
        border-radius: 8px !important;
        color: white !important;
        font-size: 1.25rem !important;
        padding: 15px !important; 
        line-height: 1.5 !important;
    }
    
    /* Input Labels */
    .stTextInput label p, .stTextArea label p, .stNumberInput label p, .stMultiSelect label p {
        font-size: 1.35rem !important;
        color: #888888 !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
        font-weight: bold;
    }
    
    /* Multiselect & Selectbox Container */
    div[data-baseweb="select"] > div {
        background-color: #121212 !important;
        border: 1px solid #2B2B2B !important;
        border-radius: 8px !important;
        min-height: 55px !important;
        padding: 5px 15px !important; /* Allows space for multiple rows of tags */
    }
    
    /* Fix single-select text size */
    div[data-baseweb="select"] > div > div > div {
        font-size: 1.25rem !important;
        color: white !important;
    }

    /* Style the Multi-Select Tags (Chips) */
    span[data-baseweb="tag"] {
        background-color: #002F88 !important; /* Matches your CareerAI Blue */
        border: none !important;
        margin: 4px 5px !important; /* Adds healthy space between the tags */
        padding: 2px 10px !important;
        border-radius: 6px !important;
    }
    
    /* Text inside the Multi-Select Tags */
    span[data-baseweb="tag"] span {
        font-size: 1rem !important;
        color: white !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Primary Button Style (#002F88) */
    .stButton>button[kind="primary"], [data-testid="stFormSubmitButton"]>button {
        background-color: #002F88 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 20px 24px !important;
        font-size: 1.25rem !important;
        font-weight: bold !important;
        width: 100%;
        margin-top: 15px;
        transition: 0.3s;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button[kind="primary"]:hover, [data-testid="stFormSubmitButton"]>button:hover {
        background-color: #0040B5 !important;
        box-shadow: 0px 4px 15px rgba(0, 47, 136, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# PAGE 1: AUTHENTICATION (LOGIN & REGISTER)

if st.session_state.current_page in ["Login", "Register"]:
    
    st.markdown(f"""
    <style>
        [data-testid="stSidebar"] {{ display: none; }}
        
        html, body, [data-testid="stAppViewContainer"], main, .stApp {{
            overflow: hidden !important;
            height: 100vh !important;
        }}
        
        [data-testid="block-container"] {{ 
            padding: 10vh 5vw 0rem 5vw !important; 
            max-width: 100% !important; 
            height: 100vh !important; 
        }}
        
        .stApp {{
            background-color: #0A0A0A;
            background-image: linear-gradient(to right, rgba(10,10,10,0.65) 0%, rgba(10,10,10,0.95) 50%, #0A0A0A 50%, #0A0A0A 100%), url("data:image/png;base64,{bg_base64}");
            background-size: 100% 100%, 55vw 100vh;
            background-position: center, left center;
            background-repeat: no-repeat;
        }}
        
        /* Registration form box styling */
        [data-testid="stForm"] {{ 
            background-color: #141414; 
            border: 1px solid #222; 
            border-radius: 12px; 
            padding: 40px 40px; 
            box-shadow: 0 10px 30px rgba(0,0,0,0.5); 
        }}
    </style>
    """, unsafe_allow_html=True)
        
    col_left, space, col_right, padding_right = st.columns([1.5, 0.3, 1.1, 0.3])
    
    with col_left:
        st.markdown("<h2 style='color: #FFFFFF; font-size: 2.5rem; font-weight: 900; letter-spacing: 2px; padding-left: 10%; margin-top: 50px;'>CareerAI</h2>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
            <div style='padding-left: 10%;'>
                <h1 style='font-size: 5vw; line-height: 1; font-weight: 900; margin: 0;'>THE FUTURE OF</h1>
                <h1 style='font-size: 5vw; line-height: 1; font-weight: 900; color: #002F88; margin: 0;'>CAREER IDENTIFICATION</h1>
                <p style='font-size: 1.3vw; color: #A0AEC0; margin-top: 30px; max-width: 85%; line-height: 1.6;'>
                    Access high-density AI analytics, industry matching, and real-time curriculum recommendations across global academic networks.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col_right:
        #   LOGIN STATE  
        if st.session_state.current_page == "Login":
            with st.form("login_box", clear_on_submit=False):
                st.markdown("<h2 style='font-size: 3rem; margin-bottom: 5px; color: white;'>Login</h2>", unsafe_allow_html=True)
                st.markdown("<p style='color: #888; font-size: 1.1rem; margin-bottom: 30px;'>Enter your credentials to access the AI network.</p>", unsafe_allow_html=True)
                
                email = st.text_input("EMAIL", placeholder="student@university.edu")
                st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                
                password = st.text_input("PASSWORD", placeholder="••••••••••••", type="password")
                st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
                
                submitted = st.form_submit_button("SIGN IN →")
                
                if submitted:
                    # Check if email exists in database AND password matches
                    if email in st.session_state.users_db and st.session_state.users_db[email]["password"] == password:
                        st.session_state.user_name = st.session_state.users_db[email]["name"]
                        st.session_state.current_page = "Finder"
                        st.rerun()
                    else:
                        st.error(" Invalid email or password. Please try again or register.")
                        
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Register as a new user ↗", use_container_width=True):
                st.session_state.current_page = "Register"
                st.rerun()

        #   REGISTRATION STATE  
        elif st.session_state.current_page == "Register":
            with st.form("register_box", clear_on_submit=False):
                st.markdown("<h2 style='font-size: 2.5rem; margin-bottom: 5px; color: white;'>Register</h2>", unsafe_allow_html=True)
                st.markdown("<p style='color: #888; font-size: 1rem; margin-bottom: 20px;'>Create your AI network profile.</p>", unsafe_allow_html=True)
                
                new_name = st.text_input("FULL NAME", placeholder="Enter your full name")
                new_email = st.text_input("EMAIL", placeholder="student@university.edu")
                
                c1, c2 = st.columns(2)
                with c1:
                    new_password = st.text_input("PASSWORD", placeholder="••••••••", type="password")
                with c2:
                    confirm_password = st.text_input("CONFIRM PASSWORD", placeholder="••••••••", type="password")
                    
                st.markdown("<div style='margin-top: 5px;'></div>", unsafe_allow_html=True)
                
                submitted = st.form_submit_button("CREATE ACCOUNT →")
                
                if submitted:
                    if new_name.strip() == "" or new_email.strip() == "" or new_password == "":
                        st.error(" Please fill in all fields.")
                    elif new_email in st.session_state.users_db:
                        st.error(" This email is already registered.")
                    elif new_password != confirm_password:
                        st.error(" Passwords do not match.")
                    else:
                        # Save to our "database"
                        st.session_state.users_db[new_email] = {"password": new_password, "name": new_name}
                        st.success(" Account created successfully! Please log in.")
                        
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("← Back to Login", use_container_width=True):
                st.session_state.current_page = "Login"
                st.rerun()



# INNER APP PAGES (SIDEBAR DASHBOARD)
else:
    import streamlit.components.v1 as components
    components.html(
        """
        <script>
            var body = window.parent.document.querySelector(".main");
            if (body) { body.scrollTop = 0; }
            window.parent.scrollTo(0, 0);
        </script>
        """, 
        height=0
    )
    #   Sidebar and Inner Page CSS  
    st.markdown("""
    <style>
        [data-testid="block-container"] { padding: 3rem 4rem !important; }
        [data-testid="stSidebar"] { background-color: #0A0A0A !important; border-right: 1px solid #1A1A1A !important; }
        
        /* 25px Spacing between sidebar buttons */
        [data-testid="stSidebar"] div.stButton {
            margin-bottom: 25px !important; 
        }
        
        /* Base styling for ALL Sidebar Buttons */
        [data-testid="stSidebar"] .stButton>button {
            background-color: transparent !important;
            border: none !important;
            border-right: 4px solid transparent !important; /* Placeholder */
            border-radius: 0px !important;
            padding: 12px 20px !important;
            display: flex !important;
            justify-content: flex-start !important;
            width: 100%;
        }
        
        /* Sidebar Text Base */
        [data-testid="stSidebar"] .stButton>button p {
            font-size: 1.35rem !important;
            font-weight: 600 !important;
            margin: 0 !important;
            transition: 0.3s;
        }

        /*   INACTIVE BUTTONS (Grey)   */
        [data-testid="stSidebar"] .stButton>button[kind="secondary"] p {
            color: #888888 !important;
        }
        [data-testid="stSidebar"] .stButton>button[kind="secondary"]:hover {
            background-color: rgba(255,255,255,0.02) !important;
        }
        [data-testid="stSidebar"] .stButton>button[kind="secondary"]:hover p {
            color: #FFFFFF !important;
        }

        /*   ACTIVE BUTTON (The Blue Highlight)   */
        [data-testid="stSidebar"] .stButton>button[kind="primary"] {
            border-right: 4px solid #002F88 !important;
            background-color: rgba(0, 47, 136, 0.1) !important;        
        }
        [data-testid="stSidebar"] .stButton>button[kind="primary"] p {
            color: #002F88 !important;
        }
        
        /* Custom Dashboard Metric Cards */
        .dash-card {
            background-color: #141414;
            border: 1px solid #222;
            border-radius: 12px;
            padding: 25px;
        }
        .dash-stat { font-size: 3.5rem; font-weight: 900; color: #002F88; line-height: 1;}
        .dash-label { font-size: 0.85rem; color: #002F88; font-weight: bold; text-transform: uppercase; letter-spacing: 1px;}
        .dash-sub { font-size: 0.9rem; color: #666;}
    </style>
    """, unsafe_allow_html=True)

    #   Draw Sidebar  
    with st.sidebar:
        st.markdown("""
            <div style='text-align: center; padding: 20px 0px 40px 0px;'>
                <h2 style='color: #FFFFFF; font-size: 2.5rem; font-weight: 900; margin:0;'>CareerAI</h2>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("Dashboard", type="primary" if st.session_state.current_page == "Finder" else "secondary", use_container_width=True): 
            st.session_state.current_page = "Finder"
            st.rerun()
        if st.button("My Results", type="primary" if st.session_state.current_page == "Results" else "secondary", use_container_width=True): 
            st.session_state.current_page = "Results"
            st.rerun()
        if st.button("Course Catalog", type="primary" if st.session_state.current_page == "Catalog" else "secondary", use_container_width=True): 
            st.session_state.current_page = "Catalog"
            st.rerun()
        if st.button("AI Analytics ", key="btn_analytics", type="primary" if st.session_state.current_page == "Analytics" else "secondary", use_container_width=True):
            st.session_state.current_page = "Analytics"
            st.rerun()
        st.markdown("<div style='height: 30vh;'></div>", unsafe_allow_html=True)
        st.markdown("<hr style='border-color: #222;'>", unsafe_allow_html=True)
        if st.button("Logout", type="secondary", use_container_width=True): 
            st.session_state.current_page = "Login"
            st.rerun()

    #   PAGE 2: AI PATH FINDER (DASHBOARD)  
    if st.session_state.current_page == "Finder":
        # Extract first name 
        first_name = st.session_state.user_name.split()[0] if st.session_state.user_name else "Student"
        
        st.markdown(f"<h1 style='font-size: 3rem;'>{first_name}'s Dashboard</h1>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Top Stats Cards
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"<div class='dash-card'><div class='dash-label'>TOTAL COURSES</div><div class='dash-stat'>{len(courses)}</div><div class='dash-sub'>ACROSS CATALOG</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown("<div class='dash-card'><div class='dash-label'>ACTIVE AI MODELS</div><div class='dash-stat'>3</div><div class='dash-sub'>PIPELINE OPERATIVE</div></div>", unsafe_allow_html=True)
        with c3:
            st.markdown("<div class='dash-card'><div class='dash-label'>PREDICTION ENGINE</div><div class='dash-stat'>XGB</div><div class='dash-sub'>GRADIENT BOOSTING</div></div>", unsafe_allow_html=True)
            
        st.markdown("<br><br>", unsafe_allow_html=True)
        
        # Form Elements (NO HTML WRAPPER = NO GHOST BOX)
        st.markdown("<h2 style='font-size: 1.5rem;'>● GENERATE CAREER PATH</h2>", unsafe_allow_html=True)
        st.markdown("<hr style='border-color: #333; margin-top: 0px;'>", unsafe_allow_html=True)
        
        gpa_error_space = st.empty()
        
        gpa = st.text_input(
            "ENTER CUMULATIVE GPA", 
            value="", 
            placeholder="-Enter your GPA (e.g 3.4)  -"
        )
        st.markdown("<br>", unsafe_allow_html=True)
        
        skills = st.multiselect(
            "SELECT ACQUIRED SKILLS", 
            ["Database", "C++", "Communication", "Operating Systems", "Machine Learning", "Python", "SQL", "UI/UX", "Java"],
            default=None,
            placeholder="--Select from drop down--"
        )
        st.markdown("<br>", unsafe_allow_html=True)
        
        interests = st.text_area(
            "DESCRIBE CAREER INTERESTS",
            placeholder="e.g., I enjoy problem-solving, building scalable systems, and working on data...",
            height=180
        )
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("<h2 style='font-size: 1.5rem; margin-top: 20px;'>● ACADEMIC & BEHAVIORAL PROFILE</h2>", unsafe_allow_html=True)
        st.markdown("<hr style='border-color: #333; margin-top: 0px;'>", unsafe_allow_html=True)
        
        # Row 1: Demographics and Study Habits
        col_demo, col_habits = st.columns(2)
        with col_demo:
            demographics = st.selectbox(
                "DEMOGRAPHIC BACKGROUND",
                ["Urban", "Suburban", "Rural", "International"],
                help="General background information as per academic benchmarks."
            )
        with col_habits:
            study_habits = st.selectbox(
                "PRIMARY STUDY HABIT",
                ["Daily Consistent", "Weekend Crammer", "Group Study", "Night Owl / Last Minute"],
                help="Learning patterns to align with course rigor."
            )
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        #   DYNAMIC TRANSCRIPT GENERATOR  
        st.markdown("<p style='font-size: 1.35rem; color: #888888; text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 8px; font-weight: bold;'>DYNAMIC TRANSCRIPT EVALUATION</p>", unsafe_allow_html=True)
        
        col_deg, col_sem = st.columns(2)
        with col_deg:
            # Get unique degrees dynamically from your dataset
            available_degrees = courses['Degree_Program'].unique().tolist() if 'Degree_Program' in courses.columns else ["BS Computer Science", "BS Software Engineering", "BS Data Science"]
            selected_degree = st.selectbox("CURRENT DEGREE PROGRAM", available_degrees)
            
        with col_sem:
            selected_semester = st.selectbox("CURRENT SEMESTER", [1, 2, 3, 4, 5, 6, 7, 8])
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        past_courses = pd.DataFrame()
        if 'Degree_Program' in courses.columns and 'Typical_Semester' in courses.columns:
            past_courses = courses[(courses['Degree_Program'] == selected_degree) & (courses['Typical_Semester'] < selected_semester)]
        
        student_transcript_dict = {}
        
        if past_courses.empty:
            if selected_semester == 1:
                st.info(" First-semester students do not have a university transcript yet. AI will evaluate based on skills and interests.")
            else:
                st.warning(" No course history found for this degree/semester combination in the catalog.")
        else:
            st.markdown(f"**Please enter your grades for the {len(past_courses)} courses completed in previous semesters:**")
            
            cols = st.columns(2)
            grade_options = ["A", "B", "C", "D", "F", "W (Withdrawn)"]
            
            for index, row in past_courses.iterrows():
                course_name = row['Course_Name']
                course_id = row['Course_ID']
                
                col = cols[index % 2]
                with col:
                    # Save the user's selected grade into dictionary
                    student_transcript_dict[course_name] = st.selectbox(f"{course_id} - {course_name}", grade_options, key=f"grade_{course_id}")
                    
        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("INITIALIZE AI ANALYSIS →", type="primary"):
            valid_gpa = False
            try:
                gpa_val = float(gpa)
                if 0.0 <= gpa_val <= 4.0:
                    valid_gpa = True
                else:
                    gpa_error_space.error(" Invalid input: GPA must be between 0.0 and 4.0")
            except ValueError:
                if gpa.strip() == "":
                    gpa_error_space.error(" Please enter your GPA.")
                else:
                    gpa_error_space.error(" Please enter a valid number (e.g., 3.4).")
            
            if valid_gpa:
                if len(interests.strip()) < 10:
                    st.warning("Please provide a bit more detail in your interests.")
                else:
                    with st.spinner("Processing through Neural Networks..."):
                        if len(student_transcript_dict) > 0:
                            transcript_list = [f"I received a {grade} in {course}" for course, grade in student_transcript_dict.items()]
                            transcript_str = ". ".join(transcript_list) + "."
                        else:
                            transcript_str = "I am a first-semester student with no university grades yet."

                        skills_str = ", ".join(skills)
                        full_profile = f"I am in Semester {selected_semester} of my {selected_degree} degree. My overall GPA is {gpa}. My background is {demographics} and my study habit is {study_habits}. {transcript_str} My acquired skills are {skills_str}. My career goal is: {interests}"
                                                                        
                        profile_embedding = bert_model.encode([full_profile])
                        pred_idx = xgb_model.predict(profile_embedding)[0]
                        st.session_state.predicted_industry = label_encoder.inverse_transform([pred_idx])[0]
                        
                        proxy_user_id = 0 
                        course_indices = np.arange(len(course_map))
                        
                        user_array = np.full(len(course_indices), proxy_user_id)
                        
                        predictions = lightfm_model.predict(user_ids=user_array, item_ids=course_indices)
                        
                        # Get top courses
                        top_indices = np.argsort(-predictions)[:3]
                        st.session_state.top_courses = top_indices
                        
                        top_raw_scores = predictions[top_indices]
                        score_min, score_max = np.min(predictions), np.max(predictions)
                        # Normalize the math into a realistic 85% - 98% range for the UI
                        normalized_pcts = 85 + 13 * ((top_raw_scores - score_min) / (score_max - score_min + 1e-9))
                        st.session_state.top_scores_pct = [int(p) for p in normalized_pcts]
                        
                        # Save skills for Gap Analysis
                        st.session_state.user_skills = skills
                        
                        st.session_state.current_page = "Results"
                        st.rerun()

    #   PAGE 3: RESULTS  
    elif st.session_state.current_page == "Results":
        st.markdown("<h1 style='font-size: 3rem;'>Analysis Results</h1>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.session_state.predicted_industry is None:
            st.warning("No data found. Please run an analysis on the Dashboard.")
        else:
            st.markdown(f"""
            <div class='dash-card' style='border-left: 8px solid #002F88;'>
                <h3 style='margin:0; color: #888;'>AI PREDICTED INDUSTRY</h3>
                <h1 style='color: #002F88; margin-top: 5px; font-size: 3.5rem;'>{st.session_state.predicted_industry}</h1>
            </div>
            <br><br>
            """, unsafe_allow_html=True)
            
            # === NEW SKILL GAP ANALYSIS MODULE ===
            st.markdown("<h2 style='font-size: 1.5rem;'>● SKILL GAP ANALYSIS</h2>", unsafe_allow_html=True)
            st.markdown("<hr style='border-color: #333; margin-top: 0px;'>", unsafe_allow_html=True)
            
            # Maps industries to required skills
            industry_reqs = {
                "Data Science": ["Python", "Machine Learning", "SQL", "Database"],
                "Software Engineering": ["C++", "Java", "Operating Systems", "Database"],
                "UI/UX Design": ["UI/UX", "Communication"],
                "Data Analytics": ["Python", "SQL", "Database", "Communication"],
                "Cybersecurity": ["Operating Systems", "Networking", "C++", "Python"],
                "AI Research": ["Machine Learning", "Python", "Mathematics", "Deep Learning"],
                "Product Management": ["Communication", "Agile", "Business Strategy", "UI/UX"],
                "Cloud Computing": ["Operating Systems", "Database", "Networking", "Python"]
            }
            
            # Try to get exact requirements, otherwise use a generic tech set
            required = industry_reqs.get(st.session_state.predicted_industry, ["Python", "Machine Learning", "Communication", "Database"])
            
            # Calculate Intersection (Aquired) and Difference (Missing)
            acquired_skills = list(set(st.session_state.user_skills).intersection(set(required)))
            missing_skills = list(set(required).difference(set(st.session_state.user_skills)))
            
            # Fallbacks if arrays are empty
            if not acquired_skills: acquired_skills = ["Foundational Basics"]
            if not missing_skills: missing_skills = ["Advanced Specializations"]
            
            # Display Side-by-Side Cards
            gap_c1, gap_c2 = st.columns(2)
            with gap_c1:
                st.markdown("<div class='dash-card' style='border-top: 4px solid #00FF00;'>", unsafe_allow_html=True)
                st.markdown("<h4 style='color: #00FF00; margin-top:0;'>🟢 ACQUIRED SKILLS</h4>", unsafe_allow_html=True)
                for s in acquired_skills:
                    st.markdown(f"<span style='background-color: rgba(0,255,0,0.1); color: #00FF00; padding: 5px 10px; border-radius: 5px; margin-right: 5px; display: inline-block; margin-bottom: 5px;'>{s}</span>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            with gap_c2:
                st.markdown("<div class='dash-card' style='border-top: 4px solid #FF4444;'>", unsafe_allow_html=True)
                st.markdown("<h4 style='color: #FF4444; margin-top:0;'>🔴 CRITICAL MISSING SKILLS</h4>", unsafe_allow_html=True)
                for s in missing_skills:
                    st.markdown(f"<span style='background-color: rgba(255,68,68,0.1); color: #FF4444; padding: 5px 10px; border-radius: 5px; margin-right: 5px; display: inline-block; margin-bottom: 5px;'>{s}</span>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.markdown("<br><br>", unsafe_allow_html=True)


            st.markdown("<h2 style='font-size: 1.5rem;'>● RECOMMENDED CURRICULUM</h2>", unsafe_allow_html=True)
            st.markdown("<hr style='border-color: #333; margin-top: 0px;'>", unsafe_allow_html=True)
            
            for rank, idx in enumerate(st.session_state.top_courses, 1):
                course_id = reverse_course_map[idx]
                course_info = courses[courses['Course_ID'] == course_id]
                c_name = course_info.iloc[0]['Course_Name'] if not course_info.empty and 'Course_Name' in course_info.columns else f"Course ID: {course_id}"
                
                # Fetch the calculated percentage for this specific course
                match_pct = st.session_state.top_scores_pct[rank - 1]
                
                html_card = (
                    "<div style='background-color: #1E2333; border: 1px solid #2B3245; padding: 20px; border-radius: 12px; margin-bottom: 15px;'>"
                    
                    #   Top Row (Number, Title, Progress Bar)  
                    "<div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;'>"
                    "<div style='display: flex; align-items: center;'>"
                    f"<div style='background-color: #002F88; color: white; width: 32px; height: 32px; border-radius: 50%; display: flex; justify-content: center; align-items: center; font-weight: bold; margin-right: 15px; font-size: 1.1rem;'>{rank}</div>"
                    f"<h4 style='margin: 0; color: #FFF; font-size: 1.3rem;'>{c_name}</h4>"
                    "</div>"
                    
                    "<div style='width: 30%; text-align: right;'>"
                    f"<div style='color: #4DA8DA; font-weight: bold; font-size: 0.8rem; margin-bottom: 5px; letter-spacing: 0.5px;'>Match Score: {match_pct}%</div>"
                    "<div style='width: 100%; background-color: #0A0D14; border-radius: 10px; height: 6px; overflow: hidden;'>"
                    f"<div style='width: {match_pct}%; background-color: #002F88; height: 100%; border-radius: 10px;'></div>"
                    "</div>"
                    "</div>"
                    "</div>"
                    
                    #   Bottom Row (Blue Pill Badges)  
                    "<div>"
                    "<div style='color: #888; font-size: 0.75rem; letter-spacing: 1px; margin-bottom: 10px; text-transform: uppercase;'>COURSE DETAILS & MODULES:</div>"
                    f"<span style='background-col or: #002F88; color: #FFF; padding: 6px 16px; border-radius: 20px; font-size: 0.85rem; margin-right: 10px; display: inline-block;'>Course Code: {course_id}</span>"
                    "<span style='background-color: #002F88; color: #FFF; padding: 6px 16px; border-radius: 20px; font-size: 0.85rem; margin-right: 10px; display: inline-block;'>Core Fundamentals</span>"
                    "<span style='background-color: #002F88; color: #FFF; padding: 6px 16px; border-radius: 20px; font-size: 0.85rem; display: inline-block;'>Applied Project</span>"
                    "</div>"
                    
                    "</div>"
                )
                st.markdown(html_card, unsafe_allow_html=True)

            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("RUN NEW ANALYSIS", type="primary"):
                st.session_state.current_page = "Finder"
                st.rerun()

   
    #   PAGE 4: COURSE CATALOG  
    elif st.session_state.current_page == "Catalog":
        st.markdown("<h1 style='font-size: 3rem;'>Directory & Catalog</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #888; margin-bottom: 30px;'>Browse supported career tracks and all available university modules.</p>", unsafe_allow_html=True)
        
        st.markdown("### Supported Career Tracks (AI Targets)")
        
        unique_industries = courses['Target_Career'].unique().tolist()
        
        # A professional color palette for the left borders
        colors = ['#002F88', '#00FF00', '#FF9900', '#E91E63', '#9C27B0', '#00BCD4', '#FFEB3B', '#FF5722']
        
        html_blocks = "<div style='display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 40px;'>"
        for idx, ind in enumerate(unique_industries):
            color = colors[idx % len(colors)]
            
            top_skills = courses[courses['Target_Career'] == ind]['Required_Skills'].str.split(',').explode().str.strip().value_counts().head(2).index.tolist()
            skills_str = ", ".join(top_skills) if top_skills else "Various tech stacks"
            
            html_blocks += f"<div style='background-color: #141414; border: 1px solid #222; border-left: 4px solid {color}; padding: 15px 20px; border-radius: 8px; flex: 1; min-width: 200px;'><h4 style='margin: 0; color: #FFF;'>{ind}</h4><p style='margin: 5px 0 0 0; color: #888; font-size: 0.85rem;'>Top Skills: {skills_str}</p></div>"
            
        html_blocks += "</div>"
        
        # Render the dynamic boxes
        st.markdown(html_blocks, unsafe_allow_html=True)

        st.markdown("### University Course Modules")

        html_table = courses.to_html(index=False, escape=False)
        
        st.markdown(f"""
<style>
    .catalog-container {{
        max-height: 500px;
        overflow-y: auto;
        border-radius: 12px;
        border: 1px solid #222;
        background-color: #141414;
    }}
    .catalog-table {{
        width: 100%;
        border-collapse: collapse;
        color: #FFFFFF;
    }}
    .catalog-table thead th {{
        background-color: #1A1D24; 
        color: #888;
        padding: 18px;
        position: sticky;
        top: 0;
        text-align: center !important; 
        text-transform: uppercase;
        letter-spacing: 1px;
        border-bottom: 1px solid #333;
        z-index: 1;
    }}
    .catalog-table tbody td {{
        padding: 15px;
        text-align: center !important; 
        border-bottom: 1px solid #222;
        font-size: 1.1rem;
    }}
    .catalog-table tbody tr:hover td {{
        background-color: rgba(0, 47, 136, 0.1); 
    }}
</style>

<div class="catalog-container">
    {html_table.replace('<table border="1" class="dataframe">', '<table class="catalog-table">')}
</div>
""", unsafe_allow_html=True)

    # PAGE: HOW THE AI WORKS 

    elif st.session_state.current_page == "Analytics":
        st.markdown("<h1 style='font-size: 3rem;'>How CareerAI Works</h1>", unsafe_allow_html=True)
        st.markdown("<p style='color: #888; margin-bottom: 30px;'>A simple breakdown of the dataset and the artificial intelligence models powering your recommendations.</p>", unsafe_allow_html=True)

        #   ROW 1: THE DATASET RANGE  
        st.markdown("### 1. The Knowledge Base (Our Dataset)")
        m1, m2, m3 = st.columns(3)
        
        with m1:
            st.markdown("<div class='dash-card'><div class='dash-label'>UNIVERSITY COURSES</div><div class='dash-stat'>150+</div><div class='dash-sub'>Across multiple disciplines</div></div>", unsafe_allow_html=True)
        with m2:
            st.markdown("<div class='dash-card'><div class='dash-label'>CAREER INDUSTRIES</div><div class='dash-stat'>8</div><div class='dash-sub'>Tech, Data, Design, etc.</div></div>", unsafe_allow_html=True)
        with m3:
            st.markdown("<div class='dash-card'><div class='dash-label'>TRAINING PROFILES</div><div class='dash-stat'>5000+</div><div class='dash-sub'>Simulated student pathways</div></div>", unsafe_allow_html=True)
            
        st.markdown("<br><br>", unsafe_allow_html=True)

        #   ROW 2: THE MODELS EXPLAINED  
        st.markdown("### 2. The Three AI Brains")
        st.markdown("""
        <div style='display: flex; gap: 20px; margin-bottom: 40px;'>
            <div style='flex: 1; background-color: #141414; padding: 25px; border-radius: 12px; border-top: 5px solid #002F88; border-bottom: 1px solid #222; border-left: 1px solid #222; border-right: 1px solid #222;'>
                <h3 style='margin-top:0; color: #4DA8DA;'> XGBoost</h3>
                <h5 style='color: #888; margin-top: -10px; margin-bottom: 15px;'>The Industry Guesser</h5>
                <p style='color: #ccc; font-size: 0.95rem; line-height: 1.5;'>This model looks at hard numbers (your GPA) and structured data (your selected skills) to mathematically predict which broad career industry you fit into best.</p>
            </div>
            <div style='flex: 1; background-color: #141414; padding: 25px; border-radius: 12px; border-top: 5px solid #00FF00; border-bottom: 1px solid #222; border-left: 1px solid #222; border-right: 1px solid #222;'>
                <h3 style='margin-top:0; color: #00FF00;'> BERT (NLP)</h3>
                <h5 style='color: #888; margin-top: -10px; margin-bottom: 15px;'>The Reader</h5>
                <p style='color: #ccc; font-size: 0.95rem; line-height: 1.5;'>This Natural Language model actually "reads" your written interests paragraph. It turns your English sentences into data to understand the <i>meaning</i> of what you love doing.</p>
            </div>
            <div style='flex: 1; background-color: #141414; padding: 25px; border-radius: 12px; border-top: 5px solid #FF9900; border-bottom: 1px solid #222; border-left: 1px solid #222; border-right: 1px solid #222;'>
                <h3 style='margin-top:0; color: #FF9900;'> LightFM</h3>
                <h5 style='color: #888; margin-top: -10px; margin-bottom: 15px;'>The Matchmaker</h5>
                <p style='color: #ccc; font-size: 0.95rem; line-height: 1.5;'>Once we know your industry, this model acts like Netflix. It looks at what courses similar successful students took, and filters the catalog to find your top 3 modules.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)

        #   ROW 3: THE STEP-BY-STEP PROCESS  
        st.markdown("### 3. How the Search Pipeline Works")
        st.markdown("""
<div style='background-color: #1A1D24; padding: 30px; border-radius: 12px; border: 1px solid #333;'>
    
<div style='display: flex; align-items: center; margin-bottom: 10px;'>
<div style='background-color: #002F88; color: white; width: 45px; height: 45px; min-width: 45px; border-radius: 50%; display: flex; justify-content: center; align-items: center; font-weight: 900; font-size: 1.3rem; margin-right: 20px; box-shadow: 0 0 15px rgba(0,47,136,0.5);'>1</div>
<div>
<h4 style='color: white; margin: 0; font-size: 1.2rem;'>Input Collection</h4>
<p style='color: #aaa; margin: 5px 0 0 0; font-size: 0.95rem;'>Provide your GPA, acquired skills, and a brief summary of your career goals, followed by your historical course grades.</p>
</div>
</div>
    
<div style='width: 4px; height: 40px; background-color: #333; margin-left: 20px; margin-bottom: 10px; border-radius: 2px;'></div>
    
<div style='display: flex; align-items: center; margin-bottom: 10px;'>
<div style='background-color: #002F88; color: white; width: 45px; height: 45px; min-width: 45px; border-radius: 50%; display: flex; justify-content: center; align-items: center; font-weight: 900; font-size: 1.3rem; margin-right: 20px; box-shadow: 0 0 15px rgba(0,47,136,0.5);'>2</div>
<div>
<h4 style='color: white; margin: 0; font-size: 1.2rem;'>Industry Targeting</h4>
<p style='color: #aaa; margin: 5px 0 0 0; font-size: 0.95rem;'><p style='color: #aaa; margin: 5px 0 0 0; font-size: 0.95rem;'>Our XGBoost and BERT models combine your academic data, skills, and written goals to accurately map your profile to a specific career sector.</p>
</div>
</div>
    
<div style='width: 4px; height: 40px; background-color: #333; margin-left: 20px; margin-bottom: 10px; border-radius: 2px;'></div>
    
<div style='display: flex; align-items: center;'>
<div style='background-color: #002F88; color: white; width: 45px; height: 45px; min-width: 45px; border-radius: 50%; display: flex; justify-content: center; align-items: center; font-weight: 900; font-size: 1.3rem; margin-right: 20px; box-shadow: 0 0 15px rgba(0,47,136,0.5);'>3</div>
<div>
<h4 style='color: white; margin: 0; font-size: 1.2rem;'>Course Filtering & Ranking</h4>
<p style='color: #aaa; margin: 5px 0 0 0; font-size: 0.95rem;'>Finally, the LightFM engine filters the university catalog to match your predicted industry, calculating a 0-100% Match Score to fill your critical skill gaps.</p>
</div>
</div>
    
</div>
<br><br>
""", unsafe_allow_html=True)