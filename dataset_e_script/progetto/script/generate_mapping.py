import os
import pandas as pd
import json
import re
from collections import defaultdict
from collections import defaultdict

# Path to human_resp folder
base_path = r'c:\Users\eadin\Desktop\DEEP_ANN_project_Adinolfi\dataset_e_script\human_resp'

# Function to clean topic
def clean_topic(key, all_bases, battery_groups):
    # Remove _W followed by digits
    base = re.sub(r'_W\d+', '', key)
    # If ends with number, always remove it
    if re.match(r'.*[0-9]$', base):
        return base[:-1]
    # If ends with lowercase letter, try to group
    if re.match(r'.*[a-z]$', base):
        potential_base = base[:-1]
        if len(battery_groups.get(potential_base, [])) > 0 or potential_base in all_bases:
            return potential_base
    return base

# Function to assign macro_area
def assign_macro_area(key, question):
    key_upper = key.upper()
    question_lower = question.lower()
    
    # Institutional Confidence (prioritize CONF prefix)
    if key_upper.startswith('CONF_') or key_upper.startswith('CONF') or 'institution' in question_lower or 'trust' in question_lower:
        return 'Institutional Confidence'
    
    # Guns
    if 'GUN' in key_upper or 'gun' in question_lower or 'firearm' in question_lower:
        return 'Guns'
    
    # Education
    if any(keyword in key_upper for keyword in ['ADMISSION', 'COLSPEECH', 'COLLEGE', 'FREECOLL', 'HIGHED', 'SCHOOL', 'STUDENT', 'EDUC']) or \
       any(keyword in question_lower for keyword in ['education', 'college', 'school', 'student', 'university', 'higher education']):
        return 'Education'
    
    # Politics (broad keywords)
    if any(keyword in key_upper for keyword in ['POL', 'ELECT', 'VOTE', 'PARTY', 'GOVERN', 'CONGRESS', 'SENAT', 'DEMOCRA', 'REPUBLICAN', 'GOVAID', 'GOVSIZE', 'GOVT_ROLE', 'GOVWASTE', 'GOVPROTCT', 'COMPROMISE']) or \
       any(keyword in question_lower for keyword in ['political', 'president', 'party', 'congress', 'senate', 'democrat', 'republican', 'election', 'vote', 'government']):
        return 'Politics'
    
    # Religion
    if any(keyword in key_upper for keyword in ['RELIG', 'GOD', 'EVOLU', 'EVO', 'GOODEVIL', 'FAITH', 'CHURCH']) or \
       any(keyword in question_lower for keyword in ['religion', 'church', 'faith', 'god', 'evolution', 'belief']):
        return 'Religion'
    
    # Economy (National) - differentiate from Personal Finance
    if ('ECON' in key_upper and 'SIT' in key_upper) or \
       any(keyword in key_upper for keyword in ['ECONFAIR', 'BUSPROFIT', 'BUSINESS', 'BILLION', 'CLASS', 'GAP', 'INEQ', 'INDUSTRY', 'GOODJOBS', 'JOBSECURITY', 'JOBTRAIN', 'HIRING', 'GLBLZE']) or \
       any(keyword in question_lower for keyword in ['economy', 'economic', 'business', 'inequality', 'class', 'industry', 'jobs', 'globalization']):
        return 'Economy'
    
    # Personal Finance
    if ('FIN' in key_upper and 'SIT' in key_upper) or \
       any(keyword in key_upper for keyword in ['DEBT', 'ELDFINANCE', 'FINANCE']) or \
       any(keyword in question_lower for keyword in ['personal finance', 'debt', 'savings', 'income', 'financial']):
        return 'Personal Finance'
    
    # Technology (expanded)
    if any(keyword in key_upper for keyword in ['TECH', 'AUTO', 'DRONE', 'CAR', 'DNA', 'DNATEST', 'DATAUSE', 'BIOTECH', 'BIO', 'ROBOT', 'INTERNET', 'ONLINE', 'DIGITAL', 'INFO', 'FITTRACK']) or \
       any(keyword in question_lower for keyword in ['technology', 'autonomous', 'drone', 'dna', 'data', 'biotech', 'robot', 'internet', 'digital', 'online', 'information']):
        return 'Technology'
    
    # Family (expanded)
    if any(keyword in key_upper for keyword in ['FAM', 'CHILD', 'PARENT', 'CAREGIV', 'ADKIDS', 'COHAB', 'MARRY', 'DIVORCE', 'FATHER', 'MOTHER', 'FERTIL', 'MARRDUR']) or \
       any(keyword in question_lower for keyword in ['family', 'child', 'parent', 'caregiver', 'marriage', 'divorce', 'children', 'father', 'mother', 'fertility']):
        return 'Family'
    
    # Health (expanded)
    if any(keyword in key_upper for keyword in ['HEALTH', 'ABORTION', 'BLOODPR', 'ELDCARE', 'MEDIC', 'HOSPITAL', 'DOCTOR']) or \
       any(keyword in question_lower for keyword in ['health', 'abortion', 'medical', 'hospital', 'doctor', 'healthcare']):
        return 'Health'
    
    # Race/Ethnicity
    if 'RACE' in key_upper or 'race' in question_lower or 'ethnic' in question_lower or 'racial' in question_lower:
        return 'Race/Ethnicity'
    
    # Immigration
    if any(keyword in key_upper for keyword in ['IMMIG', 'OPENIDEN', 'IMMCULT']) or \
       any(keyword in question_lower for keyword in ['immigration', 'immigrant', 'openness', 'border']):
        return 'Immigration'
    
    # Environment
    if 'ENV' in key_upper or 'environment' in question_lower or 'climate' in question_lower:
        return 'Environment'
    
    # Crime
    if 'CRIME' in key_upper or 'crime' in question_lower or 'criminal' in question_lower:
        return 'Crime'
    
    # Personal Security
    if 'WORRY' in key_upper or 'worry' in question_lower or 'safe' in question_lower or 'security' in question_lower:
        return 'Personal Security'
    
    # Quality of Life (expanded)
    if any(keyword in key_upper for keyword in ['LIFE', 'COMM', 'COMMUNITY', 'NEIGHBOR', 'CITY']) or \
       any(keyword in question_lower for keyword in ['quality of life', 'community', 'neighborhood', 'satisfaction']):
        return 'Quality of Life'
    
    # Housing
    if any(keyword in key_upper for keyword in ['HOUS', 'HOMEASSIST']) or \
       any(keyword in question_lower for keyword in ['house', 'housing', 'home']):
        return 'Housing'
    
    # Gender (expanded)
    if any(keyword in key_upper for keyword in ['WOMEN', 'GENDER', 'EQUAL', 'BOYS', 'GIRLS', 'HARASS', 'FEM']) or \
       any(keyword in question_lower for keyword in ['women', 'gender', 'equality', 'discrimination', 'harassment', 'feminine']):
        return 'Gender'
    
    # Foreign Policy
    if any(keyword in key_upper for keyword in ['ALLIES', 'FOREIGN', 'MILITARY', 'WAR']) or \
       any(keyword in question_lower for keyword in ['foreign policy', 'military', 'allies', 'international']):
        return 'Foreign Policy'
    
    # Default
    return 'Other'

# Dictionary to store mappings
mapping = {}

# Collect all base keys
all_bases = set()

# Collect battery groups for letters
battery_groups = defaultdict(list)

# First pass to collect bases and batteries
for folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder)
    if os.path.isdir(folder_path) and folder.startswith('American_Trends_Panel_W'):
        info_path = os.path.join(folder_path, 'info.csv')
        if os.path.exists(info_path):
            df = pd.read_csv(info_path)
            for _, row in df.iterrows():
                key = row['key']
                base = re.sub(r'_W\d+', '', key)
                all_bases.add(base)
                if re.match(r'.*[a-zA-Z]$', base):
                    potential_base = base[:-1]
                    suffix = base[-1]
                    battery_groups[potential_base].append(suffix)

# Now, second pass to create mapping
for folder in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder)
    if os.path.isdir(folder_path) and folder.startswith('American_Trends_Panel_W'):
        wave = folder.split('_W')[-1]
        info_path = os.path.join(folder_path, 'info.csv')
        if os.path.exists(info_path):
            df = pd.read_csv(info_path)
            for _, row in df.iterrows():
                key = row['key']
                question = row['question']
                topic = clean_topic(key, all_bases, battery_groups)
                macro_area = assign_macro_area(key, question)
                mapping[key] = {
                    'topic': topic,
                    'macro_area': macro_area,
                    'wave': wave
                }

# Post-process topics - keep only the initial uppercase letter sequence
for key in mapping:
    topic = mapping[key]['topic']
    original = topic
    
    # Find the longest prefix that is only uppercase letters and underscores
    # Stop at the first lowercase letter or digit
    match = re.match(r'^([A-Z_]+)', topic)
    if match:
        cleaned = match.group(1)
        # Remove trailing underscores
        cleaned = cleaned.rstrip('_')
        if len(cleaned) > 0:
            topic = cleaned
    
    mapping[key]['topic'] = topic if len(topic) > 0 else original

# Second pass - detect and merge single-letter variants
all_topics = [v['topic'] for v in mapping.values()]
topic_counter = {}
for t in all_topics:
    topic_counter[t] = topic_counter.get(t, 0) + 1

# Find variants: if we have BIOTECHA, BIOTECHB and they share a base BIOTECH
variants_to_base = {}
for topic in set(all_topics):
    if len(topic) > 1 and re.match(r'.*[A-Z]$', topic):
        potential_base = topic[:-1]
        # Check if there are multiple variants with this base
        similar = [t for t in topic_counter.keys() if t.startswith(potential_base) and len(t) == len(topic)]
        if len(similar) > 1:
            variants_to_base[topic] = potential_base

# Apply the variant mapping
for key in mapping:
    topic = mapping[key]['topic']
    if topic in variants_to_base:
        mapping[key]['topic'] = variants_to_base[topic]

# Save to JSON
output_path = os.path.join(base_path, 'question_mapping.json')
with open(output_path, 'w') as f:
    json.dump(mapping, f, indent=4)

print(f"Mapping saved to {output_path}")