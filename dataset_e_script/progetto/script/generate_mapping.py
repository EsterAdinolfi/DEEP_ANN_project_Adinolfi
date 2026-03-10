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

# ── 23 macro-aree OpinionQA (Santurkar et al., 2023) ───────────────────
# Nessuna categoria "Other": ogni domanda è assegnata a una delle 23 aree.
COARSE_TOPICS = [
    'Community health',
    'Corporations, tech, banks and automation',
    'Crime/security',
    'Discrimination',
    'Economy and inequality',
    'Education',
    'Future',
    'Gender & sexuality',
    'Global attitudes and foreign policy',
    'Healthcare',
    'Immigration',
    'Job/career',
    'Leadership',
    'News, social media, data, privacy',
    'Personal finance',
    'Personal health',
    'Political issues',
    'Race',
    'Relationships and family',
    'Religion',
    'Science',
    'Self-perception and values',
    'Status in life',
]

def assign_macro_area(key, question):
    """Assegna ogni domanda a una delle 23 macro-aree OpinionQA."""
    key_upper = key.upper()
    question_lower = question.lower()
    base = re.sub(r'_W\d+$', '', key_upper)
    # Prefisso alfabetico pulito (es. GUNRESPKIDSA → GUNRESPKIDS)
    m = re.match(r'^([A-Z_]+)', base)
    pfx = m.group(1).rstrip('_') if m else base

    # ── 1. Crime/security ──────────────────────────────────────────────
    if pfx.startswith('GUN') or pfx.startswith('REASONGUN') or pfx in {
        'CARRYGUN', 'NOCARRYGUN', 'DEFENDGUN', 'EVEROWN', 'EVERSHOT',
        'NEVEROWN', 'IMPREASONGUN', 'SHOOTFREQ', 'MARGUN', 'GROWUPGUN',
        'GROWUPVIOL', 'CRIMEVICTIM', 'CRIM_SENT', 'SAFECRIME',
        'WORLDDANGER', 'SOCIETY_GUNS'}:
        return 'Crime/security'
    # WORRY about crime/violence
    if pfx.startswith('WORRY') and any(w in question_lower for w in
            ['home broken', 'terrorist', 'violent crime', 'mass shooting']):
        return 'Crime/security'

    # ── 2. Education ───────────────────────────────────────────────────
    if pfx in {'ADMISSION', 'COLSPEECH', 'FREECOLL', 'HIGHED',
               'INSTN_CLGS', 'INSTN_K', 'SOCIETY_JBCLL'}:
        return 'Education'
    if pfx.startswith('HIGHEDWRNG'):
        return 'Education'

    # ── 3. Religion ────────────────────────────────────────────────────
    if pfx in {'EVOONE', 'EVOTWO', 'EVOTHREE', 'GODMORALIMP', 'GOODEVIL',
               'INSTN_CHR', 'RELIG_GOV', 'SOCIETY_RELG'}:
        return 'Religion'
    if pfx.startswith('EVOBIO') or pfx.startswith('EVOPERS'):
        return 'Religion'

    # ── 4. Global attitudes and foreign policy ─────────────────────────
    if pfx.startswith('GAP'):
        return 'Global attitudes and foreign policy'
    if pfx in {'ALLIES', 'FP_AUTH', 'PEACESTR', 'SUPERPWR', 'USEXCEPT',
               'USMILSIZ'}:
        return 'Global attitudes and foreign policy'

    # ── 5. Leadership ──────────────────────────────────────────────────
    if pfx in {'CONF', 'ELITEUNDMOD'}:
        return 'Leadership'
    if pfx.startswith(('ESSENBIZF', 'ESSENPOLF', 'BETTERBIZ', 'BETTERPOL',
                        'TRAITBIZMF', 'TRAITBIZWF', 'TRAITPOLMF', 'TRAITPOLWF',
                        'STYLE')):
        return 'Leadership'

    # ── 6. Gender & sexuality ──────────────────────────────────────────
    if pfx in {'GAYMARR', 'ORIENTATIONMOD', 'SOCIETY_SSM', 'SOCIETY_TRANS',
               'WOMENOBS', 'WOMENOPPS', 'WMNPRZ', 'SEENFEM', 'SEENMASC',
               'EXECCHF', 'POLCHF', 'EQUALBIZF', 'EQUALPOLF',
               'EASIERBIZF', 'EASIERPOLF'}:
        return 'Gender & sexuality'
    if pfx.startswith(('DIFF', 'BOYSF', 'GIRLSF', 'FEM', 'MASC',
                        'MOREWMN', 'WHYNOTBIZF', 'WHYNOTPOLF',
                        'AMNTWMN', 'IMPROVE', 'MAN', 'SPOUSESEX',
                        'TRANSGEND')):
        return 'Gender & sexuality'

    # ── 7. Discrimination ──────────────────────────────────────────────
    if pfx in {'PERSDISCR'}:
        return 'Discrimination'
    if pfx.startswith(('HARASS', 'HELPHURT', 'WHADVANT')):
        return 'Discrimination'

    # ── 8. Race ────────────────────────────────────────────────────────
    if pfx in {'RACATTN', 'IDIMPORT', 'INTRMAR', 'SOCIETY_RHIST',
               'SOCIETY_WHT', 'ETHNCMAJ', 'ETHNCMAJMOD'}:
        return 'Race'
    if pfx.startswith(('RACESURV', 'PROG_R')):
        return 'Race'

    # ── 9. Immigration ─────────────────────────────────────────────────
    if pfx in {'IL_IMM_PRI', 'IMMCOMM', 'IMMCULT', 'IMMIMPACT',
               'OPENIDEN', 'LEGALIMMIGAMT', 'LEGALIMG', 'UNIMMIGCOMM'}:
        return 'Immigration'

    # ── 10. Political issues ───────────────────────────────────────────
    if pfx in {'CANDEXP', 'CANMTCHPOL', 'CANQUALPOL', 'COMPROMISEVAL',
               'DEMDIRCT', 'DIFFPARTY', 'GOVAID', 'GOVPROTCT',
               'GOVWASTE', 'GOVT_ROLE', 'GOPDIRCT', 'LOCALELECT',
               'REPRSNTDEM', 'REPRSNTREP', 'VTRGHTPRIV', 'VTRS_VALS',
               'CNTRYFAIR', 'POSNEGGOV',
               'POORASSIST'}:
        return 'Political issues'
    if pfx.startswith(('POL', 'ELECT_', 'GOVSIZE', 'GOVRESP', 'GOVPRIO',
                        'GOVPRIORITY', 'POLICY', 'FEDSHARE', 'POP')):
        return 'Political issues'

    # ── 11. Economy and inequality ─────────────────────────────────────
    if pfx in {'BUSPROFIT', 'BILLION', 'CLASS', 'GLBLZE', 'NATDEBT',
               'WORKHARD', 'POOREASY', 'FIN_SITMOST', 'FIN_SITCOMM',
               'INSTN_LBRUN', 'AVGFAM', 'INDUSTRY'}:
        return 'Economy and inequality'
    if pfx.startswith(('ECON', 'INEQ', 'ECIMP', 'ECONFAIR')):
        return 'Economy and inequality'

    # ── 12. Personal finance ───────────────────────────────────────────
    if pfx in {'EARN', 'INC', 'INCFUTURE', 'SSCUT', 'SSMONEY',
               'WORRYBILL', 'WORRYRET', 'FIN_SIT', 'FIN_SITFUT',
               'FIN_SITGROWUP', 'LOYALTY'}:
        return 'Personal finance'
    if pfx.startswith(('DEBT', 'FINANCE', 'ELDFINANCEF', 'BENEFITS',
                        'WORRY')):
        # WORRY fallback (worry about debt, bills, retirement → personal finance)
        # Note: crime-related WORRY was already caught above
        return 'Personal finance'

    # ── 13. Job/career ─────────────────────────────────────────────────
    if pfx in {'GOODJOBS', 'JOBBENEFITS', 'JOBSECURITY', 'JOBSFUTURE',
               'JOBTRAIN'}:
        return 'Job/career'
    if pfx.startswith('WRKTRN'):
        return 'Job/career'

    # ── 14. Healthcare ─────────────────────────────────────────────────
    if pfx in {'GOVTHC', 'NOGOVTHC', 'SNGLPYER',
               'ABORTION', 'ABORTIONALLOW', 'ABORTIONRESTR'}:
        return 'Healthcare'

    # ── 15. Personal health ────────────────────────────────────────────
    if pfx in {'BIO', 'BLOODPR', 'FERTIL', 'NOWSMK_NHIS'}:
        return 'Personal health'
    if pfx.startswith(('EAT', 'FUD', 'WORRYG')):
        return 'Personal health'
    if pfx == 'G':
        return 'Personal health'

    # ── 16. Science ────────────────────────────────────────────────────
    if pfx in {'SC', 'FUTURE', 'PAST'}:
        return 'Science'
    if pfx.startswith(('BIOTECH', 'DNA', 'DNATEST', 'MED', 'SCI', 'SCM',
                        'SOLVPROB', 'Q', 'PQ', 'RQ')):
        return 'Science'

    # ── 17. Corporations, tech, banks and automation ───────────────────
    if pfx in {'INSTN_BNKS', 'INSTN_LGECRP', 'INSTN_TECHCMP',
               'FITTRACK', 'AUTOLKLY', 'AUTOWKPLC', 'SMARTAPP',
               'SMARTPHONE', 'HOMEIOT', 'POSNEGCO'}:
        return 'Corporations, tech, banks and automation'
    if pfx.startswith(('CARS', 'VOICE', 'DRONE', 'CAREGIV', 'ROBJOB',
                        'ROBWRK', 'ROBIMPACT', 'HIRING', 'FACE',
                        'HOMEASSIST')):
        return 'Corporations, tech, banks and automation'

    # ── 18. News, social media, data, privacy ──────────────────────────
    if pfx in {'ACCCHECK', 'BENEFITCO', 'BENEFITGOV', 'CONCERNCO',
               'CONCERNGOV', 'CONTROLCO', 'CONTROLGOV', 'DIGWDOG',
               'DISAVOID', 'ELECTFTGSNSINT', 'FCFAIR', 'LEAD',
               'MEDIALOYAL', 'NEWSPREFV', 'ONLINESOURCE', 'PRIVACYNEWS',
               'PRIVACYREG', 'PUBLICDATA', 'RESTRICTWHO', 'SECUR',
               'SEEK', 'SHARE', 'SMSHARE', 'SMSHARER', 'SNSUSE',
               'VIDOFT', 'TALKCMNSNSINT', 'TALKDISASNSINT',
               'UNDERSTANDCO', 'UNDERSTANDGOV', 'INSTN_MSCENT'}:
        return 'News, social media, data, privacy'
    if pfx.startswith(('ANONYMOUS', 'CONCERNGRP', 'CONTROLGRP', 'DATAUSE',
                        'DB', 'GOVREGV', 'INFO', 'MADEUP', 'MISINF',
                        'NEWSPROB', 'NEWS_PLATFORM', 'PP', 'PROFILE',
                        'PWMAN', 'RTBF', 'RTD', 'SMLIKES', 'SOCMEDIAUSE',
                        'SOURCESKEP', 'TRACKCO', 'TRACKGOV', 'WATCHDOG')):
        return 'News, social media, data, privacy'

    # ── 19. Future ─────────────────────────────────────────────────────
    if pfx in {'AGEMAJ', 'ELDCARE', 'OPTIMIST', 'POPPROB', 'ENVC'}:
        return 'Future'
    if pfx.startswith(('HAPPEN', 'FTRWORRY', 'FUTRCLASS', 'FUTR_',
                        'PREDICT')):
        return 'Future'

    # ── 20. Community health ───────────────────────────────────────────
    if pfx in {'COMATTACH', 'COMMYRS', 'COMTYPE', 'CITYSIZE',
               'GROWUPNEAR', 'GROWUPUSR', 'LIFELOC', 'FAMNEAR',
               'NEIGHKEYS', 'NEIGHKIDS', 'NEIGHKNOW',
               'SUBURBNEAR', 'BIGHOUSES', 'WANTMOVE', 'WILLMOVE',
               'TALK_CPS', 'FAVORS_CPS'}:
        return 'Community health'
    if pfx.startswith(('COMMIMP', 'NEIGHINTER', 'NEIGHSAME', 'LOCALPROB',
                        'HOOD_NHIS', 'MOVERURAL', 'MOVESUBURB', 'MOVEURBAN',
                        'VALUERURAL', 'VALUESUBURB', 'VALUEURBAN',
                        'PROBRURAL', 'PROBSUBURB', 'PROBURBAN',
                        'PARTICIPATE')):
        return 'Community health'

    # ── 21. Status in life (PRIMA di Relationships) ────────────────────
    if pfx in {'LIFEFIFTY'}:
        return 'Status in life'
    if pfx.startswith('SATLIFE'):
        return 'Status in life'

    # ── 22. Self-perception and values ─────────────────────────────────
    if pfx in {'HAPPYLIFE', 'PPLRESP', 'E'}:
        return 'Self-perception and values'
    if pfx.startswith(('MESUM', 'TRAITS', 'WORK', 'FEEL', 'SOCTRUST',
                        'PAR', 'PROBOFF', 'SUCCESSIMP')):
        return 'Self-perception and values'

    # ── 23. Relationships and family ───────────────────────────────────
    if pfx in {'ADKIDS', 'COHABDUR', 'ENG', 'CAREREL', 'FATHER',
               'HAVEKIDS', 'LWPSP', 'LWPT', 'MAR', 'MARRDUR',
               'MOTHER', 'REMARR', 'ROMRELDUR', 'ROMRELSER', 'SIB',
               'S'}:
        return 'Relationships and family'
    if pfx.startswith(('FAMSURV', 'MARRFAM', 'MARRYPREF')):
        return 'Relationships and family'

    # ── Keyword fallback (per chiavi non coperte sopra) ────────────────
    if any(w in question_lower for w in ['gun', 'firearm', 'shooting']):
        return 'Crime/security'
    if any(w in question_lower for w in ['college', 'university', 'higher education', 'school']):
        return 'Education'
    if any(w in question_lower for w in ['immigration', 'immigrant']):
        return 'Immigration'
    if any(w in question_lower for w in ['abortion']):
        return 'Healthcare'
    if any(w in question_lower for w in ['race ', 'racial', 'ethnic']):
        return 'Race'
    if any(w in question_lower for w in ['gender', 'women', 'feminist', 'sexual orientation']):
        return 'Gender & sexuality'
    if any(w in question_lower for w in ['religion', 'church', 'faith', 'god']):
        return 'Religion'
    if any(w in question_lower for w in ['robot', 'autonomous', 'automation', 'drone', 'driverless']):
        return 'Corporations, tech, banks and automation'
    if any(w in question_lower for w in ['social media', 'privacy', 'news', 'data collect']):
        return 'News, social media, data, privacy'
    if any(w in question_lower for w in ['economy', 'economic', 'inequality']):
        return 'Economy and inequality'
    if any(w in question_lower for w in ['family', 'marriage', 'children', 'spouse', 'partner']):
        return 'Relationships and family'
    if any(w in question_lower for w in ['community', 'neighborhood', 'neighbour']):
        return 'Community health'
    if any(w in question_lower for w in ['job', 'career', 'employment', 'workforce']):
        return 'Job/career'
    if any(w in question_lower for w in ['financial', 'income', 'debt', 'savings']):
        return 'Personal finance'
    if any(w in question_lower for w in ['health', 'medical', 'doctor']):
        return 'Personal health'
    if any(w in question_lower for w in ['science', 'scientific', 'scientist']):
        return 'Science'
    if any(w in question_lower for w in ['future', 'next 30 years', '2050', 'next 20 years']):
        return 'Future'
    if any(w in question_lower for w in ['foreign policy', 'military', 'allies', 'international']):
        return 'Global attitudes and foreign policy'
    if any(w in question_lower for w in ['leader', 'confidence', 'president handling']):
        return 'Leadership'
    if any(w in question_lower for w in ['discrimination', 'harassment']):
        return 'Discrimination'
    if any(w in question_lower for w in ['satisfaction', 'satisfied', 'quality of life']):
        return 'Status in life'
    if any(w in question_lower for w in ['value', 'identity', 'describe you']):
        return 'Self-perception and values'

    # Ultima rete di sicurezza: assegna a 'Self-perception and values'
    return 'Self-perception and values'

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