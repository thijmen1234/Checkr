import streamlit as st
import requests
import re
import spacy
from collections import defaultdict
import math
import os
import json
from datetime import datetime

# --- Streamlit UI Configuratie ---
try:
    st.set_page_config(layout="centered", page_title="Welkom bij CheckR!", page_icon="Checkr Logo.png")
except Exception as e:
    st.set_page_config(layout="centered", page_title="CheckR")
    st.error(f"Kon logo niet laden als page_icon: {e}. Zorg dat 'Checkr Logo.png' in de juiste map staat.")

# --- Custom CSS Styling ---
def apply_custom_styling():
    custom_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
        html, body, [class*="st-"], .stButton>button, 
        div[data-testid="stTextInput"] input,
        .stTextArea>div>div>textarea,
        .stSelectbox>div>div,
        .stRadio>div, .stCheckbox, .stMultiSelect > div[data-baseweb="select"] > div,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stSlider"]
        {
            font-family: 'Poppins', sans-serif;
        }
        .stButton>button {
            background-color: #2963E0; color: white; border: none; padding: 10px 18px; 
            border-radius: 8px; text-align: center; text-decoration: none; display: inline-block;
            font-size: 15px; font-weight: 500; margin: 4px 2px;
            transition: background-color 0.2s ease, transform 0.1s ease, box-shadow 0.2s ease;
        }
        .stButton>button:hover {
            background-color: #1F4EAD; transform: scale(1.03); box-shadow: 0 2px 5px rgba(0,0,0,0.15);
        }
        .stButton>button:active {
            background-color: #153A80; transform: scale(0.98);
        }
        div[data-testid="stTextInput"] input {
            background-color: #F0F2F6; border-radius: 25px; border: 1px solid #F0F2F6; 
            padding-top: 12px; padding-bottom: 12px; padding-left: 20px; padding-right: 20px;      
            line-height: 1.6; box-shadow: none; font-weight: 500;
        }
        div[data-testid="stTextInput"] input:focus {
            border-color: #2963E0 !important; background-color: #FFFFFF !important; 
            box-shadow: 0 0 0 2px rgba(41, 99, 224, 0.30) !important; 
        }
        div[data-testid="stTextInput"] input::placeholder {
            color: #A0AEC0; font-weight: 400; opacity: 1; 
        }
        div[data-baseweb="alert"][data-kind="info"] {
            background-color: #2963E0 !important; color: white !important;
            border-radius: 8pt !important; border: none !important; padding: 1rem !important;
        }
        div[data-baseweb="alert"][data-kind="info"] div[data-testid="stMarkdownContainer"] p,
        div[data-baseweb="alert"][data-kind="info"] div[data-testid="stMarkdownContainer"] li {
            color: white !important;
        }
        div[data-baseweb="alert"][data-kind="info"] svg {
            fill: white !important;
        }

        /* Centering for the entire main content area */
        .main .block-container {
            max-width: 700px; /* Adjust as needed */
            padding-left: 1rem;
            padding-right: 1rem;
            margin: auto; /* This centers the block-container */
        }
        
        /* Centering for specific elements if not covered by parent */
        [data-testid="stImage"] {
            display: block; /* Images are inline by default */
            margin-left: auto;
            margin-right: auto;
        }
        
        /* Centering for headers and general text */
        h1, h2, h3, h4, h5, h6, p, .stMarkdown, .stText, .stAlert {
            text-align: center;
        }

        /* Overrides for Streamlit specific elements that might not center by default */
        div[data-testid="stVerticalBlock"] {
            align-items: center;
        }

        /* Specific styling for the centered metric */
        .st-emotion-cache-1r6dm7w { /* This is a common container for st.metric, but can vary */
            display: flex;
            flex-direction: column;
            align-items: center; /* Centers horizontally */
            text-align: center;
        }
        [data-testid="stMetricValue"] {
            font-size: 2.5em !important; /* Groter bedrag */
            font-weight: 600 !important;
            color: #2963E0 !important; /* Kleur aanpassen indien gewenst */
            text-align: center; /* Zorg ervoor dat de waarde zelf ook gecentreerd is */
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.8em !important; /* Kleinere tekst "al bespaard" */
            color: #777777 !important;
            margin-top: -10px; /* Zorgt dat het dichter op het bedrag staat */
            text-align: center; /* Zorg ervoor dat het label zelf ook gecentreerd is */
        }

        /* Specific centering for buttons */
        div.stButton {
            display: flex;
            justify-content: center;
        }
        /* Specific centering for inputs/selects */
        div[data-testid="stTextInput"] div.st-emotion-cache-1c7y2kl,
        div[data-testid="stMultiSelect"] div.st-emotion-cache-1c7y2kl,
        div[data-testid="stNumberInput"] div.st-emotion-cache-1c7y2kl {
            display: flex;
            flex-direction: column;
            align-items: center;
            text-align: center;
        }
        /* Make multiselect dropdown wider if it gets too narrow by default due to centering */
        .stMultiSelect > div[data-baseweb="select"] > div {
            width: 100% !important; /* Ensures it fills available width within its centered container */
        }
        
        /* Override for dataframes which look bad when centered */
        div[data-testid="stDataFrame"] {
            display: block; /* Reset to default block display */
            margin-left: auto;
            margin-right: auto;
            text-align: left; /* Reset text align for content inside dataframe */
        }
        div[data-testid="stDataFrame"] table {
            margin-left: auto;
            margin-right: auto;
        }

        /* For column-based layouts, ensure content within columns can be centered too */
        .st-emotion-cache-nahz7x { /* st.columns container */
            display: flex;
            flex-direction: row;
            justify-content: center;
            align-items: flex-start;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

apply_custom_styling()

# --- SpaCy Model Laden ---
@st.cache_resource
def load_spacy_model():
    import spacy 
    model_name = "nl_core_news_sm"
    try:
        nlp_model = spacy.load(model_name)
        return nlp_model
    except OSError:
        st.info(f"SpaCy Nederlands model ({model_name}) wordt gedownload...")
        try:
            from spacy.cli import download 
            download(model_name)
            nlp_model = spacy.load(model_name)
            return nlp_model
        except SystemExit: 
            st.error(f"Fout bij downloaden van SpaCy model ({model_name}) via spacy.cli. "
                     f"Probeer het model handmatig te downloaden in je terminal: python -m spacy download {model_name}")
            return None 
        except Exception as e:
            st.error(f"Algemene fout bij downloaden/laden van SpaCy model ({model_name}): {e}")
            return None
    except Exception as e: 
        st.error(f"Onverwachte fout bij het laden van SpaCy model ({model_name}): {e}")
        return None

nlp = load_spacy_model()
if nlp is None:
    st.error("Het SpaCy NLP model kon niet geladen worden. De app kan niet volledig functioneren.")
    st.stop() 

# --- Constanten en Mappings ---
SUPERMARKETS = {"ah": "Albert Heijn", "aldi": "Aldi", "coop": "COOP", "dekamarkt": "DEKAMarkt", "dirk": "Dirk", "hoogvliet": "Hoogvliet", "jumbo": "Jumbo", "picnic": "PicNic", "plus": "Plus", "spar": "SPAR", "vomar": "Vomar"}
RETAILER_KEYS = set(SUPERMARKETS.keys())
RETAILER_NAMES = set(name.lower().strip() for name in SUPERMARKETS.values())
ALL_RETAILER_PREFIXES_SORTED = sorted(list(RETAILER_NAMES) + list(RETAILER_KEYS), key=len, reverse=True)

DIGIT_PATTERN = re.compile(r"\b\d+([\.,]?\d+)?\s*(l|ml|cl|kg|g|gr|st|stuk|x|per|gram|liter|ltr)?\b", re.IGNORECASE)
REMOVE_WORDS = {"stuk", "stuks", "st", "pieces", "piece", "per", "x", "gram", "kilogram", "liter", "ltr", "de", "het", "een", "en", "of"}
EXTRA_STOPWORDS = {"met", "zonder", "voor", "tegen", "bij", "van", "op", "onder", "boven", "door", "in", "uit", "tot", "tussen", "achter", "langs", "om", "over", "na", "aan", "binnen", "buiten", "sinds", "zoals", "vers", "verse"}
ALL_STOPWORDS = REMOVE_WORDS.union(EXTRA_STOPWORDS)

MANUAL_KEYWORD_MERGE_RULES = {
    frozenset({"go", "tan"}): "Go-Tan",
    frozenset({"earl", "grey"}): "Earl Grey", # Handmatig toegevoegd voor betere naamgeving
}

# --- GLOBALE SORTEERFUNCTIE ---
def get_product_sort_key(product_row_data):
    product_naam_lower = product_row_data.get("Naam", "").lower()
    is_missing = "(ontbreekt)" in product_naam_lower

    category_value = product_row_data.get("Categorie")
    if category_value is None:
        category_value = ""  # Standaard naar lege string om NoneType error te voorkomen

    # Sorteervolgorde:
    # 1. Ontbrekende items eerst (False komt voor True bij sorteren)
    # 2. Daarna op categorie (alfabetisch)
    return (not is_missing, category_value)

# --- Session State Initialisatie ---
if 'shopping_list' not in st.session_state: st.session_state.shopping_list = []
if 'all_products' not in st.session_state: st.session_state.all_products = []
if 'comparison_results' not in st.session_state: st.session_state.comparison_results = None
if 'active_swap' not in st.session_state: st.session_state.active_swap = None
if 'shopping_mode_active' not in st.session_state: st.session_state.shopping_mode_active = False
if 'active_shopping_list_details' not in st.session_state: st.session_state.active_shopping_list_details = None
if 'active_shopping_supermarket_name' not in st.session_state: st.session_state.active_shopping_supermarket_name = None
if 'skipped_products_log' not in st.session_state: st.session_state.skipped_products_log = []
if 'savings_log' not in st.session_state: st.session_state.savings_log = []

# Laad de besparingslog bij het opstarten van de app.
def load_savings_log(filename="savings_log.json"):
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                st.session_state.savings_log = json.load(f)
        except Exception as e:
            st.error(f"Fout bij laden van besparingslog: {e}")
            st.session_state.savings_log = []
    else:
        st.session_state.savings_log = []

load_savings_log() # Belangrijk: deze aanroep is nu vroeg in het script


# --- Functie Definities ---
def clean_name(name, search_term=None):
    name_to_process = name.lower()
    for prefix in ALL_RETAILER_PREFIXES_SORTED:
        if name_to_process.startswith(prefix + " ") or \
           name_to_process.startswith(prefix + ":") or \
           name_to_process == prefix:
            name_to_process = name_to_process[len(prefix):].lstrip(": ").strip()
            break 
    name_cleaned_after_supermarket = DIGIT_PATTERN.sub("", name_to_process)
    name_cleaned_after_supermarket = re.sub(r"[-/]", " ", name_cleaned_after_supermarket)
    name_cleaned_after_supermarket = re.sub(r"[^\w\s]", "", name_cleaned_after_supermarket)
    name_cleaned_after_supermarket = re.sub(r"\s+", " ", name_cleaned_after_supermarket).strip()
    final_cleaned_text = name_cleaned_after_supermarket
    if search_term:
        search_term_lower = search_term.lower().strip()
        if search_term_lower:
            temp_text_for_search_removal = final_cleaned_text
            variations_to_remove = sorted([search_term_lower + "jes", search_term_lower + "en", search_term_lower + "s", search_term_lower], key=len, reverse=True)
            for var in variations_to_remove:
                temp_text_for_search_removal = re.sub(r'\b' + re.escape(var) + r'\b', '', temp_text_for_search_removal, flags=re.IGNORECASE)
            if search_term_lower in temp_text_for_search_removal : 
                temp_text_for_search_removal = temp_text_for_search_removal.replace(search_term_lower, "")
            final_cleaned_text = re.sub(r"\s+", " ", temp_text_for_search_removal).strip()
    doc = nlp(final_cleaned_text)
    lemmas = [tok.lemma_.lower() for tok in doc if tok.lemma_.isalpha()]
    lemma_fix = {"varkens":"varken","kippen":"kip","runders":"runder","worsten":"worst","broden":"brood","aardappelen":"aardappel","groenten":"groente","fruiten":"fruit","broodjes":"brood","kips":"kip"}
    corrected_lemmas = [lemma_fix.get(lemma, lemma) for lemma in lemmas]
    filtered_words_set = {word for word in corrected_lemmas if word not in ALL_STOPWORDS}
    normalized_words_list = []
    for word in list(filtered_words_set): 
        if word.endswith("s") and len(word)>4 and (word[:-1] in nlp.vocab) and (word[:-1] not in ALL_STOPWORDS) :
            normalized_words_list.append(word[:-1])
        else:
            normalized_words_list.append(word)
    return sorted(list(set(normalized_words_list)))

@st.cache_data(ttl=3600)
def fetch_supermarket_data(url: str):
    try:
        resp = requests.get(url); resp.raise_for_status(); data = resp.json()
    except requests.exceptions.RequestException as e: st.error(f"Fout bij ophalen data: {e}."); return []
    products = []
    if 'skipped_products_log_fetch' not in st.session_state or st.session_state.get('fetch_url_last_run') != url:
        st.session_state.skipped_products_log_fetch = []
        st.session_state.fetch_url_last_run = url

    def parse_size_string_from_text(text_to_parse, product_name_for_context=""):
        qty, un = None, None
        if not text_to_parse: return qty, un
        text_lower = text_to_parse.lower()
        pers_match = re.search(r"(\d+)(?:\s*-\s*\d+)?\s*(persoon|personen|pers)\b", text_lower)
        if pers_match: qty = float(pers_match.group(1)); un = "personen"; return qty, un
        was_match = re.search(r"(\d+)\s*(wasbeurten|wasbeurt)\b", text_lower)
        if was_match: qty = float(was_match.group(1)); un = "wasbeurten"; return qty, un
        meter_match = re.search(r"(?:\d+[\.,]?\d*\s*)?(meter|m)\b", text_lower)
        if meter_match and not any(sub in text_lower for sub in ["ml", "mm", "gram", "mg", "komeet"]): un = "stuks"; qty = 1.0; return qty, un 
        vellen_match = re.search(r"(?:\d+[\.,]?\d*\s*)?(vellen|vel)\b", text_lower)
        if vellen_match: un = "stuks"; qty = 1.0; return qty, un
        kilo_match_with_number = re.search(r"(\d+[\.,]?\d*)\s*(?:per\s+)?(kg|kilo|kilogram|kg1)\b", text_lower)
        kilo_match_no_number = re.search(r"\b(?:los\s+per\s+|per\s+)?(kg|kilo|kilogram|kg1)\b", text_lower)
        if kilo_match_with_number:
            try: num_val = float(kilo_match_with_number.group(1).replace(",", ".")); qty = num_val * 1000; un = "gram"; return qty, un
            except ValueError: pass
        elif kilo_match_no_number: qty = 1000.0; un = "gram"; return qty, un
        size_unit_pattern_with_number = re.compile(r"(\d+[\.,]?\d*)\s*-?(ml|milliliter|milliliters|cl|centiliter|ctl|dl|deciliter|l|liter|lt|ltr|gram|g|gr|stuks?|stk|st|x|rollen|rol|zakjes|zakje|zakken|pak|pack|paar|set)\b",re.IGNORECASE)
        match_num = size_unit_pattern_with_number.search(text_to_parse) 
        if match_num:
            num_str = match_num.group(1).replace(",", "."); raw_unit_str_matched = match_num.group(2).lower()
            try:
                num_val = float(num_str)
                if raw_unit_str_matched in ["ml", "milliliter", "milliliters"]: un = "ml"; qty = num_val 
                elif raw_unit_str_matched in ["cl", "centiliter", "ctl"]: un = "ml"; qty = num_val * 10
                elif raw_unit_str_matched in ["dl", "deciliter"]: un = "ml"; qty = num_val * 100
                elif raw_unit_str_matched in ["l", "liter", "ltr", "lt"]: un = "ml"; qty = num_val * 1000 
                elif raw_unit_str_matched in ["kg", "kilogram", "kilo", "kg1"]: un = "gram"; qty = num_val * 1000 
                elif raw_unit_str_matched in ["gram", "g", "gr"]: un = "gram"; qty = num_val
                elif raw_unit_str_matched in ["stuks", "stuk", "stk", "st", "x", "rollen", "rol", "zakjes", "zakje", "zakken", "pak", "pack", "paar", "set"]: un = "stuks"; qty = num_val
            except ValueError: pass
            if qty is not None and un is not None: return qty, un
        per_unit_pattern = re.compile(r"^(?:per\s+)?(stuk|stuks|stk|bos|zak|net|pakket|fles|blik|doos|rol|zakje|pak|pack|paar|set)\b", re.IGNORECASE)
        match_unit_only = per_unit_pattern.search(text_lower)
        if match_unit_only: qty = 1.0; un = "stuks" ; return qty, un
        fallback_num_match = re.search(r"(\d+[\.,]?\d+)", text_to_parse)
        if fallback_num_match:
            try:
                num_val = float(fallback_num_match.group(1).replace(",","."))
                if num_val > 100 and not re.search(r"(?:stuk|stuks|stk|rol|zakje|pak|pack|paar|set|fles|blik|doos|bos|net|pakket|wasbeurt|persoon|meter|liter|ltr|lt|ml|cl|dl|kg|gram|gr)", text_lower, re.IGNORECASE):
                    qty = num_val; un = "gram"; return qty, un
            except ValueError: pass
        return qty, un

    for entry in data:
        store_code = entry.get("c", "").lower(); store_name = SUPERMARKETS.get(store_code, store_code.capitalize())
        items = entry.get("d") or next((v for v in entry.values() if isinstance(v, list)), [])
        for item_entry in items: 
            name = item_entry.get("n", "").strip(); raw_size_str = item_entry.get("s", "").strip(); price = item_entry.get("p")
            if not name or price is None: continue
            quantity, unit = None, None; source_of_size = "Onbekend/Default"; grootte_origineel_display = raw_size_str if raw_size_str else "(geen s-veld)"
            if raw_size_str:
                quantity, unit = parse_size_string_from_text(raw_size_str, name)
                if quantity is not None: source_of_size = "Grootte-veld ('s')"
            if quantity is None and unit is None: 
                quantity, unit = parse_size_string_from_text(name, name)
                if quantity is not None: source_of_size = "Productnaam ('n')"; grootte_origineel_display = f"(uit naam: {name})"
            if quantity is None and unit is None:
                quantity = 1.0; unit = "stuks"; source_of_size = "Default (1 stuks)"; grootte_origineel_display = f"Onbekend (standaard: 1 stuks) - Origineel 's': '{raw_size_str if raw_size_str else 'Leeg'}'"
            if quantity is not None and unit is not None: products.append({"Supermarkt": store_name, "Naam": name, "Grootte_Origineel": grootte_origineel_display, "Hoeveelheid": quantity, "Eenheid": unit, "Prijs": float(price)})
            else:
                log_entry = {"Naam": name, "Supermarkt": store_name, "Grootte_Origineel": raw_size_str, "Prijs": price, "Reden": "Naam of prijs ontbreekt?"}
                if log_entry not in st.session_state.skipped_products_log_fetch and len(st.session_state.skipped_products_log_fetch) < 500: st.session_state.skipped_products_log_fetch.append(log_entry)
    st.session_state.skipped_products_log = list(st.session_state.skipped_products_log_fetch) 
    return products

# --- Functie Definities (vervolg) ---
def add_to_shopping_list(category, desired_quantities_list, selected_products_for_category):
    filtered_desired_quantities = [dq for dq in desired_quantities_list if dq.get("Hoeveelheid", 0) > 0]
    if not filtered_desired_quantities: st.warning(f"Geen geldige hoeveelheden voor '{category}'."); return
    existing_item_index = next((i for i, item in enumerate(st.session_state.shopping_list) if item["Categorie"] == category), -1)
    new_item_data = {"Categorie": category, "DesiredQuantities": filtered_desired_quantities, "Producten": selected_products_for_category}
    qty_display_strs = []
    for dq in filtered_desired_quantities:
        is_stuks_dq = dq['Eenheid'].lower() == "stuks"
        hoeveelheid_main_val = dq['Hoeveelheid']
        if is_stuks_dq:
            hoeveelheid_display_str = str(int(hoeveelheid_main_val)) if hoeveelheid_main_val == int(hoeveelheid_main_val) else str(hoeveelheid_main_val)
        else:
            hoeveelheid_display_str = f"{float(hoeveelheid_main_val):.2f}".rstrip('0').rstrip('.') if '.' in f"{float(hoeveelheid_main_val):.2f}" else f"{float(hoeveelheid_main_val):.0f}"
            if hoeveelheid_display_str == "" and float(hoeveelheid_main_val) == 0 : hoeveelheid_display_str ="0"
        s = f"{hoeveelheid_display_str} {dq['Eenheid']}"
        tol_waarde = dq.get("TolerantieWaarde", 0)
        tol_eenheid = dq.get("TolerantieEenheid")
        if dq.get("TolerantiePercentage", 0) > 0 and tol_waarde > 0 and tol_eenheid:
            s += f" (±{int(tol_waarde)} {tol_eenheid})"
        qty_display_strs.append(s)
    if existing_item_index != -1: 
        st.session_state.shopping_list[existing_item_index] = new_item_data
        st.success(f"'{category}' ({', '.join(qty_display_strs)}) bijgewerkt.")
    else: 
        st.session_state.shopping_list.append(new_item_data)
        st.success(f"'{category}' ({', '.join(qty_display_strs)}) toegevoegd!")
    if 'comparison_results' in st.session_state: del st.session_state.comparison_results
    if 'active_swap' in st.session_state: del st.session_state.active_swap

def clear_shopping_list():
    st.session_state.shopping_list = []; 
    if 'comparison_results' in st.session_state: del st.session_state.comparison_results
    if 'active_swap' in st.session_state: del st.session_state.active_swap
    st.success("Boodschappenlijstje geleegd.")

def save_shopping_list(filename="shopping_list.json"):
    try:
        with open(filename, 'w', encoding='utf-8') as f: json.dump(st.session_state.shopping_list, f, ensure_ascii=False, indent=4)
        st.success(f"Lijst opgeslagen als '{filename}'")
    except IOError as e: st.error(f"Fout bij opslaan: {e}")

def load_shopping_list(filename="shopping_list.json"):
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f: st.session_state.shopping_list = json.load(f)
            if 'comparison_results' in st.session_state: del st.session_state.comparison_results
            if 'active_swap' in st.session_state: del st.session_state.active_swap
            st.success(f"Lijst geladen van '{filename}'")
        except Exception as e: st.error(f"Fout bij laden: {e}")
    else: st.warning(f"Bestand '{filename}' niet gevonden.")

def save_savings_log(filename="savings_log.json"):
    try:
        with open(filename, 'w', encoding='utf-8') as f: json.dump(st.session_state.savings_log, f, ensure_ascii=False, indent=4)
    except IOError as e: st.error(f"Fout bij opslaan van besparingslog: {e}")


def calculate_new_item_cost(new_product_data, original_desired_quantities, original_category_name_for_swap, shopping_list_item_idx_for_swap):
    min_cost = float('inf'); best_option_details = None
    if not original_desired_quantities: return None
    for dq_pair in original_desired_quantities:
        qty_desired = dq_pair["Hoeveelheid"]; unit_desired = dq_pair["Eenheid"].lower()
        if new_product_data["Eenheid"].lower() == unit_desired:
            pack_size_val = new_product_data["Hoeveelheid"]
            if pack_size_val == 0: continue
            prijs_product = new_product_data["Prijs"]
            n_packs_needed = max(1, math.ceil(qty_desired / pack_size_val)) if qty_desired > 0 else 0
            if n_packs_needed == 0 and qty_desired > 0 : n_packs_needed = 1
            current_cost = prijs_product * n_packs_needed
            if current_cost < min_cost:
                min_cost = current_cost
                is_stuks_dq_calc = dq_pair['Eenheid'].lower() == "stuks"
                hoeveelheid_main_val_calc = dq_pair['Hoeveelheid']
                if is_stuks_dq_calc:
                    hoeveelheid_display_dq_calc_str = str(int(hoeveelheid_main_val_calc)) if hoeveelheid_main_val_calc == int(hoeveelheid_main_val_calc) else str(hoeveelheid_main_val_calc)
                else:
                    hoeveelheid_display_dq_calc_str = f"{float(hoeveelheid_main_val_calc):.2f}".rstrip('0').rstrip('.') if '.' in f"{float(hoeveelheid_main_val_calc):.2f}" else f"{float(hoeveelheid_main_val_calc):.0f}"
                gekozen_optie_display_text_swap = f"{hoeveelheid_display_dq_calc_str} {dq_pair['Eenheid']}"
                if dq_pair.get('TolerantiePercentage',0) > 0 and dq_pair.get('TolerantieWaarde',0) > 0:
                    gekozen_optie_display_text_swap += f" (±{int(dq_pair['TolerantieWaarde'])} {dq_pair['TolerantieEenheid']})"
                best_option_details = {"shopping_list_item_idx": shopping_list_item_idx_for_swap, "Categorie": original_category_name_for_swap, "Naam": new_product_data["Naam"], "Gekozen voor optie": gekozen_optie_display_text_swap, "Aantal Pakken": n_packs_needed, "Verpakking Grootte": f"{new_product_data['Hoeveelheid']} {new_product_data['Eenheid']}", "Prijs Per Pakket": f"€{new_product_data['Prijs']:.2f}", "Kosten Totaal": f"€{min_cost:.2f}", "is_swapped": True, "original_product_data_if_swapped": None, "raw_product_data": dict(new_product_data), "_original_desired_quantities": original_desired_quantities}
    return best_option_details

def run_comparison_and_store_results(shopping_list_items):
    totaal_per_supermarkt = defaultdict(float)
    details_per_super = defaultdict(list)
    item_status_per_super = defaultdict(lambda: {"gevonden_categorien": set(), "ontbrekende_categorien": set()})
    if not shopping_list_items: st.session_state.comparison_results = None; return
    all_configured_supermarket_names = set(SUPERMARKETS.values())

    # Nieuwe variabele om de hoogste prijs van een complete lijst bij te houden
    highest_complete_list_total = 0.0

    for item_idx, item_in_shopping_list in enumerate(shopping_list_items):
        category_name = item_in_shopping_list["Categorie"]; products_for_this_category_item = item_in_shopping_list["Producten"]
        for sup_name in all_configured_supermarket_names:
            best_option_details_for_item_at_sup = None; min_cost_for_item_at_sup = float('inf')
            sup_specific_products = [p for p in products_for_this_category_item if p["Supermarkt"] == sup_name]
            for desired_qty_unit_pair in item_in_shopping_list["DesiredQuantities"]:
                qty_desired = desired_qty_unit_pair["Hoeveelheid"]; unit_desired = desired_qty_unit_pair["Eenheid"].lower()
                if qty_desired <= 0: continue
                current_wish_best_product_at_sup = {"cost": float('inf'), "product": None, "n_packs": 0}
                if sup_specific_products:
                    for p in sup_specific_products:
                        if p["Eenheid"].lower() == unit_desired:
                            pack_size_val = p["Hoeveelheid"]
                            if pack_size_val == 0: continue
                            prijs_product = p["Prijs"]; n_packs_needed = max(1, math.ceil(qty_desired / pack_size_val))
                            cost_for_this_product_for_this_wish = prijs_product * n_packs_needed
                            if cost_for_this_product_for_this_wish < current_wish_best_product_at_sup["cost"]:
                                current_wish_best_product_at_sup = {"cost":cost_for_this_product_for_this_wish, "product":p, "n_packs":n_packs_needed}
                if current_wish_best_product_at_sup["product"] is not None:
                    if current_wish_best_product_at_sup["cost"] < min_cost_for_item_at_sup:
                        min_cost_for_item_at_sup = current_wish_best_product_at_sup["cost"]; chosen_prod = current_wish_best_product_at_sup["product"]
                        is_stuks_compare = desired_qty_unit_pair['Eenheid'].lower() == "stuks"
                        hoeveelheid_main_val_comp = desired_qty_unit_pair['Hoeveelheid']
                        if is_stuks_compare:
                            hoeveelheid_display_comp_str = str(int(hoeveelheid_main_val_comp)) if hoeveelheid_main_val_comp == int(hoeveelheid_main_val_comp) else str(hoeveelheid_main_val_comp)
                        else:
                            hoeveelheid_display_comp_str = f"{float(hoeveelheid_main_val_comp):.2f}".rstrip('0').rstrip('.') if '.' in f"{float(hoeveelheid_main_val_comp):.2f}" else f"{float(hoeveelheid_main_val_comp):.0f}"
                        gekozen_optie_display_text = f"{hoeveelheid_display_comp_str} {desired_qty_unit_pair['Eenheid']}"
                        if desired_qty_unit_pair.get('TolerantiePercentage',0) > 0 and desired_qty_unit_pair.get('TolerantieWaarde',0) > 0:
                            gekozen_optie_display_text += f" (±{int(desired_qty_unit_pair['TolerantieWaarde'])} {desired_qty_unit_pair['TolerantieEenheid']})"
                        best_option_details_for_item_at_sup = {"shopping_list_item_idx": item_idx, "Categorie": category_name, "Naam": chosen_prod["Naam"], "Gekozen voor optie": gekozen_optie_display_text,"Aantal Pakken": current_wish_best_product_at_sup["n_packs"], "Verpakking Grootte": f"{chosen_prod['Hoeveelheid']} {chosen_prod['Eenheid']}", "Prijs Per Pakket": f"€{chosen_prod['Prijs']:.2f}", "Kosten Totaal": f"€{min_cost_for_item_at_sup:.2f}", "is_swapped": False, "original_product_data_if_swapped": None, "raw_product_data": dict(chosen_prod), "_original_desired_quantities": item_in_shopping_list["DesiredQuantities"] }
            if best_option_details_for_item_at_sup:
                totaal_per_supermarkt[sup_name] += min_cost_for_item_at_sup; details_per_super[sup_name].append(best_option_details_for_item_at_sup); item_status_per_super[sup_name]["gevonden_categorien"].add(category_name)
            else: item_status_per_super[sup_name]["ontbrekende_categorien"].add(category_name)
    
    summary_entries_for_sorting = []
    for sup_name in all_configured_supermarket_names:
        num_found = len(item_status_per_super[sup_name]["gevonden_categorien"]); num_missing = len(shopping_list_items) - num_found
        
        raw_total = totaal_per_supermarkt.get(sup_name, 0.0) 

        # Als de lijst compleet is, check of dit de hoogste totale prijs is
        if num_missing == 0 and len(shopping_list_items) > 0:
            if raw_total > highest_complete_list_total:
                highest_complete_list_total = raw_total

        status_text = "Compleet" if num_missing == 0 and len(shopping_list_items) > 0 else (f"{num_missing} item(s) ontbreken" if len(shopping_list_items) > 0 else "N.v.t. (lijst leeg)")
        price_str = f"€{raw_total:.2f}"
        if num_found == 0 and len(shopping_list_items) > 0 : price_str = "N.v.t." 
        elif num_missing > 0 and num_found > 0 : price_str += "*" 
        
        summary_entries_for_sorting.append({"Supermarkt":sup_name, "Totaalprijs_str":price_str, "Status_str":status_text, "_sort_missing_count":num_missing, "_sort_raw_total":raw_total if num_found > 0 else float('inf'), "raw_total_price": raw_total, "is_complete": (num_missing == 0 and len(shopping_list_items) > 0)})
    
    summary_entries_for_sorting.sort(key=lambda x: (x["_sort_missing_count"], x["_sort_raw_total"]))
    
    # Maak een nieuwe dictionary voor de gesorteerde details
    # en gebruik de globale sorteerfunctie get_product_sort_key
    sorted_details_data = defaultdict(list)
    for sup_name, product_list in details_per_super.items():
        sorted_details_data[sup_name] = sorted(product_list, key=get_product_sort_key) # GEWIJZIGD

    # Gebruik de gesorteerde details bij het opslaan in session state
    st.session_state.comparison_results = {
        "summary": summary_entries_for_sorting,
        "details": sorted_details_data, # HIER GEBRUIK JE DE GESORTEERDE DATA
        "item_status": defaultdict(lambda: {"gevonden_categorien":set(), "ontbrekende_categorien":set()}, {k: {"gevonden_categorien":set(v["gevonden_categorien"]), "ontbrekende_categorien":set(v["ontbrekende_categorien"])} for k,v in item_status_per_super.items()}),
        "totals": defaultdict(float, totaal_per_supermarkt),
        "highest_complete_list_total": highest_complete_list_total
    }


# --- Functie: Weergave Boodschappen Modus ---
def display_shopping_mode_view():
    st.sidebar.empty() 
    col_spacer1_shop, col_logo_shop, col_spacer2_shop = st.columns([0.5, 2, 0.5])
    with col_logo_shop:
        try: st.image("Checkr Logo.png", width=200) 
        except Exception: pass 
    st.title(f"🛒 Jouw Boodschappen bij {st.session_state.active_shopping_supermarket_name}")
    st.markdown("---")
    if not st.session_state.active_shopping_list_details: st.warning("Geen lijst geselecteerd.")
    else:
        for idx, item in enumerate(st.session_state.active_shopping_list_details):
            item_name = item.get("Naam", "Onbekend"); category = item.get("Categorie", "")
            gekozen_optie_display = item.get("Gekozen voor optie", "N.v.t.")
            aantal_pakken = item.get("Aantal Pakken", "-")
            verpakking_grootte = item.get("Verpakking Grootte", "-"); kosten_totaal = item.get("Kosten Totaal", "N.v.t.")
            is_swapped = item.get("is_swapped", False); display_name_suffix = " (Gewisseld)" if is_swapped else ""
            is_missing_placeholder = "(ontbreekt)" in item_name
            checkbox_key = f"bought_{st.session_state.active_shopping_supermarket_name}_{category.replace(' ', '_')}_{item_name.replace(' ', '_')}_{idx}"
            if checkbox_key not in st.session_state: st.session_state[checkbox_key] = False
            col_check, col_details = st.columns([1, 9])
            with col_check:
                if not is_missing_placeholder: st.checkbox(" ", value=st.session_state[checkbox_key], key=checkbox_key, label_visibility="collapsed")
            with col_details:
                if is_missing_placeholder: st.markdown(f"<span style='color: #777777;'><s>**{category}**</s> ({item_name})</span>", unsafe_allow_html=True)
                else: 
                    st.markdown(f"**{category}**: {item_name}{display_name_suffix}")
                    st.caption(f"{aantal_pakken}x {verpakking_grootte} (voor {gekozen_optie_display}) - {kosten_totaal}")
            st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)
    st.markdown("---")
    if st.button("⬅️ Terug naar Overzicht / Klaar met Boodschappen", key="exit_shopping_mode"):
        st.session_state.shopping_mode_active = False; st.session_state.active_shopping_list_details = None; st.session_state.active_shopping_supermarket_name = None
        # Reset de afgevinkte items bij het verlaten van de boodschappenmodus
        keys_to_delete = [key for key in st.session_state.keys() if key.startswith("bought_")]; 
        for key in keys_to_delete: del st.session_state[key]
        st.rerun()

# --- Functie: Hoofd App Weergave ---
def display_main_app_view():
    try: st.sidebar.image("Checkr Logo.png", width=100) 
    except Exception as e: st.sidebar.warning(f"Kon logo niet laden in sidebar: {e}")
    st.sidebar.header("Acties")
    if st.sidebar.button("Leeg Jouw Mandje", key="leeg_checkr_sidebar_knop"): clear_shopping_list()
    if st.sidebar.button("Jouw Mandje Opslaan", key="opslaan_checkr_sidebar_knop"): save_shopping_list()
    if st.sidebar.button("Jouw Mandje Laden", key="laden_checkr_sidebar_knop"): load_shopping_list()
    st.sidebar.markdown("---") 

    # Gebruik st.columns voor centrering van het logo en de besparing
    # We creëren een kolom voor het logo, een lege kolom aan elke kant om het te centreren.
    # En een aparte kolom voor de besparing, die we ook via CSS centreren.
    col1, col_logo, col2 = st.columns([1, 2, 1]) # Breedteverhouding aanpassen indien nodig
    with col_logo:
        st.image("Checkr Logo.png", width=300) # Logo is al gecentreerd met CSS

    # Direct onder het logo, gecentreerd
    total_saved_amount = sum(float(entry["amount_saved"]) for entry in st.session_state.savings_log)
    
    st.markdown(
        f"<div style='text-align: center; margin-bottom: 20px;'><h1>€{total_saved_amount:.2f}</h1><p style='font-size: 0.8em; margin-top: -10px;'>al bespaard</p></div>", 
        unsafe_allow_html=True
    )
    
    st.title("Welkom bij CheckR!") 
    st.header("Zoek producten en voeg toe aan je lijstje")
    search_term = st.text_input("Typ een zoekterm (bijv. 'braadworst', 'melk', 'thee'):", key="search_input")

    if search_term:
        filtered_products = [p for p in st.session_state.all_products if search_term.lower() in p["Naam"].lower()]
        if filtered_products:
            # --- START VERNIEUWDE CATEGORIE-LOGICA MET AUTOMATISCHE MERGING ---
            products_with_base_keywords = [] 
            for p_idx, p_data_dict in enumerate(filtered_products):
                base_keywords = clean_name(p_data_dict["Naam"], search_term=search_term)
                products_with_base_keywords.append({'id': p_idx, 'product': p_data_dict, 'base_keywords': set(base_keywords)})

            keyword_to_product_indices = defaultdict(set)
            for item_data in products_with_base_keywords:
                for kw in item_data['base_keywords']:
                    keyword_to_product_indices[kw].add(item_data['id'])

            product_indices_fset_to_keywords_group = defaultdict(set)
            for kw, product_indices_set in keyword_to_product_indices.items():
                if product_indices_set: 
                    product_indices_frozenset = frozenset(product_indices_set)
                    product_indices_fset_to_keywords_group[product_indices_frozenset].add(kw)
            
            dynamic_keyword_merge_rules = {}
            processed_keywords_in_any_dynamic_rule = set() 
            
            sorted_groups = sorted(product_indices_fset_to_keywords_group.items(), key=lambda item: (len(item[1]), len(item[0])), reverse=True)

            for _product_indices_fset, group_of_keywords in sorted_groups:
                if len(group_of_keywords) >= 2: 
                    if not group_of_keywords.intersection(processed_keywords_in_any_dynamic_rule):
                        if frozenset(group_of_keywords) not in MANUAL_KEYWORD_MERGE_RULES:
                            combined_name_parts = sorted([k.capitalize() for k in group_of_keywords])
                            if len(combined_name_parts) == 2:
                                combined_name = " ".join(combined_name_parts) 
                            else:
                                combined_name = " & ".join(combined_name_parts)
                            
                            dynamic_keyword_merge_rules[frozenset(group_of_keywords)] = combined_name
                            processed_keywords_in_any_dynamic_rule.update(group_of_keywords) 
            
            final_merge_rules = MANUAL_KEYWORD_MERGE_RULES.copy()
            for dyn_keys, dyn_name in dynamic_keyword_merge_rules.items():
                if dyn_keys not in final_merge_rules: 
                    final_merge_rules[dyn_keys] = dyn_name
            
            sorted_all_merge_rules = sorted(final_merge_rules.items(), key=lambda item_rule: len(item_rule[0]), reverse=True)

            products_with_final_keywords = []
            for item_data in products_with_base_keywords:
                p_dict = item_data['product']
                current_processing_keywords = set(item_data['base_keywords']) 
                final_keywords_for_this_product = set()
                keywords_merged_this_product = set()

                for merge_set_keys, combined_name in sorted_all_merge_rules: 
                    if merge_set_keys.issubset(current_processing_keywords) and not merge_set_keys.intersection(keywords_merged_this_product):
                        final_keywords_for_this_product.add(combined_name)
                        keywords_merged_this_product.update(merge_set_keys)
                
                for kw in current_processing_keywords:
                    if kw not in keywords_merged_this_product:
                        final_keywords_for_this_product.add(kw)
                
                products_with_final_keywords.append((p_dict, list(final_keywords_for_this_product)))
            
            word_super_counts = defaultdict(set)
            for p_tuple_entry, final_kw_list_for_p in products_with_final_keywords:
                p_data = p_tuple_entry 
                for w in final_kw_list_for_p: 
                    word_super_counts[w].add(p_data["Supermarkt"])
            # --- EINDE VERNIEUWDE CATEGORIE-LOGICA ---
            
            radio_options_for_display_struct = []
            generic_products_by_super = defaultdict(list)
            for p_generic_tuple, final_keywords_list_for_p_generic in products_with_final_keywords:
                if not final_keywords_list_for_p_generic: 
                    generic_products_by_super[p_generic_tuple["Supermarkt"]].append(p_generic_tuple)
            
            num_supers_for_generic = len(generic_products_by_super.keys())
            search_term_capitalized = search_term.capitalize()
            
            if num_supers_for_generic > 0:
                radio_options_for_display_struct.append({
                    "display_text": f"{search_term_capitalized} (algemeen) ({num_supers_for_generic})",
                    "actual_name": search_term_capitalized, 
                    "is_generic": True,
                    "count_for_sort": num_supers_for_generic + 0.51 
                })

            temp_specific_categories = []

            
            for word, supers_set in word_super_counts.items():
                num_supers_for_word = len(supers_set)
                if num_supers_for_word > 1: 
                    search_term_display_prefix = search_term.capitalize()
                    word_display_suffix = word
                    if ' ' not in word and '&' not in word and not any(char.isupper() for char in word):
                        word_display_suffix = word.capitalize()
                    
                    effective_display_name = ""
                    if search_term_display_prefix.lower() == word_display_suffix.lower():
                        effective_display_name = word_display_suffix 
                    else:
                        effective_display_name = f"{search_term_display_prefix} {word_display_suffix}"
                    
                    item_dict = {
                        "display_text": f"{effective_display_name} ({num_supers_for_word})",
                        "actual_name": word, 
                        "is_generic": False,
                        "count_for_sort": num_supers_for_word
                    }
                    temp_specific_categories.append(item_dict)
            radio_options_for_display_struct.extend(temp_specific_categories)
            radio_options_for_display_struct.sort(key=lambda x: (not x["is_generic"], -x["count_for_sort"]))


            final_display_options = [] 
            if num_supers_for_generic > 0: 
                final_display_options.append(next(opt for opt in radio_options_for_display_struct if opt["is_generic"]))
            
            temp_specific_categories.sort(key=lambda x: x["count_for_sort"], reverse=True)
            final_display_options.extend(temp_specific_categories)
            
            unique_display_options = []
            seen_display_texts = set()
            for option in final_display_options:
                if option["display_text"] not in seen_display_texts:
                    unique_display_options.append(option)
                    seen_display_texts.add(option["display_text"])
            
            final_radio_options_data = unique_display_options[:20]
            sub_category_display_strings = [opt["display_text"] for opt in final_radio_options_data]
            
            selected_sub_category_display_strings = []
            if sub_category_display_strings:
                st.subheader("Kies sub-categorieën om te combineren (of selecteer er één):")
                selected_sub_category_display_strings = st.multiselect("Beschikbare sub-categorieën:", sub_category_display_strings, key=f"multiselect_sub_cat_{search_term}")
            if selected_sub_category_display_strings:
                default_combined_name = search_term.capitalize()
                if len(selected_sub_category_display_strings) == 1:
                    single_selected_option_data = next((opt for opt in final_radio_options_data if opt["display_text"] == selected_sub_category_display_strings[0]), None)
                    if single_selected_option_data: default_combined_name = single_selected_option_data["actual_name"] 
                elif len(selected_sub_category_display_strings) > 1: default_combined_name = f"{search_term.capitalize()} (gecombineerd)"
                if not any(c.isupper() for c in default_combined_name) and '&' not in default_combined_name:
                    default_combined_name = default_combined_name.capitalize()
                combined_category_name = st.text_input("Geef een naam voor deze selectie (druk Enter om te bevestigen):", value=default_combined_name, key=f"combined_cat_name_input_{search_term}_{'_'.join(sorted(selected_sub_category_display_strings))}").strip()
                if combined_category_name:
                    aggregated_products = []; product_identifiers_added = set()
                    for selected_display_str in selected_sub_category_display_strings:
                        option_data = next((opt for opt in final_radio_options_data if opt["display_text"] == selected_display_str), None)
                        if option_data:
                            sub_cat_actual_name = option_data["actual_name"]; is_sub_cat_generic = option_data["is_generic"]
                            for p, final_keywords_for_p in products_with_final_keywords: 
                                product_matches_this_sub_category = False
                                if is_sub_cat_generic: 
                                    if not final_keywords_for_p: product_matches_this_sub_category = True
                                else: 
                                    if sub_cat_actual_name.lower() in [w.lower() for w in final_keywords_for_p]: product_matches_this_sub_category = True
                                if product_matches_this_sub_category:
                                    prod_id = (p["Supermarkt"], p["Naam"], p["Grootte_Origineel"], p["Prijs"])
                                    if prod_id not in product_identifiers_added: aggregated_products.append(p); product_identifiers_added.add(prod_id)
                    if aggregated_products:
                        with st.expander(f"Bekijk producten voor '{combined_category_name}' ({len(aggregated_products)} gevonden) - klik om details te zien", expanded=False):
                            st.dataframe([{"Supermarkt":p_agg["Supermarkt"], "Naam":p_agg["Naam"], "Grootte (Origineel)":p_agg["Grootte_Origineel"], "Prijs":f"€{p_agg['Prijs']:.2f}"} for p_agg in aggregated_products], height=300, hide_index=True, use_container_width=True)
                        st.markdown("---") 
                        st.subheader(f"Gewenste hoeveelheid voor '{combined_category_name}'")
                        st.info("Geef hieronder je 'of'-opties op. Bij het vergelijken wordt de goedkoopste optie per supermarkt gekozen.")
                        unique_units_combined = sorted(list(set(p_agg["Eenheid"] for p_agg in aggregated_products)))
                        if unique_units_combined:
                            desired_quantities_inputs_combined = []
                            for unit_option_comb in unique_units_combined:
                                st.markdown(f"**Optie voor eenheid: {unit_option_comb}**")
                                is_stuks = unit_option_comb.lower() == "stuks"
                                main_qty_key = f"main_qty_COMBINED_{combined_category_name}_{unit_option_comb}_{search_term}"
                                if is_stuks:
                                    main_step = 1 ; main_format = "%d"; main_qty_default_val = int(st.session_state.get(main_qty_key, 0)); min_val_for_input = 0 
                                else:
                                    main_step = 0.01 ; main_format = "%.2f"; main_qty_default_val = float(st.session_state.get(main_qty_key, 0.0)); min_val_for_input = 0.0
                                main_qty = st.number_input("Basishoeveelheid", min_value=min_val_for_input, value=main_qty_default_val, step=main_step, format=main_format, key=main_qty_key, label_visibility="collapsed", help=f"Voer de gewenste basishoeveelheid in {unit_option_comb} in.")
                                tolerance_percentage_input = 0; abs_tolerance_value_calculated = 0
                                if main_qty > 0:
                                    slider_key = f"tol_perc_slider_{combined_category_name}_{unit_option_comb}_{search_term}"
                                    current_slider_value = st.session_state.get(slider_key, 0)
                                    temp_abs_tolerance_display = 0
                                    if current_slider_value > 0:
                                        temp_abs_tolerance_display = round(float(main_qty) * (current_slider_value / 100.0))
                                        if temp_abs_tolerance_display == 0 and (float(main_qty) * (current_slider_value / 100.0)) >= 0.5:
                                            if float(main_qty) >= 2 or not is_stuks: temp_abs_tolerance_display = 1
                                        max_allowed_temp = math.floor(0.5 * float(main_qty))
                                        if temp_abs_tolerance_display > max_allowed_temp: temp_abs_tolerance_display = max_allowed_temp
                                        if temp_abs_tolerance_display < 1 and current_slider_value > 0 : temp_abs_tolerance_display = 0
                                    if current_slider_value > 0 and temp_abs_tolerance_display > 0: st.caption(f"Ingestelde tolerantie: {main_qty} {unit_option_comb} **± {temp_abs_tolerance_display} {unit_option_comb}** ({current_slider_value}%)")
                                    elif current_slider_value > 0: st.caption(f"Tolerantie van {current_slider_value}% is te klein (resulteert in ±0 {unit_option_comb}).")
                                    else: st.caption("Geen tolerantie geselecteerd (0%). Sleep slider om in te stellen.")
                                    tolerance_percentage_input = st.slider(f"Tolerantie (±%) voor {main_qty} {unit_option_comb}", min_value=0, max_value=50, value=current_slider_value, step=1, format="%d%%", key=slider_key, help=f"Max 50% van {main_qty} {unit_option_comb}.")
                                    if tolerance_percentage_input > 0: 
                                        abs_tolerance_value_calculated = round(float(main_qty) * (tolerance_percentage_input / 100.0))
                                        if abs_tolerance_value_calculated == 0 and (float(main_qty) * (tolerance_percentage_input / 100.0)) >= 0.5:
                                            if float(main_qty) >= 2 or not is_stuks : abs_tolerance_value_calculated = 1
                                        max_abs_tol_allowed_calc = math.floor(0.5 * float(main_qty))
                                        if abs_tolerance_value_calculated > max_abs_tol_allowed_calc: abs_tolerance_value_calculated = max_abs_tol_allowed_calc
                                        if abs_tolerance_value_calculated < 1: abs_tolerance_value_calculated = 0
                                else: st.caption("Voer eerst een basishoeveelheid (>0) in om tolerantie in te stellen.")
                                if main_qty > 0: 
                                    stored_main_qty = int(main_qty) if is_stuks and main_qty == int(main_qty) else float(main_qty)
                                    desired_quantities_inputs_combined.append({"Hoeveelheid": stored_main_qty, "Eenheid": unit_option_comb, "TolerantiePercentage": tolerance_percentage_input, "TolerantieWaarde": int(abs_tolerance_value_calculated), "TolerantieEenheid": unit_option_comb if tolerance_percentage_input > 0 and abs_tolerance_value_calculated > 0 else None})
                                st.markdown("---") 
                            if st.button(f"Voeg '{combined_category_name}' met opgegeven opties toe aan lijstje", key=f"add_btn_COMBINED_{combined_category_name}_{search_term}"):
                                add_to_shopping_list(combined_category_name, desired_quantities_inputs_combined, aggregated_products)
                        else: st.info(f"Geen bruikbare eenheden voor producten in '{combined_category_name}'.")
                    else: st.info(f"Geen producten voor combinatie van sub-categorieën.")
        elif search_term: st.info(f"Geen producten gevonden die '{search_term}' bevatten.")

    st.header("Jouw Mandje") 
    if st.session_state.shopping_list:
        if not st.session_state.shopping_list:
            st.info("Je boodschappenlijstje is momenteel leeg.")
        else:
            for idx, item_data_main in enumerate(st.session_state.shopping_list):
                col1_item, col2_remove = st.columns([0.9, 0.1]) 
                with col1_item:
                    qty_strs = []
                    for dq in item_data_main["DesiredQuantities"]:
                        is_stuks_dq_display = dq['Eenheid'].lower() == "stuks"
                        hoeveelheid_main_val_display = dq['Hoeveelheid']
                        if is_stuks_dq_display:
                            hoeveelheid_display_str_val = str(int(hoeveelheid_main_val_display)) if hoeveelheid_main_val_display == int(hoeveelheid_main_val_display) else str(hoeveelheid_main_val_display)
                        else:
                            hoeveelheid_display_str_val = f"{float(hoeveelheid_main_val_display):.2f}".rstrip('0').rstrip('.') if '.' in f"{float(hoeveelheid_main_val_display):.2f}" else f"{float(hoeveelheid_main_val_display):.0f}"
                            if hoeveelheid_display_str_val == "" and float(hoeveelheid_main_val_display) == 0: hoeveelheid_display_str_val = "0"
                        qty_str = f"{hoeveelheid_display_str_val} {dq['Eenheid']}"
                        if dq.get("TolerantiePercentage", 0) > 0 and dq.get("TolerantieWaarde", 0) > 0:
                            qty_str += f" (±{int(dq['TolerantieWaarde'])} {dq['TolerantieEenheid']})"
                        qty_strs.append(qty_str)
                    st.markdown(f"**{idx + 1}. {item_data_main['Categorie']}**")
                    st.caption(f"Gewenste 'OF'-opties: {', '.join(qty_strs) if qty_strs else 'Geen'}")
                with col2_remove:
                    remove_button_key = f"remove_item_btn_{idx}_{item_data_main['Categorie'].replace(' ','_')}" 
                    if st.button("🗑️", key=remove_button_key, help=f"Verwijder '{item_data_main['Categorie']}' uit lijstje"):
                        removed_item = st.session_state.shopping_list.pop(idx) 
                        st.success(f"'{removed_item['Categorie']}' verwijderd!")
                        if 'comparison_results' in st.session_state: del st.session_state.comparison_results
                        if 'active_swap' in st.session_state: del st.session_state.active_swap
                        st.rerun()
                st.markdown("---") 
        
        if st.session_state.shopping_list: 
            if st.button("Vergelijk Prijzen", key="vergelijk_prijzen_hoofd_knop"):
                run_comparison_and_store_results(st.session_state.shopping_list) 
                st.session_state.active_swap = None; st.rerun() 
    else: 
        st.info("Je boodschappenlijstje is leeg.") 
        if 'comparison_results' in st.session_state: del st.session_state.comparison_results
        if 'active_swap' in st.session_state: del st.session_state.active_swap

    if st.session_state.get('skipped_products_log') and len(st.session_state.skipped_products_log) > 0:
        st.markdown("---") 
        with st.expander(f"Overgeslagen producten bij laden ({len(st.session_state.skipped_products_log)} voorbeelden)", expanded=False):
            st.caption("Deze producten konden niet worden ingeladen omdat hun 'grootte' informatie niet direct herkend werd.")
            st.dataframe(st.session_state.skipped_products_log)

    if 'comparison_results' in st.session_state and st.session_state.comparison_results is not None:
        results = st.session_state.comparison_results; summary_entries = results.get("summary", []) 
        details_data = results.get("details", defaultdict(list)); item_status_data = results.get("item_status", defaultdict(lambda: {"gevonden_categorien": set(), "ontbrekende_categorien": set()}))
        highest_complete_list_total = results.get("highest_complete_list_total", 0.0)

        st.subheader("💶 Overzicht Totaalprijzen per Supermarkt")
        if summary_entries:
            df_summary_data = [{"Supermarkt": entry["Supermarkt"], "Totaalprijs": entry["Totaalprijs_str"], "Status": entry["Status_str"]} for entry in summary_entries]
            st.dataframe(df_summary_data, hide_index=True, use_container_width=True, column_order=("Supermarkt", "Totaalprijs", "Status"))
            if any("*" in entry["Totaalprijs"] for entry in df_summary_data if isinstance(entry.get("Totaalprijs"), str)): st.caption("* Totaalprijs is berekend voor de beschikbare producten.")
        else: st.info("Geen supermarkten te vergelijken (na filtering).")
        st.subheader("📋 Prijsdetail per Supermarkt")
        if not summary_entries and st.session_state.shopping_list: st.info("Geen details te tonen.")
        else: 
            for summary_idx, entry in enumerate(summary_entries): 
                sup_name = entry["Supermarkt"]
                current_sup_total = entry["raw_total_price"]
                is_current_sup_complete = entry["is_complete"]
                
                expander_label = f"{sup_name} (Totaal: {entry['Totaalprijs_str']}) - Klik om producten te zien/wisselen"
                
                with st.expander(expander_label, expanded=False): 
                    # Knop om de boodschappenmodus te starten
                    if st.button(f"Start Boodschappen Modus voor {sup_name}", key=f"start_shopping_mode_{sup_name}"):
                        st.session_state.shopping_mode_active = True
                        st.session_state.active_shopping_supermarket_name = sup_name
                        st.session_state.active_shopping_list_details = details_data.get(sup_name, [])

                        # Bereken en sla de besparing op
                        if highest_complete_list_total > 0 and is_current_sup_complete:
                            
                            if highest_complete_list_total > current_sup_total:
                                saved_amount = highest_complete_list_total - current_sup_total
                                savings_entry = {
                                    "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "supermarket_chosen": sup_name,
                                    "cost_chosen_supermarket": f"{current_sup_total:.2f}",
                                    "highest_complete_list_cost": f"{highest_complete_list_total:.2f}",
                                    "amount_saved": f"{saved_amount:.2f}",
                                    "shopping_list_items": [item["Categorie"] for item in st.session_state.shopping_list]
                                }
                                st.session_state.savings_log.append(savings_entry)
                                save_savings_log() # Sla de bijgewerkte log op
                                st.success(f"Gefeliciteerd! Je hebt €{saved_amount:.2f} bespaard door bij {sup_name} te winkelen t.o.v. de duurste complete lijst.")
                            else:
                                st.info("Geen besparing opgeslagen, omdat de gekozen supermarkt niet goedkoper was dan de 'duurste complete lijst', of er was geen duurdere complete lijst.")
                        elif not is_current_sup_complete and len(st.session_state.shopping_list) > 0:
                            st.warning(f"Besparing niet berekend: De boodschappenlijst voor {sup_name} is niet compleet.")
                        else:
                             st.info("Geen besparing berekend, omdat er geen complete lijsten waren om mee te vergelijken.")


                        st.rerun()

                    st.markdown("---")
                    rows_to_display = details_data.get(sup_name, []) # GEWIJZIGD: geen extra sortering hier
                    
                    if rows_to_display:
                        for detail_idx, row_data in enumerate(rows_to_display):
                            product_naam_lower = row_data.get("Naam", "").lower() 
                            is_missing_item = "(ontbreekt)" in product_naam_lower

                            item_display_name = row_data.get('Naam', 'Onbekend product') 
                            if row_data.get('is_swapped'): 
                                item_display_name += " (Gewisseld)"

                            item_cols = st.columns([5, 1]) 
                            with item_cols[0]:
                                if is_missing_item:
                                    st.markdown(
                                        f"<span style='font-weight:bold;'>► {row_data.get('Categorie', 'N/A')}</span>: <span style='color: #F87575;'>{item_display_name}</span>",
                                        unsafe_allow_html=True
                                    )
                                    st.caption("Dit product is niet direct gevonden. Klik 🔁 om een alternatief te zoeken.")
                                else:
                                    st.markdown(f"**{row_data.get('Categorie', 'N/A')}**: {item_display_name}")
                                    
                                    gekozen_optie_display = row_data.get('Gekozen voor optie', 'N/A')
                                    aantal_pakken_display = row_data.get('Aantal Pakken', 'N/A')
                                    kosten_totaal_display = row_data.get('Kosten Totaal', 'N/A')
                                    st.caption(f"Keuze: {gekozen_optie_display} | Aantal: {aantal_pakken_display}x | Kosten: {kosten_totaal_display}")
                            
                            with item_cols[1]: 
                                swap_button_key = f"start_swap_{sup_name.replace(' ', '_')}_{detail_idx}_{summary_idx}_{row_data.get('Categorie', 'NOCAT').replace(' ', '_')}" 
                                if st.button("🔁", key=swap_button_key, help=f"Wissel '{row_data.get('Naam', 'Product')}' bij {sup_name}"):
                                    original_sl_item_idx = row_data.get("shopping_list_item_idx", -1)
                                    original_desired_quantities_for_swap = row_data.get("_original_desired_quantities", []) 
                                    if not original_desired_quantities_for_swap and original_sl_item_idx != -1 and original_sl_item_idx < len(st.session_state.shopping_list): 
                                        original_desired_quantities_for_swap = st.session_state.shopping_list[original_sl_item_idx]["DesiredQuantities"]
                                    st.session_state.active_swap = {
                                        "sup_name": sup_name, 
                                        "detail_item_index": detail_idx, 
                                        "original_item_data": dict(row_data), 
                                        "original_category_name": row_data.get("Categorie", "N/A"), 
                                        "original_shopping_list_item_idx": original_sl_item_idx, 
                                        "original_desired_quantities": original_desired_quantities_for_swap 
                                    }
                                    if f"swap_search_term_{sup_name}_{detail_idx}" in st.session_state: 
                                        del st.session_state[f"swap_search_term_{sup_name}_{detail_idx}"]
                                    st.rerun()
                            st.markdown("---") 
                    if st.session_state.active_swap and st.session_state.active_swap["sup_name"] == sup_name:
                        active_swap_info = st.session_state.active_swap; detail_item_idx_being_swapped = active_swap_info["detail_item_index"]
                        original_item_display_name_swap = active_swap_info["original_item_data"]["Naam"]; original_item_chosen_option_swap = active_swap_info["original_item_data"]["Gekozen voor optie"]
                        st.markdown(f"##### Wisselen: _{original_item_display_name_swap}_ (voor _{original_item_chosen_option_swap}_)")
                        swap_search_key_ui = f"swap_search_term_{sup_name}_{detail_item_idx_being_swapped}" 
                        swap_search_term_input = st.text_input("Zoek nieuw product:", key=swap_search_key_ui, placeholder=f"Typ hier om te zoeken bij {sup_name}")
                        if swap_search_term_input:
                            swap_candidates = [p for p in st.session_state.all_products if p["Supermarkt"] == sup_name and swap_search_term_input.lower() in p["Naam"].lower()]
                            if not swap_candidates: st.info("Geen producten gevonden met deze zoekterm.")
                            else:
                                st.write("**Kies vervangend product:**")
                                for cand_idx, candidate_prod_data in enumerate(swap_candidates[:10]):
                                    btn_label = f"{candidate_prod_data['Naam']} ({candidate_prod_data['Grootte_Origineel']}) - €{candidate_prod_data['Prijs']:.2f}"
                                    if st.button(btn_label, key=f"do_swap_action_{sup_name}_{detail_item_idx_being_swapped}_{cand_idx}"):
                                        newly_calculated_item_details = calculate_new_item_cost(candidate_prod_data, active_swap_info["original_desired_quantities"], active_swap_info["original_category_name"], active_swap_info["original_shopping_list_item_idx"])
                                        if newly_calculated_item_details:
                                            current_details_list = st.session_state.comparison_results["details"].get(sup_name, [])
                                            if detail_item_idx_being_swapped < len(current_details_list):
                                                current_details_list[detail_item_idx_being_swapped] = newly_calculated_item_details
                                                new_total_for_sup = sum(float(str(d["Kosten Totaal"]).replace("€", "").replace(",", ".")) for d in current_details_list if d["Kosten Totaal"] not in ["N.v.t.", "-"])
                                                st.session_state.comparison_results["totals"][sup_name] = new_total_for_sup
                                                for summ_entry in st.session_state.comparison_results["summary"]:
                                                    if summ_entry["Supermarkt"] == sup_name:
                                                        summ_entry["_sort_raw_total"] = new_total_for_sup; price_str_upd = f"€{new_total_for_sup:.2f}"
                                                        if "(ontbreekt)" in active_swap_info["original_item_data"]["Naam"]:
                                                            st.session_state.comparison_results["item_status"][sup_name]["gevonden_categorien"].add(active_swap_info["original_category_name"])
                                                            if active_swap_info["original_category_name"] in st.session_state.comparison_results["item_status"][sup_name]["ontbrekende_categorien"]: st.session_state.comparison_results["item_status"][sup_name]["ontbrekende_categorien"].remove(active_swap_info["original_category_name"])
                                                        new_num_missing = len(st.session_state.comparison_results["item_status"][sup_name]["ontbrekende_categorien"])
                                                        summ_entry["_sort_missing_count"] = new_num_missing; summ_entry["Status_str"] = "Compleet" if new_num_missing == 0 else f"{new_num_missing} item(s) ontbreken"
                                                        if new_num_missing > 0 and len(st.session_state.comparison_results["item_status"].get(sup_name,{}).get("gevonden_categorien",set())) > 0 : price_str_upd += "*" 
                                                        summ_entry["Totaalprijs_str"] = price_str_upd; break
                                                st.session_state.comparison_results["summary"].sort(key=lambda x: (x["_sort_missing_count"], x["_sort_raw_total"]))
                                                st.success(f"'{active_swap_info['original_item_data']['Naam']}' vervangen door '{candidate_prod_data['Naam']}'.")
                                        else: st.error(f"Kon '{candidate_prod_data['Naam']}' niet gebruiken om de gewenste hoeveelheid te vervullen (check eenheden).")
                                        st.session_state.active_swap = None; st.rerun()
                        if st.button("Annuleer Wisselen", key=f"cancel_swap_{sup_name}_{active_swap_info['detail_item_index'] if st.session_state.active_swap else 'cancel_placeholder'}"):
                            st.session_state.active_swap = None; st.rerun()
    
    # De besparingsweergave is nu naar de top verplaatst en vereenvoudigd.
    # Als je de 'Bekijk Besparingsgeschiedenis' knop weer wilt, moet je die hier weer toevoegen.


# --- Hoofd App Flow ---
if st.session_state.get('shopping_mode_active', False):
    display_shopping_mode_view()
else:
    display_main_app_view()

# --- Data Laden bij het starten van de app (blijft onderaan) ---
if 'all_products' not in st.session_state or not st.session_state.all_products:
    with st.spinner("Productdata wordt geladen..."):
        st.session_state.all_products = fetch_supermarket_data("https://raw.githubusercontent.com/supermarkt/checkjebon/main/data/supermarkets.json")
    if st.session_state.all_products:
        if 'initial_load_success_shown' not in st.session_state: 
            st.success(f"{len(st.session_state.all_products)} producten succesvol geladen!", icon="✅")
            st.session_state.initial_load_success_shown = True
    else:
        st.error("Kon geen productdata laden.")