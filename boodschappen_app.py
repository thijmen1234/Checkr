import streamlit as st
import requests
import re
import spacy
from collections import defaultdict
import math
import os
import json

# --- Streamlit UI Configuratie ---
try:
    st.set_page_config(layout="centered", page_title="Welkom bij CheckR!", page_icon="Checkr Logo.png")
except Exception as e:
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
        div[data-testid="stNumberInput"] input, /* Target number input fields */
        div[data-testid="stSlider"] /* Target sliders */
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
            border-radius: 8px !important; border: none !important; padding: 1rem !important;
        }
        div[data-baseweb="alert"][data-kind="info"] div[data-testid="stMarkdownContainer"] p,
        div[data-baseweb="alert"][data-kind="info"] div[data-testid="stMarkdownContainer"] li {
            color: white !important;
        }
        div[data-baseweb="alert"][data-kind="info"] svg {
            fill: white !important;
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

apply_custom_styling()

# --- SpaCy Model Laden ---
@st.cache_resource
def load_spacy_model():
    try: return spacy.load("nl_core_news_sm")
    except OSError:
        st.info("SpaCy Nederlands model (nl_core_news_sm) wordt gedownload...")
        try: spacy.cli.download("nl_core_news_sm"); return spacy.load("nl_core_news_sm")
        except Exception as e: st.error(f"Fout bij downloaden SpaCy: {e}."); st.stop()
nlp = load_spacy_model()

# --- Constanten en Mappings ---
SUPERMARKETS = {"ah": "Albert Heijn", "aldi": "Aldi", "coop": "COOP", "dekamarkt": "DEKAMarkt", "dirk": "Dirk", "hoogvliet": "Hoogvliet", "jumbo": "Jumbo", "picnic": "PicNic", "plus": "Plus", "spar": "SPAR", "vomar": "Vomar"}
RETAILER_KEYS = set(SUPERMARKETS.keys())
RETAILER_NAMES = set(name.lower() for name in SUPERMARKETS.values())
DIGIT_PATTERN = re.compile(r"\b\d+([\.,]?\d+)?\s*(l|ml|cl|kg|g|gr|st|stuk|x|per|gram|liter|ltr)?\b", re.IGNORECASE)
REMOVE_WORDS = {"stuk", "stuks", "st", "pieces", "piece", "per", "x", "gram", "kilogram", "liter", "ltr", "de", "het", "een", "en", "of"}
EXTRA_STOPWORDS = {"met", "zonder", "voor", "tegen", "bij", "van", "op", "onder", "boven", "door", "in", "uit", "tot", "tussen", "achter", "langs", "om", "over", "na", "aan", "binnen", "buiten", "sinds", "zoals", "vers", "verse"}
ALL_STOPWORDS = REMOVE_WORDS.union(EXTRA_STOPWORDS)

# NIEUW: Definieer hier je samenvoegregels
KEYWORD_MERGE_RULES = {
    frozenset({"go", "tan"}): "Go-Tan", # Als 'go' en 'tan' samen voorkomen, wordt het "Go-Tan"
    # Voeg hier meer regels toe: frozenset({lowercase_woord1, lowercase_woord2}): "Gewenste Categorienaam"
    # Bijvoorbeeld:
    # frozenset({"biologisch", "kipfilet"}): "Biologische Kipfilet",
}

# --- Session State Initialisatie ---
if 'shopping_list' not in st.session_state: st.session_state.shopping_list = []
if 'all_products' not in st.session_state: st.session_state.all_products = []
if 'comparison_results' not in st.session_state: st.session_state.comparison_results = None
if 'active_swap' not in st.session_state: st.session_state.active_swap = None
if 'shopping_mode_active' not in st.session_state: st.session_state.shopping_mode_active = False
if 'active_shopping_list_details' not in st.session_state: st.session_state.active_shopping_list_details = None
if 'active_shopping_supermarket_name' not in st.session_state: st.session_state.active_shopping_supermarket_name = None

# --- Functie Definities ---
# (clean_name, fetch_supermarket_data, etc. blijven hier)
def clean_name(name, search_term=None):
    original_name_lower = name.lower(); parts = original_name_lower.split()
    if parts:
        first = parts[0].rstrip(':')
        if first in RETAILER_KEYS or first in RETAILER_NAMES: parts = parts[1:]
    name_cleaned_initial = " ".join(parts)
    name_cleaned_initial = DIGIT_PATTERN.sub("", name_cleaned_initial)
    name_cleaned_initial = re.sub(r"[-/]", " ", name_cleaned_initial)
    name_cleaned_initial = re.sub(r"\s+", " ", name_cleaned_initial).strip()
    final_cleaned_text = name_cleaned_initial
    if search_term:
        search_term_lower = search_term.lower().strip()
        if search_term_lower:
            variations_to_remove = sorted([search_term_lower + "jes", search_term_lower + "en", search_term_lower + "s", search_term_lower], key=len, reverse=True)
            for var in variations_to_remove: final_cleaned_text = final_cleaned_text.replace(var, "")
            if search_term_lower.endswith("jes") and len(search_term_lower)>3: final_cleaned_text = final_cleaned_text.replace(search_term_lower[:-3],"")
            elif search_term_lower.endswith("en") and len(search_term_lower)>2: final_cleaned_text = final_cleaned_text.replace(search_term_lower[:-2],"")
            elif search_term_lower.endswith("s") and len(search_term_lower)>1: final_cleaned_text = final_cleaned_text.replace(search_term_lower[:-1],"")
        final_cleaned_text = re.sub(r"\s+", " ", final_cleaned_text).strip()
    doc = nlp(final_cleaned_text); lemmas = [tok.lemma_.lower() for tok in doc if tok.lemma_.isalpha()] # .lower() hier
    lemma_fix = {"varkens":"varken","kippen":"kip","runders":"runder","worsten":"worst","broden":"brood","aardappelen":"aardappel","groenten":"groente","fruiten":"fruit","broodjes":"brood","kips":"kip"}
    corrected_lemmas = [lemma_fix.get(lemma, lemma) for lemma in lemmas]
    filtered_words_set = {word for word in corrected_lemmas if word not in ALL_STOPWORDS}
    normalized_words = []
    for word in list(filtered_words_set): 
        if word.endswith("s") and len(word)>4 and (word[:-1] in nlp.vocab) and word[:-1] not in ALL_STOPWORDS :
             normalized_words.append(word[:-1])
        else:
            normalized_words.append(word)
    return sorted(list(set(normalized_words)))

@st.cache_data(ttl=3600)
def fetch_supermarket_data(url: str):
    try:
        resp = requests.get(url); resp.raise_for_status(); data = resp.json()
    except requests.exceptions.RequestException as e: st.error(f"Fout bij ophalen data: {e}."); return []
    products = []
    size_unit_pattern = re.compile(r"(\d+[\.,]?\d*)\s*(ml|cl|l|gram|g|kg|stuks?|st|x|per|stuks|stuk|s)\b", re.IGNORECASE)
    for entry in data:
        store_code = entry.get("c", "").lower(); store_name = SUPERMARKETS.get(store_code, store_code.capitalize())
        items = entry.get("d") or next((v for v in entry.values() if isinstance(v, list)), [])
        for item_entry in items: 
            name = item_entry.get("n", "").strip(); raw_size_str = item_entry.get("s", "").strip(); price = item_entry.get("p")
            if not name or price is None: continue
            quantity, unit = None, None
            match = size_unit_pattern.search(raw_size_str)
            if match:
                num_str, raw_unit_str = match.group(1).replace(",", "."), match.group(2).lower()
                try:
                    num_val = float(num_str)
                    if "ml" in raw_unit_str or "cl" in raw_unit_str: unit = "liter"; quantity = num_val / 1000 if "ml" in raw_unit_str else num_val / 100
                    elif "l" in raw_unit_str: unit = "liter"; quantity = num_val
                    elif "kg" in raw_unit_str: unit = "gram"; quantity = num_val * 1000
                    elif "gram" in raw_unit_str or "g" in raw_unit_str: unit = "gram"; quantity = num_val
                    elif any(u in raw_unit_str for u in ["st", "stuk", "x", "per", "s"]): unit = "stuks"; quantity = num_val
                except ValueError: pass
            if quantity is not None and unit is not None: products.append({"Supermarkt": store_name, "Naam": name, "Grootte_Origineel": raw_size_str, "Hoeveelheid": quantity, "Eenheid": unit, "Prijs": float(price)})
    return products

def add_to_shopping_list(category, desired_quantities_list, selected_products_for_category):
    filtered_desired_quantities = [dq for dq in desired_quantities_list if dq.get("Hoeveelheid", 0) > 0]
    if not filtered_desired_quantities: st.warning(f"Geen geldige hoeveelheden voor '{category}'."); return
    existing_item_index = next((i for i, item in enumerate(st.session_state.shopping_list) if item["Categorie"] == category), -1)
    new_item_data = {"Categorie": category, "DesiredQuantities": filtered_desired_quantities, "Producten": selected_products_for_category}
    qty_display_strs = []
    for dq in filtered_desired_quantities:
        hoeveelheid_str = f"{int(dq['Hoeveelheid'])}" if dq['Eenheid'].lower() == "stuks" and dq['Hoeveelheid'] == int(dq['Hoeveelheid']) else f"{dq['Hoeveelheid']}"
        s = f"{hoeveelheid_str} {dq['Eenheid']}"
        if dq.get("TolerantiePercentage", 0) > 0 and dq.get("TolerantieWaarde", 0) > 0:
            s += f" (±{int(dq['TolerantieWaarde'])} {dq['TolerantieEenheid']})"
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
                gekozen_optie_display_text_swap = f"{dq_pair['Hoeveelheid']} {dq_pair['Eenheid']}"
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
                        hoeveelheid_display_comp = int(desired_qty_unit_pair['Hoeveelheid']) if desired_qty_unit_pair['Eenheid'].lower() == "stuks" and desired_qty_unit_pair['Hoeveelheid'] == int(desired_qty_unit_pair['Hoeveelheid']) else desired_qty_unit_pair['Hoeveelheid']
                        gekozen_optie_display_text = f"{hoeveelheid_display_comp} {desired_qty_unit_pair['Eenheid']}"
                        if desired_qty_unit_pair.get('TolerantiePercentage',0) > 0 and desired_qty_unit_pair.get('TolerantieWaarde',0) > 0:
                            gekozen_optie_display_text += f" (±{int(desired_qty_unit_pair['TolerantieWaarde'])} {desired_qty_unit_pair['TolerantieEenheid']})"
                        best_option_details_for_item_at_sup = {"shopping_list_item_idx": item_idx, "Categorie": category_name, "Naam": chosen_prod["Naam"], "Gekozen voor optie": gekozen_optie_display_text,"Aantal Pakken": current_wish_best_product_at_sup["n_packs"], "Verpakking Grootte": f"{chosen_prod['Hoeveelheid']} {chosen_prod['Eenheid']}", "Prijs Per Pakket": f"€{chosen_prod['Prijs']:.2f}", "Kosten Totaal": f"€{min_cost_for_item_at_sup:.2f}", "is_swapped": False, "original_product_data_if_swapped": None, "raw_product_data": dict(chosen_prod), "_original_desired_quantities": item_in_shopping_list["DesiredQuantities"] }
            if best_option_details_for_item_at_sup:
                totaal_per_supermarkt[sup_name] += min_cost_for_item_at_sup; details_per_super[sup_name].append(best_option_details_for_item_at_sup); item_status_per_super[sup_name]["gevonden_categorien"].add(category_name)
            else: item_status_per_super[sup_name]["ontbrekende_categorien"].add(category_name)
    for sup_name in all_configured_supermarket_names:
        if not (item_status_per_super[sup_name]["gevonden_categorien"] or item_status_per_super[sup_name]["ontbrekende_categorien"]):
            if len(shopping_list_items) > 0: 
                for item_idx_miss, sl_item_miss in enumerate(shopping_list_items): item_status_per_super[sup_name]["ontbrekende_categorien"].add(sl_item_miss["Categorie"])
            else: continue
        for item_idx_fill, shopping_list_item_entry_fill in enumerate(shopping_list_items):
            cat_naam_fill = shopping_list_item_entry_fill["Categorie"]
            is_gevonden = any(d["Categorie"] == cat_naam_fill and not "(ontbreekt)" in d["Naam"] for d in details_per_super.get(sup_name, []))
            is_al_als_ontbrekend = any(d["Categorie"] == cat_naam_fill and "(ontbreekt)" in d["Naam"] for d in details_per_super.get(sup_name, []))
            if not is_gevonden and not is_al_als_ontbrekend: 
                dq_for_missing = shopping_list_item_entry_fill.get("DesiredQuantities", [])
                gekozen_optie_str_missing_list = []
                for dq_m in dq_for_missing:
                    hoeveelheid_display_m = int(dq_m['Hoeveelheid']) if dq_m['Eenheid'].lower() == "stuks" and dq_m['Hoeveelheid'] == int(dq_m['Hoeveelheid']) else dq_m['Hoeveelheid']
                    s_m = f"{hoeveelheid_display_m} {dq_m['Eenheid']}"
                    if dq_m.get("TolerantiePercentage", 0) > 0 and dq_m.get("TolerantieWaarde", 0) > 0: s_m += f" (±{int(dq_m['TolerantieWaarde'])} {dq_m['TolerantieEenheid']})"
                    gekozen_optie_str_missing_list.append(s_m)
                gekozen_optie_str_missing = ", ".join(gekozen_optie_str_missing_list) if gekozen_optie_str_missing_list else "N.v.t."
                details_per_super[sup_name].append({"shopping_list_item_idx":item_idx_fill, "Categorie":cat_naam_fill, "Naam":f"{cat_naam_fill} (ontbreekt)", "Gekozen voor optie": gekozen_optie_str_missing, "Aantal Pakken":"-", "Verpakking Grootte":"-", "Prijs Per Pakket":"-", "Kosten Totaal":"N.v.t.", "is_swapped":False, "original_product_data_if_swapped":None, "raw_product_data":None, "_original_desired_quantities": dq_for_missing})
                item_status_per_super[sup_name]["ontbrekende_categorien"].add(cat_naam_fill)
    summary_entries_for_sorting = []
    for sup_name in all_configured_supermarket_names:
        num_found = len(item_status_per_super[sup_name]["gevonden_categorien"]); num_missing = len(shopping_list_items) - num_found
        if not shopping_list_items and not (num_found > 0 or num_missing > 0): continue
        if len(shopping_list_items) > 0 and not (num_found > 0 or item_status_per_super[sup_name]["ontbrekende_categorien"]): continue
        raw_total = totaal_per_supermarkt.get(sup_name, 0.0) 
        status_text = "Compleet" if num_missing == 0 and len(shopping_list_items) > 0 else (f"{num_missing} item(s) ontbreken" if len(shopping_list_items) > 0 else "N.v.t. (lijst leeg)")
        price_str = f"€{raw_total:.2f}"
        if num_found == 0 and len(shopping_list_items) > 0 : price_str = "N.v.t." 
        elif num_missing > 0 and num_found > 0 : price_str += "*" 
        summary_entries_for_sorting.append({"Supermarkt":sup_name, "Totaalprijs_str":price_str, "Status_str":status_text, "_sort_missing_count":num_missing, "_sort_raw_total":raw_total if num_found > 0 else float('inf')})
    summary_entries_for_sorting.sort(key=lambda x: (x["_sort_missing_count"], x["_sort_raw_total"]))
    st.session_state.comparison_results = {"summary":summary_entries_for_sorting, "details":defaultdict(list, {k: list(v) for k, v in details_per_super.items()}), "item_status":defaultdict(lambda: {"gevonden_categorien":set(), "ontbrekende_categorien":set()}, {k: {"gevonden_categorien":set(v["gevonden_categorien"]), "ontbrekende_categorien":set(v["ontbrekende_categorien"])} for k,v in item_status_per_super.items()}), "totals":defaultdict(float, totaal_per_supermarkt)}

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
    col_spacer1_main, col_logo_main_page, col_spacer2_main = st.columns([0.5, 2, 0.5]) 
    with col_logo_main_page:
        try: st.image("Checkr Logo.png", width=300) 
        except Exception as e: st.error(f"Kon logo niet laden op hoofdpagina: {e}.")
    st.title("Welkom bij CheckR!") 
    st.header("Zoek producten en voeg toe aan je lijstje")
    search_term = st.text_input("Typ een zoekterm (bijv. 'braadworst', 'melk', 'brood'):", key="search_input")

    if search_term:
        filtered_products = [p for p in st.session_state.all_products if search_term.lower() in p["Naam"].lower()]
        if filtered_products:
            # --- START AANGEPAST CATEGORIE-LOGICA BLOK (MET KEYWORD MERGING) ---
            products_with_final_keywords = []
            for p in filtered_products:
                base_keywords_for_product = clean_name(p["Naam"], search_term=search_term)
                
                processed_keywords_for_product_set = set(base_keywords_for_product) 
                final_keywords_for_product_set = set() 
                individual_keywords_already_merged = set()

                sorted_merge_rules = sorted(KEYWORD_MERGE_RULES.items(), key=lambda item: len(item[0]), reverse=True)

                for merge_set_keys, combined_name in sorted_merge_rules:
                    if merge_set_keys.issubset(processed_keywords_for_product_set) and not merge_set_keys.intersection(individual_keywords_already_merged):
                        final_keywords_for_product_set.add(combined_name)
                        individual_keywords_already_merged.update(merge_set_keys)
                
                for kw in processed_keywords_for_product_set:
                    if kw not in individual_keywords_already_merged:
                        final_keywords_for_product_set.add(kw)
                
                products_with_final_keywords.append((p, list(final_keywords_for_product_set)))

            word_super_counts = defaultdict(set)
            for p_tuple_entry, final_kw_list_for_p in products_with_final_keywords:
                p_data = p_tuple_entry 
                for w in final_kw_list_for_p: 
                    word_super_counts[w].add(p_data["Supermarkt"])
            # --- EINDE AANGEPAST CATEGORIE-LOGICA BLOK ---
            
            radio_options_for_display_struct = []
            generic_products_by_super = defaultdict(list)
            # Gebruik products_with_final_keywords om generic_products_by_super te vullen
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
                    "count_for_sort": num_supers_for_generic + 0.5 
                })

            temp_specific_categories = []
            for word, supers_set in word_super_counts.items():
                if len(supers_set) > 1: 
                    temp_specific_categories.append({
                        "display_text": f"{word.capitalize()} ({len(supers_set)})", 
                        "actual_name": word,
                        "is_generic": False, 
                        "count_for_sort": len(supers_set)
                    })
            
            temp_specific_categories.sort(key=lambda x: x["count_for_sort"], reverse=True)
            radio_options_for_display_struct.extend(temp_specific_categories)
            
            unique_display_options = []
            seen_display_texts = set()
            for option in radio_options_for_display_struct:
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
                    if single_selected_option_data: default_combined_name = single_selected_option_data["actual_name"].capitalize()
                elif len(selected_sub_category_display_strings) > 1: default_combined_name = f"{search_term.capitalize()} (gecombineerd)"
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
                                main_step = 1.0 if is_stuks else 0.01 
                                main_format = "%d" if is_stuks else "%.2f"

                                main_qty_key = f"main_qty_COMBINED_{combined_category_name}_{unit_option_comb}_{search_term}"
                                main_qty = st.number_input(
                                    "Basishoeveelheid",
                                    min_value=0.0, value=st.session_state.get(main_qty_key, 0.0), step=main_step, format=main_format,
                                    key=main_qty_key, label_visibility="collapsed", 
                                    help=f"Voer de gewenste basishoeveelheid in {unit_option_comb} in."
                                )
                                
                                tolerance_percentage_input = 0
                                abs_tolerance_value_calculated = 0
                                
                                if main_qty > 0:
                                    slider_key = f"tol_perc_slider_{combined_category_name}_{unit_option_comb}_{search_term}"
                                    
                                    # Bereken eerst de actuele absolute tolerantie voor weergave
                                    current_slider_percentage = st.session_state.get(slider_key, 0)
                                    current_abs_tolerance = 0
                                    if current_slider_percentage > 0:
                                        current_abs_tolerance = round(main_qty * (current_slider_percentage / 100.0))
                                        if current_abs_tolerance == 0 and (main_qty * (current_slider_percentage / 100.0)) >= 0.5:
                                            if main_qty >= 2 or not is_stuks: current_abs_tolerance = 1
                                        max_abs_tol_allowed_display = math.floor(0.5 * main_qty)
                                        if current_abs_tolerance > max_abs_tol_allowed_display: current_abs_tolerance = max_abs_tol_allowed_display
                                        if current_abs_tolerance < 1: current_abs_tolerance = 0
                                    
                                    if current_slider_percentage > 0 and current_abs_tolerance > 0:
                                        st.caption(f"Geselecteerde tolerantie: {main_qty} {unit_option_comb} **± {current_abs_tolerance} {unit_option_comb}** ({current_slider_percentage}%)")
                                    elif current_slider_percentage > 0 and current_abs_tolerance == 0:
                                         st.caption(f"Tolerantie van {current_slider_percentage}% is te klein (resulteert in ±0 {unit_option_comb}).")
                                    else:
                                        st.caption("Geen tolerantie geselecteerd (0%). Sleep slider om in te stellen.")

                                    tolerance_percentage_input = st.slider(
                                        f"Tolerantie (±%)",
                                        min_value=0, max_value=50, 
                                        value=current_slider_percentage, # Gebruik waarde uit session state
                                        step=1, format="%d%%",
                                        key=slider_key, 
                                        help=f"Max 50% van {main_qty} {unit_option_comb}."
                                    )
                                    
                                    if tolerance_percentage_input > 0: # Herbereken na slider interactie
                                        abs_tolerance_value_calculated = round(main_qty * (tolerance_percentage_input / 100.0))
                                        if abs_tolerance_value_calculated == 0 and (main_qty * (tolerance_percentage_input / 100.0)) >= 0.5:
                                            if main_qty >= 2 or not is_stuks : abs_tolerance_value_calculated = 1
                                        max_abs_tol_allowed_calc = math.floor(0.5 * main_qty)
                                        if abs_tolerance_value_calculated > max_abs_tol_allowed_calc: abs_tolerance_value_calculated = max_abs_tol_allowed_calc
                                        if abs_tolerance_value_calculated < 1: abs_tolerance_value_calculated = 0
                                else: 
                                    st.caption("Voer eerst een basishoeveelheid (>0) in om tolerantie in te stellen.")
                                
                                if main_qty > 0: 
                                    desired_quantities_inputs_combined.append({
                                        "Hoeveelheid": int(main_qty) if is_stuks and main_qty == int(main_qty) else main_qty, 
                                        "Eenheid": unit_option_comb,
                                        "TolerantiePercentage": tolerance_percentage_input,
                                        "TolerantieWaarde": int(abs_tolerance_value_calculated), 
                                        "TolerantieEenheid": unit_option_comb if tolerance_percentage_input > 0 and abs_tolerance_value_calculated > 0 else None
                                    })
                                st.markdown("---") 
                            
                            if st.button(f"Voeg '{combined_category_name}' met opgegeven opties toe aan lijstje", key=f"add_btn_COMBINED_{combined_category_name}_{search_term}"):
                                add_to_shopping_list(combined_category_name, desired_quantities_inputs_combined, aggregated_products)
                        else: st.info(f"Geen bruikbare eenheden voor producten in '{combined_category_name}'.")
                    else: st.info(f"Geen producten voor combinatie van sub-categorieën.")
        elif search_term: st.info(f"Geen producten gevonden die '{search_term}' bevatten.")

    st.header("Jouw Mandje") 
    if st.session_state.shopping_list:
        list_for_display = []
        for idx, item_data_main in enumerate(st.session_state.shopping_list): 
            qty_strs = []
            for dq in item_data_main["DesiredQuantities"]:
                hoeveelheid_display = int(dq['Hoeveelheid']) if dq['Eenheid'].lower() == "stuks" and dq['Hoeveelheid'] == int(dq['Hoeveelheid']) else dq['Hoeveelheid']
                qty_str = f"{hoeveelheid_display} {dq['Eenheid']}"
                if dq.get("TolerantiePercentage", 0) > 0 and dq.get("TolerantieWaarde", 0) > 0:
                    qty_str += f" (±{int(dq['TolerantieWaarde'])} {dq['TolerantieEenheid']})"
                qty_strs.append(qty_str)
            list_for_display.append({"Nr.": idx + 1, "Categorie": item_data_main["Categorie"], "Gewenste 'OF'-opties": ", ".join(qty_strs) if qty_strs else "Geen specifieke hoeveelheden"})
        st.dataframe(list_for_display, hide_index=True, use_container_width=True)
        if st.button("Vergelijk Prijzen", key="vergelijk_prijzen_hoofd_knop"):
            run_comparison_and_store_results(st.session_state.shopping_list) 
            st.session_state.active_swap = None; st.rerun() 
    else: 
        st.info("Je boodschappenlijstje is leeg.") 
        if 'comparison_results' in st.session_state: del st.session_state.comparison_results
        if 'active_swap' in st.session_state: del st.session_state.active_swap

    if 'comparison_results' in st.session_state and st.session_state.comparison_results is not None:
        results = st.session_state.comparison_results; summary_entries = results.get("summary", []) 
        details_data = results.get("details", defaultdict(list)); item_status_data = results.get("item_status", defaultdict(lambda: {"gevonden_categorien": set(), "ontbrekende_categorien": set()}))
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
                with st.container(): 
                    st.markdown(f"#### {sup_name} (Totaal: {entry['Totaalprijs_str']})")
                    current_supermarket_details_list = details_data.get(sup_name, [])
                    sorted_rows_for_sup = sorted(current_supermarket_details_list, key=lambda x: x["Categorie"]) 
                    if sorted_rows_for_sup:
                        for detail_idx, row_data in enumerate(sorted_rows_for_sup):
                            item_cols = st.columns([5, 1]) 
                            with item_cols[0]:
                                item_display_name = row_data['Naam']; 
                                if row_data.get('is_swapped'): item_display_name += " (Gewisseld)"
                                st.markdown(f"**{row_data['Categorie']}**: {item_display_name}")
                                gekozen_optie_display = row_data['Gekozen voor optie']
                                st.caption(f"Keuze: {gekozen_optie_display} | Aantal: {row_data['Aantal Pakken']}x | Kosten: {row_data['Kosten Totaal']}")
                            with item_cols[1]:
                                swap_button_key = f"start_swap_{sup_name.replace(' ', '_')}_{detail_idx}_{summary_idx}_{row_data['Categorie'].replace(' ', '_')}" 
                                if st.button("🔁", key=swap_button_key, help=f"Wissel '{row_data['Naam']}' bij {sup_name}"):
                                    original_sl_item_idx = row_data.get("shopping_list_item_idx", -1)
                                    original_desired_quantities_for_swap = row_data.get("_original_desired_quantities", []) 
                                    if not original_desired_quantities_for_swap and original_sl_item_idx != -1 and original_sl_item_idx < len(st.session_state.shopping_list): 
                                        original_desired_quantities_for_swap = st.session_state.shopping_list[original_sl_item_idx]["DesiredQuantities"]
                                    st.session_state.active_swap = {"sup_name": sup_name, "detail_item_index": detail_idx, "original_item_data": dict(row_data), "original_category_name": row_data["Categorie"], "original_shopping_list_item_idx": original_sl_item_idx, "original_desired_quantities": original_desired_quantities_for_swap }
                                    if f"swap_search_term_{sup_name}_{detail_idx}" in st.session_state: del st.session_state[f"swap_search_term_{sup_name}_{detail_idx}"]
                                    st.rerun()
                            st.markdown("---") 
                        ontbrekende_voor_sup = item_status_data.get(sup_name, {}).get("ontbrekende_categorien", set())
                        if ontbrekende_voor_sup: st.caption(f"Ontbrekende categorieën voor {sup_name}: {', '.join(sorted(list(ontbrekende_voor_sup)))}")
                        go_shopping_key = f"go_shopping_{sup_name.replace(' ', '_')}_{summary_idx}"
                        if st.button(f"🛍️ Boodschappen doen met lijst van {sup_name}", key=go_shopping_key):
                            st.session_state.shopping_mode_active = True; st.session_state.active_shopping_supermarket_name = sup_name
                            st.session_state.active_shopping_list_details = [dict(item) for item in sorted_rows_for_sup]; st.rerun()
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