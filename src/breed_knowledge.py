"""Farmer & general-public knowledge base for all 68 classifier breeds.

Turns a breed prediction into practical advice: origin, local names, utility,
milk yield, visual verification markers, feeding, climate/disease notes, market
price and applicable government schemes — then specialises that advice by
gender and age group.

DATA POLICY
  * ``CURATED`` entries follow NBAGR / ICAR breed descriptions and standard
    Indian feeding norms. Figures are *typical field ranges*, not guarantees;
    every advisory carries that caveat.
  * Rare breeds without a reliable published production figure use the utility
    template (milch / draught / dual / dwarf) and are marked
    ``data_confidence: "regional_default"`` so the UI can say so.
  * ``physical_markers`` are derived from ``data/breed_traits.json`` (the same
    curated morphology the model was trained on) plus each entry's
    key distinguisher, so the farmer can visually sanity-check the AI.

GENERATED FILE
  ``data/breed_knowledge.json`` is auto-created from this module on first use.
  It is safe to hand-edit afterwards: the loader only regenerates *missing*
  breed keys, and ``--regenerate`` rewrites the whole file.

PUBLIC API
  ``get_breed_knowledge(species, breed)``   -> one breed's knowledge dict
  ``get_farmer_advisory(species, breed, gender, age_group)`` -> advisory dict
  ``list_breeds()``                        -> [(species, slug, display_name)]
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

from .config import PROJECT_ROOT, TRAIT_FILE

KNOWLEDGE_FILE = os.path.join(PROJECT_ROOT, "data", "breed_knowledge.json")
SCHEMA_VERSION = 1

CATTLE_BREEDS = (
    "amritmahal", "ayrshire", "bachaur", "badri", "bargur", "bhelai", "dagri",
    "dangi", "deoni", "gangatari", "gaolao", "ghumsari", "gir", "hallikar",
    "hariana", "himachali_pahari", "kangayam", "kankrej", "kenkatha", "khariar",
    "kherigarh", "khillari", "konkan_kapila", "kosali", "krishna_valley",
    "ladakhi", "lakhimi", "malnad_gidda", "malvi", "mewati", "motu", "nagori",
    "nari", "nimari", "ongole", "poda_thirupu", "ponwar", "pulikulam",
    "punganur", "purnea", "rathi", "red_kandhari", "red_sindhi", "sahiwal",
    "shweta_kapila", "siri", "tharparkar", "thutho", "umblachery", "vechur",
)
BUFFALO_BREEDS = (
    "banni", "bargur", "bhadwari", "chhattisgarhi", "chilika", "gojri",
    "jaffarabadi", "kalahandi", "luit", "marathwada", "mehsana", "murrah",
    "nagpuri", "nili_ravi", "other_buffalo", "pandharpuri", "surti", "toda",
)

# Extra cattle classes reachable through ALIASES (commercial crossbred types).
# They are documented but excluded from the 68-breed count.
EXTRA_CATTLE_BREEDS = ("holstein_friesian", "jersey")

# Alternate spellings found in datasets / the traits file.
ALIASES = {
    "belahi": "bhelai", "gangatiri": "gangatari", "ghumusari": "ghumsari",
    "khillar": "khillari", "poda_thurpu": "poda_thirupu",
    "bhadawari": "bhadwari", "marathwadi": "marathwada",
    "alambadi": "other_buffalo", "hf_cross": "holstein_friesian",
    "holstein": "holstein_friesian", "jersey_cross": "jersey",
    "crossbred_cattle": "holstein_friesian", "mixed_buffalo": "other_buffalo",
}

# ---------------------------------------------------------------------------
#  Origin / native tract
# ---------------------------------------------------------------------------

ORIGIN_REGION = {
    # cattle
    "amritmahal": "Chitradurga, Hassan and Mysuru districts, Karnataka",
    "ayrshire": "Ayrshire, Scotland; kept on Indian temperate / high-altitude farms",
    "bachaur": "Darbhanga, Madhubani and Sitamarhi, Bihar (Nepal border belt)",
    "badri": "Chamoli, Rudraprayag and Pauri Garhwal, Uttarakhand",
    "bargur": "Bargur hills, Krishnagiri and Erode, Tamil Nadu",
    "bhelai": "Bhagalpur and Banka, Bihar",
    "dagri": "Panchmahal and Dahod, Gujarat",
    "dangi": "Ahmednagar, Nashik and Pune, Maharashtra",
    "deoni": "Latur, Osmanabad and Parbhani, Maharashtra; Bidar, Karnataka",
    "gangatari": "Ballia, Ghazipur and eastern Uttar Pradesh",
    "gaolao": "Wardha, Nagpur and Amravati, Maharashtra; Balaghat, Madhya Pradesh",
    "ghumsari": "Ganjam and adjoining districts, Odisha",
    "gir": "Saurashtra — Junagadh, Amreli, Bhavnagar and Porbandar, Gujarat",
    "hallikar": "Tumakuru, Mysuru and Mandya, Karnataka",
    "hariana": "Rohtak, Hisar, Jind and Gurugram, Haryana; western Uttar Pradesh",
    "himachali_pahari": "Kangra, Mandi and Shimla hills, Himachal Pradesh",
    "kangayam": "Erode, Tiruppur, Karur and Dindigul, Tamil Nadu",
    "kankrej": "Banaskantha and Kutch, Gujarat; Jalore and Barmer, Rajasthan",
    "kenkatha": "Bundelkhand — Jhansi and Banda, Uttar Pradesh; Chhatarpur, Madhya Pradesh",
    "khariar": "Nuapada and Kalahandi, Odisha",
    "kherigarh": "Lakhimpur Kheri and Sitapur, Uttar Pradesh",
    "khillari": "Solapur, Ahmednagar and Satara, Maharashtra",
    "konkan_kapila": "Sindhudurg, Ratnagiri and Raigad, Maharashtra; Goa",
    "kosali": "Chhattisgarh plains — Bilaspur, Raipur and Durg",
    "krishna_valley": "Bagalkot, Vijayapura and Belagavi, Karnataka",
    "ladakhi": "Leh and Kargil, Ladakh",
    "lakhimi": "Brahmaputra valley, Assam",
    "malnad_gidda": "Shivamogga, Chikkamagaluru and Uttara Kannada, Karnataka",
    "malvi": "Ujjain, Indore, Dewas and Shajapur, Madhya Pradesh",
    "mewati": "Alwar, Bharatpur and Dholpur, Rajasthan; Mewat, Haryana",
    "motu": "Koraput and Malkangiri, Odisha",
    "nagori": "Nagaur, Jodhpur and Bikaner, Rajasthan",
    "nari": "Patan and Mehsana, north Gujarat",
    "nimari": "Khandwa, Khargone and Burhanpur, Madhya Pradesh",
    "ongole": "Prakasam, Guntur and Nellore, Andhra Pradesh",
    "poda_thirupu": "Srikakulam and Vizianagaram, Andhra Pradesh",
    "ponwar": "Pilibhit and Shahjahanpur, Uttar Pradesh",
    "pulikulam": "Madurai, Sivaganga and Virudhunagar, Tamil Nadu",
    "punganur": "Chittoor and Palamaner, Andhra Pradesh",
    "purnea": "Purnea, Katihar and Araria, Bihar",
    "rathi": "Bikaner, Ganganagar and Hanumangarh, Rajasthan",
    "red_kandhari": "Nanded, Parbhani and Hingoli, Maharashtra",
    "red_sindhi": "Sindh (Pakistan); Indian herds in Gujarat, Rajasthan and Kerala",
    "sahiwal": "Firozpur, Amritsar and Patiala belt, Punjab; also Rajasthan and Haryana",
    "shweta_kapila": "Konkan coast, Maharashtra (recently registered breed)",
    "siri": "Sikkim, Darjeeling (West Bengal) and the Bhutan border",
    "tharparkar": "Jodhpur, Barmer and Jaisalmer, Rajasthan; Rann of Kutch",
    "thutho": "Mokokchung and Wokha, Nagaland",
    "umblachery": "Thanjavur, Nagapattinam and Tiruvarur, Tamil Nadu",
    "vechur": "Kottayam and Thrissur, Kerala",
    # buffalo
    "banni": "Banni grassland, Kutch, Gujarat",
    "bhadwari": "Etawah and Agra, Uttar Pradesh; Bhind and Morena, Madhya Pradesh",
    "chhattisgarhi": "Chhattisgarh plains",
    "chilika": "Chilika lagoon belt — Puri, Khordha and Ganjam, Odisha",
    "gojri": "Rajasthan and Gujarat (migratory pastoral herds)",
    "jaffarabadi": "Junagadh, Gir Somnath and Kutch, Gujarat",
    "kalahandi": "Kalahandi and Nuapada, Odisha",
    "luit": "Brahmaputra valley, Assam",
    "marathwada": "Latur, Nanded and Parbhani, Maharashtra",
    "mehsana": "Mehsana, Banaskantha and Sabarkantha, Gujarat",
    "murrah": "Rohtak, Hisar, Jind and Bhiwani, Haryana; Patiala, Punjab",
    "nagpuri": "Nagpur, Wardha and Amravati, Maharashtra",
    "nili_ravi": "Firozpur and Amritsar, Punjab; Rajasthan; Punjab (Pakistan)",
    "other_buffalo": "Non-descript / mixed buffalo across India",
    "pandharpuri": "Solapur, Kolhapur and Sangli, Maharashtra",
    "surti": "Anand, Kheda and Vadodara, Gujarat",
    "toda": "Nilgiri hills, Tamil Nadu",
    # crossbred / composite classes
    "holstein_friesian": "Holstein-Friesian origin (Netherlands); crossbred herds across India",
    "jersey": "Jersey Island origin; crossbred herds across India",
}

# Species-specific corrections where a slug exists in both class maps.
ORIGIN_OVERRIDES = {
    ("Buffalo", "bargur"): ("Bargur hills, Tamil Nadu — listed as a buffalo class in the "
                            "training data; verify against source imagery"),
}

# ---------------------------------------------------------------------------
#  Local names (Tamil / Hindi). Breeds whose native script spelling is not
#  published consistently fall back to the English name at build time.
# ---------------------------------------------------------------------------

LOCAL_NAMES = {
    "amritmahal": {"hi": "अमृतमहल", "ta": "அமிர்தமகால்"},
    "ayrshire": {"hi": "आयरशायर"},
    "bachaur": {"hi": "बछौर", "ta": "பசௌர்"},
    "badri": {"hi": "बद्री", "ta": "பத்ரி"},
    "bargur": {"hi": "बरगूर", "ta": "பர்கூர்"},
    "bhelai": {"hi": "भेलाई"},
    "dagri": {"hi": "डागरी"},
    "dangi": {"hi": "डांगी"},
    "deoni": {"hi": "देवनी", "ta": "தேவனி"},
    "gangatari": {"hi": "गंगातरी"},
    "gaolao": {"hi": "गाओलाओ"},
    "ghumsari": {"hi": "घुमुसरी"},
    "gir": {"hi": "गीर", "ta": "கிர்"},
    "hallikar": {"hi": "हल्लीकर", "ta": "ஹல்லிகர்"},
    "hariana": {"hi": "हरियाणा", "ta": "ஹரியானா"},
    "himachali_pahari": {"hi": "हिमाचली पहाड़ी"},
    "kangayam": {"hi": "कांगायम", "ta": "கங்கையம்"},
    "kankrej": {"hi": "कांकरेज", "ta": "காங்க்ரேஜ்"},
    "kenkatha": {"hi": "केनकाथा"},
    "khariar": {"hi": "खरियार"},
    "kherigarh": {"hi": "खेरीगढ़"},
    "khillari": {"hi": "खिल्लारी", "ta": "கில்லாரி"},
    "konkan_kapila": {"hi": "कोंकण कपिला", "ta": "கோங்கன் கபிலா"},
    "kosali": {"hi": "कोसली", "ta": "கோசலி"},
    "krishna_valley": {"hi": "कृष्णा वैली", "ta": "கிருஷ்ணா வேலி"},
    "ladakhi": {"hi": "लद्दाखी", "ta": "லடாக்கி"},
    "lakhimi": {"hi": "लखिमी", "ta": "லக்ஷ்மி"},
    "malnad_gidda": {"hi": "मालनाड गिड्डा", "ta": "மல்னாடு கிட்டா"},
    "malvi": {"hi": "मालवी", "ta": "மால்வி"},
    "mewati": {"hi": "मेवाती", "ta": "மேவாதி"},
    "motu": {"hi": "मोतू"},
    "nagori": {"hi": "नागोरी", "ta": "நாகோரி"},
    "nari": {"hi": "नारी"},
    "nimari": {"hi": "निमाड़ी", "ta": "நிமாரி"},
    "ongole": {"hi": "ओंगोल", "ta": "ஓங்கோல்"},
    "poda_thirupu": {"hi": "पोडा तिरुपु"},
    "ponwar": {"hi": "पोनवार"},
    "pulikulam": {"hi": "पुलिकुलम", "ta": "புலிக்குளம்"},
    "punganur": {"hi": "पुंगनूर", "ta": "பூங்கனூர்"},
    "purnea": {"hi": "पूर्णिया"},
    "rathi": {"hi": "राठी", "ta": "ராத்தி"},
    "red_kandhari": {"hi": "लाल कंधारी", "ta": "ரெட் கந்தாரி"},
    "red_sindhi": {"hi": "लाल सिंधी", "ta": "ரெட் சிந்தி"},
    "sahiwal": {"hi": "साहीवाल", "ta": "சாஹிவால்"},
    "shweta_kapila": {"hi": "श्वेता कपिला"},
    "siri": {"hi": "सिरी", "ta": "சிரி"},
    "tharparkar": {"hi": "थारपारकर", "ta": "தார்பார்க்கர்"},
    "thutho": {"hi": "थुथो"},
    "umblachery": {"hi": "उम्बलाचेरी", "ta": "உம்பளச்சேரி"},
    "vechur": {"hi": "वेचूर", "ta": "வீச்சூர்"},
    # buffalo
    "banni": {"hi": "बन्नी", "ta": "பன்னி"},
    "bhadwari": {"hi": "भदावरी", "ta": "பதாவரி"},
    "chhattisgarhi": {"hi": "छत्तीसगढ़ी", "ta": "சத்தீஸ்கர்"},
    "chilika": {"hi": "चिलिका", "ta": "சிலிகா"},
    "gojri": {"hi": "गोजरी", "ta": "கோஜ்ரி"},
    "jaffarabadi": {"hi": "जाफराबादी", "ta": "ஜாஃபராபாதி"},
    "kalahandi": {"hi": "कालाहांडी", "ta": "கலாஹண்டி"},
    "luit": {"hi": "लुइत", "ta": "லுயித்"},
    "marathwada": {"hi": "मराठवाड़ा", "ta": "மராத்வாடா"},
    "mehsana": {"hi": "मेहसाणा", "ta": "மெஹ்சானா"},
    "murrah": {"hi": "मुर्राह", "ta": "முர்ரா"},
    "nagpuri": {"hi": "नागपुरी", "ta": "நாகபுரி"},
    "nili_ravi": {"hi": "नीली रावी", "ta": "நீலி ராவி"},
    "other_buffalo": {"hi": "मिश्रित भैंस", "ta": "கலப்பு எருமை"},
    "pandharpuri": {"hi": "पंढरपुरी", "ta": "பந்தர்புரி"},
    "surti": {"hi": "सुरती", "ta": "சுரதி"},
    "toda": {"hi": "तोडा", "ta": "தோடா"},
    "holstein_friesian": {"hi": "होलस्टीन फ्रीजियन", "ta": "ஹோல்ஸ்டீன் பிரீஸியன்"},
    "jersey": {"hi": "जर्सी", "ta": "ஜெர்சி"},
}

# ---------------------------------------------------------------------------
#  Production / market table (curated unless noted)
#  slug: (utility, daily L (lo,hi), lactation kg (lo,hi), fat % (lo,hi),
#         price INR (lo,hi), a2, confidence, note)
# ---------------------------------------------------------------------------

MILCH, DRAUGHT, DUAL, DWARF = (
    "Milch (Dairy)", "Draught (Work)", "Dual-Purpose", "Ornamental / Dwarf Dairy")

BREED_DATA_CATTLE = {
    # ------------------------------------------------------------- cattle
    "amritmahal": (DRAUGHT, (2, 4), (400, 900), (4.5, 5.5), (35000, 70000), True, "curated",
                   "Hallikar-family draft cattle; the name comes from the Mysore army's cattle department."),
    "ayrshire": (MILCH, (12, 18), (3000, 5000), (3.5, 4.2), (45000, 90000), False, "curated",
                 "Temperate Bos taurus breed, best suited to cool highlands and crossbreeding programmes."),
    "bachaur": (DRAUGHT, (2, 4), (400, 800), (4.5, 5.0), (25000, 50000), True, "curated",
                "Compact all-white draft breed of the north Bihar plains."),
    "badri": (DUAL, (2, 4), (500, 900), (4.5, 5.5), (25000, 55000), True, "curated",
              "Hill breed; small, sure-footed and tolerant of cold, humid Himalayan conditions."),
    "bargur": (DRAUGHT, (2, 4), (400, 800), (4.5, 5.5), (25000, 50000), True, "curated",
               "Brown-and-white hill cattle used for ploughing and light transport."),
    "bhelai": (DRAUGHT, (2, 4), (400, 800), (4.5, 5.0), (22000, 45000), True, "regional_default",
               "Rare Bihar draft type; production figures are regional estimates."),
    "dagri": (DRAUGHT, (2, 4), (400, 800), (4.5, 5.0), (25000, 50000), True, "curated",
              "Grey draft breed from the Panchmahal hills; hardy on low-input grazing."),
    "dangi": (DRAUGHT, (2, 4), (400, 800), (4.5, 5.5), (25000, 55000), True, "curated",
              "Rain-tolerant draft breed with a heavy dewlap and spotted coat."),
    "deoni": (DUAL, (4, 7), (900, 1500), (4.5, 5.5), (30000, 65000), True, "curated",
              "Black-and-white patched dual-purpose breed of the Deccan plateau."),
    "gangatari": (DUAL, (3, 5), (700, 1200), (4.5, 5.5), (25000, 50000), True, "regional_default",
                  "Eastern-UP dual-purpose type; figures are regional estimates."),
    "gaolao": (DRAUGHT, (3, 5), (600, 1100), (4.5, 5.5), (25000, 55000), True, "curated",
               "White-grey draft breed of the Vidarbha region; good heat tolerance."),
    "ghumsari": (DRAUGHT, (2, 4), (450, 900), (4.5, 5.0), (22000, 45000), True, "regional_default",
                 "Rare Odisha draft breed; figures are regional estimates."),
    "gir": (MILCH, (8, 12), (1800, 3000), (4.5, 5.0), (45000, 100000), True, "curated",
            "India's flagship dairy zebu; A2 beta-casein is typical and verified by lab test, not by sight."),
    "hallikar": (DRAUGHT, (2, 4), (400, 800), (4.5, 5.5), (30000, 65000), True, "curated",
                 "Lean grey draft breed; long backward-sweeping horns and great stamina."),
    "hariana": (DUAL, (5, 8), (1000, 1800), (4.5, 5.5), (30000, 60000), True, "curated",
                "Tall white-grey dual-purpose breed; widely used in crossbreeding."),
    "himachali_pahari": (DRAUGHT, (2, 4), (400, 800), (4.5, 5.5), (20000, 45000), True, "regional_default",
                         "Hill draft type of Himachal Pradesh; figures are regional estimates."),
    "kangayam": (DRAUGHT, (2, 4), (400, 900), (4.5, 5.5), (35000, 75000), True, "curated",
                 "Tamil Nadu's premier draft breed, prized for heat tolerance and sure footing."),
    "kankrej": (DUAL, (5, 8), (1200, 2000), (4.5, 5.0), (40000, 80000), True, "curated",
                "Silver-grey dual-purpose breed famous for the fast 'Bhare' gait."),
    "kenkatha": (DRAUGHT, (2, 4), (400, 800), (4.5, 5.0), (22000, 45000), True, "curated",
                 "Bundelkhand draft breed; compact, tough and low-maintenance."),
    "khariar": (DRAUGHT, (2, 3), (350, 700), (4.5, 5.0), (20000, 40000), True, "curated",
                "Odisha draft breed similar to Hariana but smaller and shorter-faced."),
    "kherigarh": (DRAUGHT, (2, 4), (450, 900), (4.5, 5.5), (25000, 50000), True, "curated",
                  "Light draft breed from the Kheri region of Uttar Pradesh."),
    "khillari": (DRAUGHT, (2, 4), (400, 800), (4.5, 5.5), (30000, 65000), True, "curated",
                 "Fast, spirited draft breed of the Deccan; used for bullock-cart racing."),
    "konkan_kapila": (DWARF, (1, 3), (300, 800), (4.5, 6.0), (25000, 60000), True, "curated",
                      "Small Konkan coast breed; noted for rich milk and humid-climate tolerance."),
    "kosali": (DUAL, (3, 5), (700, 1200), (4.5, 5.5), (25000, 50000), True, "curated",
               "Chhattisgarh dual-purpose breed; emerging dairy use in the central plains."),
    "krishna_valley": (DRAUGHT, (4, 6), (900, 1500), (4.5, 5.5), (35000, 70000), True, "curated",
                       "Heavy draft breed of the Krishna river valley; also gives fair milk."),
    "ladakhi": (DRAUGHT, (1, 3), (300, 700), (4.5, 5.5), (25000, 50000), True, "curated",
                "High-altitude breed; extremely cold tolerant, small and hardy."),
    "lakhimi": (DUAL, (3, 5), (700, 1200), (4.5, 5.5), (25000, 50000), True, "curated",
                "Assam dual-purpose breed adapted to high rainfall and humidity."),
    "malnad_gidda": (DWARF, (1, 3), (300, 800), (4.5, 6.0), (20000, 45000), True, "curated",
                     "Western Ghats dwarf breed; grazes on hilly terrain and gives fat-rich milk."),
    "malvi": (DRAUGHT, (3, 5), (600, 1100), (4.5, 5.5), (25000, 55000), True, "curated",
              "Malwa plateau draft breed with a prominent hump and long face."),
    "mewati": (DUAL, (4, 6), (900, 1500), (4.5, 5.5), (30000, 60000), True, "curated",
               "Heavier cousin of Hariana from the Mewat belt; dual-purpose."),
    "motu": (DRAUGHT, (2, 4), (400, 800), (4.5, 5.0), (20000, 40000), True, "regional_default",
             "Rare Koraput draft type; figures are regional estimates."),
    "nagori": (DRAUGHT, (2, 4), (400, 800), (4.5, 5.5), (25000, 55000), True, "curated",
               "Tall, long-faced white-grey draft breed of the Thar desert."),
    "nari": (DRAUGHT, (2, 4), (450, 900), (4.5, 5.0), (22000, 45000), True, "regional_default",
             "Rare north-Gujarat draft type; figures are regional estimates."),
    "nimari": (DRAUGHT, (3, 5), (600, 1100), (4.5, 5.5), (25000, 50000), True, "curated",
               "Red-and-white patched Nimar draft breed."),
    "ongole": (DUAL, (5, 8), (1000, 1800), (4.5, 5.5), (45000, 90000), True, "curated",
               "Massive-humped white breed; ancestor of the American Brahman."),
    "poda_thirupu": (DRAUGHT, (1, 3), (300, 600), (4.5, 5.5), (15000, 35000), True, "regional_default",
                     "Rare small Andhra draft type; figures are regional estimates."),
    "ponwar": (DRAUGHT, (2, 4), (400, 800), (4.5, 5.0), (20000, 40000), True, "curated",
               "Terai draft breed of Pilibhit; compact and hardy."),
    "pulikulam": (DRAUGHT, (1, 3), (300, 600), (4.5, 5.5), (25000, 55000), True, "curated",
                  "Madurai-belt draft breed associated with Jallikattu; very hardy."),
    "punganur": (DWARF, (2, 4), (400, 900), (4.0, 6.0), (30000, 80000), True, "curated",
                 "One of the world's smallest cattle breeds; premium pet/orphan-calf value."),
    "purnea": (DRAUGHT, (3, 5), (600, 1100), (4.5, 5.5), (22000, 45000), True, "curated",
               "Eastern Bihar draft breed suited to humid, flood-prone plains."),
    "rathi": (MILCH, (6, 9), (1500, 2500), (4.5, 5.5), (40000, 80000), True, "curated",
              "Thar-desert dairy breed with brown-and-white patches and long ears."),
    "red_kandhari": (DRAUGHT, (3, 5), (600, 1200), (4.5, 5.5), (30000, 60000), True, "curated",
                     "Deep-red draft breed of the Marathwada region."),
    "red_sindhi": (MILCH, (7, 10), (1500, 2500), (4.5, 5.5), (40000, 85000), True, "curated",
                   "Uniform red dairy zebu; heat tolerant and widely used in crossbreeding."),
    "sahiwal": (MILCH, (8, 12), (2000, 3000), (4.5, 5.5), (50000, 110000), True, "curated",
                "India's best dairy zebu; deep red coat, loose skin and a long face."),
    "shweta_kapila": (DWARF, (2, 4), (400, 900), (4.5, 5.5), (25000, 55000), True, "regional_default",
                      "Recently registered white Kapila type from the Konkan coast; limited data."),
    "siri": (DUAL, (3, 5), (700, 1200), (4.5, 5.5), (30000, 60000), True, "curated",
             "Himalayan dual-purpose breed with a heavy head and long curved horns."),
    "tharparkar": (DUAL, (6, 10), (1500, 2500), (4.5, 5.5), (40000, 85000), True, "curated",
                   "Light-grey Thar desert breed; tolerates heat and scarce water."),
    "thutho": (DRAUGHT, (1, 3), (300, 700), (4.5, 5.5), (20000, 45000), True, "regional_default",
               "Nagaland hill draft type; figures are regional estimates."),
    "umblachery": (DRAUGHT, (2, 4), (400, 800), (4.5, 5.5), (30000, 65000), True, "curated",
                   "Grey paddy-field breed of the Cauvery delta; compact and short-horned."),
    "vechur": (DWARF, (2, 4), (400, 900), (4.5, 6.0), (35000, 100000), True, "curated",
               "Kerala dwarf breed (among the world's smallest); A2 milk commands a premium."),
    "holstein_friesian": (MILCH, (15, 25), (4000, 7000), (3.5, 4.2), (55000, 120000), False, "curated",
                          "High-yield temperate cross; needs cooling, balanced TMR and careful breeding."),
    "jersey": (MILCH, (10, 16), (2500, 4500), (4.0, 5.0), (40000, 90000), False, "curated",
               "Smaller crossbred milch type; better heat tolerance and fat than HF cross."),
}

BREED_DATA_BUFFALO = {
    # ------------------------------------------------------------ buffalo
    "banni": (MILCH, (5, 8), (1000, 1600), (6.5, 7.5), (45000, 90000), True, "curated",
              "Hardy Kutch grassland buffalo; thrives on sparse, saline grazing."),
    "bargur": (DUAL, (3, 5), (600, 1000), (6.0, 7.0), (30000, 60000), True, "regional_default",
               "Listed as a buffalo class in the training data; verify against source imagery."),
    "bhadwari": (MILCH, (4, 7), (800, 1500), (8.0, 12.0), (40000, 90000), True, "curated",
                 "Highest-fat Indian buffalo (8-12%); copper sheen on a black coat, ideal for ghee."),
    "chhattisgarhi": (DUAL, (4, 6), (800, 1200), (6.0, 7.5), (35000, 70000), True, "curated",
                      "Central Indian dual-purpose buffalo; medium frame, curved horns."),
    "chilika": (MILCH, (3, 6), (700, 1200), (6.5, 7.5), (30000, 65000), True, "curated",
                "Small coastal buffalo from the Chilika lagoon belt."),
    "gojri": (MILCH, (4, 6), (800, 1200), (6.0, 7.5), (30000, 65000), True, "curated",
              "Migratory pastoral buffalo; black or brown coat, good scavenging ability."),
    "jaffarabadi": (MILCH, (8, 12), (1800, 2700), (6.5, 8.0), (70000, 160000), True, "curated",
                    "Heaviest Indian buffalo breed; long flat drooping horns."),
    "kalahandi": (MILCH, (3, 5), (600, 1000), (6.5, 7.5), (30000, 60000), True, "curated",
                  "Odisha dairy buffalo; medium frame with gently curved horns."),
    "luit": (MILCH, (3, 6), (600, 1000), (6.5, 7.5), (30000, 65000), True, "curated",
             "Brahmaputra valley buffalo; compact with a deep body, suited to humid Assam."),
    "marathwada": (MILCH, (5, 8), (1000, 1500), (6.5, 8.0), (40000, 85000), True, "curated",
                   "Marathwada dairy type; close to Nagpuri with shorter horns."),
    "mehsana": (MILCH, (7, 11), (1500, 2200), (6.5, 7.5), (55000, 120000), True, "curated",
                "Murrah-Surti cross type; lighter frame, spiralled horns, steady yielder."),
    "murrah": (MILCH, (8, 14), (1800, 2800), (6.5, 8.0), (70000, 160000), True, "curated",
               "India's premier dairy buffalo; tightly curled horns on a jet-black frame."),
    "nagpuri": (MILCH, (5, 8), (1000, 1500), (6.5, 8.0), (40000, 85000), True, "curated",
                "Tall, lean Vidarbha buffalo with long flat swept-back horns."),
    "nili_ravi": (MILCH, (8, 12), (1800, 2500), (6.0, 7.5), (60000, 130000), True, "curated",
                  "Punjab dairy buffalo; white face and leg markings are the quickest cue."),
    "other_buffalo": (DUAL, (4, 7), (800, 1400), (6.0, 7.5), (40000, 90000), True, "regional_default",
                      "Non-descript buffalo; use the generic buffalo advisory and confirm locally."),
    "pandharpuri": (MILCH, (5, 8), (1000, 1600), (6.5, 8.0), (40000, 90000), True, "curated",
                    "Long sword-shaped horns are the defining feature of this Solapur breed."),
    "surti": (MILCH, (6, 9), (1200, 1800), (6.5, 7.5), (50000, 100000), True, "curated",
              "Sickle-shaped horns and two white chevrons on the neck."),
    "toda": (MILCH, (2, 4), (400, 700), (7.0, 8.0), (25000, 60000), True, "curated",
             "Small Nilgiri hill buffalo; high-fat milk from low-input grazing."),
}

# Breeds whose climate/disease profile deserves a specific note.
CLIMATE_NOTES = {
    "gir": ("Excellent heat tolerance; tick resistance is moderate — check ears and "
            "udder weekly in humid Saurashtra."),
    "sahiwal": ("Strong heat tolerance and tick resistance; prone to bloat on lush "
                "legume fodder, introduce greens gradually."),
    "murrah": ("Heat-tolerant but wallows to cool down; keep shade and water. Watch "
               "for mastitis in humid months."),
    "jaffarabadi": ("Good heat tolerance; heavy body needs shade and generous water. "
                    "Susceptible to foot problems in wet, muddy yards."),
    "bhadwari": ("Very heat tolerant; high-fat milk. Tick control is important in the "
                 "Bundelkhand summer."),
    "vechur": ("Humid-tropical adapted; very small body loses heat poorly in direct sun, "
               "so provide shade and ventilation."),
    "ladakhi": ("Extreme cold tolerance, poor heat tolerance; move to cooler shade above "
                "30 °C and protect from rain."),
    "toda": ("High-rainfall Nilgiri adaptation; watch for foot rot and liver fluke in "
             "wet grazing."),
    "banni": ("Saline-grassland adapted; tolerates poor fodder but needs clean water."),
    "ayrshire": ("Temperate breed — heat stress above 28-30 °C; best in hill/cool zones."),
    "holstein_friesian": ("Poor heat tolerance; needs fans/foggers and cool drinking water "
                          "in Indian summers."),
    "jersey": ("Moderate heat tolerance; smaller frame copes better than HF but still "
               "needs shade and water."),
}


def _normalize_slug(slug):
    key = str(slug or "").strip().lower().replace("-", "_").replace(" ", "_")
    return ALIASES.get(key, key)


def _normalize_species(species):
    key = str(species or "").strip().lower()
    if key.startswith("buff"):
        return "Buffalo"
    return "Cattle"


def _display_name(slug):
    special = {
        "nili_ravi": "Nili-Ravi", "malnad_gidda": "Malnad Gidda",
        "konkan_kapila": "Konkan Kapila", "red_sindhi": "Red Sindhi",
        "red_kandhari": "Red Kandhari", "krishna_valley": "Krishna Valley",
        "himachali_pahari": "Himachali Pahari", "poda_thirupu": "Poda Thirupu",
        "other_buffalo": "Other / Non-descript Buffalo",
        "holstein_friesian": "Holstein Friesian", "shweta_kapila": "Shweta Kapila",
        "gangatari": "Gangatari", "ghumsari": "Ghumsari", "bhelai": "Bhelai",
        "khillari": "Khillari", "bhadwari": "Bhadwari", "marathwada": "Marathwada",
    }
    return special.get(slug, slug.replace("_", " ").title())


# ---------------------------------------------------------------------------
#  Traits-derived visual markers
# ---------------------------------------------------------------------------

_TRAIT_ALIASES = {
    "belahi": "bhelai", "gangatiri": "gangatari", "ghumusari": "ghumsari",
    "khillar": "khillari", "poda_thurpu": "poda_thirupu",
    "bhadawari": "bhadwari", "marathwadi": "marathwada",
}


def _load_traits():
    """Load data/breed_traits.json keyed by (species, canonical slug)."""
    if not os.path.exists(TRAIT_FILE):
        return {}
    try:
        with open(TRAIT_FILE, encoding="utf-8") as f:
            spec = json.load(f)
    except Exception:
        return {}
    out = {}
    for sp, key in (("cattle", "Cattle"), ("buffalo", "Buffalo")):
        for raw, tv in (spec.get("breeds", {}).get(sp) or {}).items():
            slug = _TRAIT_ALIASES.get(raw, raw)
            out.setdefault((key, slug), tv)
    return out


MARKER_OVERRIDES = {
    ("Buffalo", "bargur"): [
        "The training data lists 'Bargur' as a buffalo class; confirm the animal is a buffalo before acting.",
        "Buffalo frame: black to brown coat, backswept horns, deep barrel.",
        "Treat all production figures here as low-confidence until verified locally.",
    ],
}


def _markers(slug, species, traits, data):
    override = MARKER_OVERRIDES.get((species, slug))
    if override:
        return override

    tv = traits.get((species, slug)) or {}
    markers = []
    coat = tv.get("coat")
    hump = tv.get("hump")
    horn = tv.get("horn")
    ear = tv.get("ear")
    dewlap = tv.get("dewlap")
    face = tv.get("face")
    size = tv.get("size")

    if coat or hump:
        bits = []
        if coat:
            bits.append(f"{humanize(coat, 'coat')} coat")
        if hump:
            bits.append(f"{humanize(hump, 'hump')} hump")
        markers.append("Body: " + " · ".join(bits) + ".")

    if horn:
        markers.append(f"Horns: {humanize(horn, 'horn')}.")
    elif ear:
        markers.append(f"Ears: {humanize(ear, 'ear')}.")

    distinguisher = (tv.get("key_distinguisher") or "").strip()
    if distinguisher:
        markers.append(distinguisher if distinguisher.endswith(".") else distinguisher + ".")
    elif dewlap or face or size:
        bits = [f"{humanize(x, f)} {label}"
                for x, f, label in ((dewlap, "dewlap", "dewlap"),
                                    (face, "face", "face"),
                                    (size, "size", "body")) if x]
        markers.append("Build: " + ", ".join(bits) + ".")

    # Guarantee three markers, using the curated note / general cues as filler.
    if len(markers) < 3:
        markers.append(f"Typical {species.lower()} frame of the {_display_name(slug)} breed.")
    if len(markers) < 3:
        note = (data or {}).get("note")
        markers.append(note if note else
                       "Compare against breed reference photos or your local KVK breed chart.")
    if len(markers) < 3:
        markers.append("Confirm age, sex and breed with a local veterinarian or extension officer.")
    return markers[:3]


def humanize(token, field=None):
    """Display string for a trait token; ``field`` disambiguates shared values."""
    from .trait_labels import humanize_trait
    if field:
        return humanize_trait(field, token)
    field = "hump"
    if token in ("black", "white", "grey", "fawn", "spotted", "red_brown",
                 "solid_brown", "red_and_white", "jet_black", "silver_grey",
                 "light_grey", "white_grey", "greyish_white", "greyish_black",
                 "black_grey", "black_brown_grey", "black_and_white",
                 "brown_white_patches", "fawn_and_white", "red_mottled"):
        field = "coat"
    elif token in ("small", "small_medium", "medium", "medium_large", "large"):
        field = "size"
    elif token in ("minimal", "moderate", "moderate_pendulous", "large_pendulous"):
        field = "dewlap"
    elif token in ("drooping", "long_pendulous", "medium_horizontal",
                   "medium_pendulous", "short_drooping", "small_erect",
                   "small_horizontal", "small_pointed"):
        field = "ear"
    elif token in ("absent", "short", "stumpy", "lyre_shaped", "semicircular",
                   "sickle_shaped", "tightly_curled", "heavy_drooping",
                   "long_curved", "long_flat", "long_backward_curved",
                   "curved_backward", "forward_curved", "flat_curved",
                   "upward_curved", "short_upward", "short_curved",
                   "short_curled", "short_thick", "short_stumpy", "short_backward",
                   "medium_curved", "medium_curved_upward", "medium_upward_curved"):
        field = "horn"
    elif token in ("broad_concave", "broad_convex", "broad_flat", "long_convex",
                   "long_narrow", "narrow_straight", "short_concave",
                   "short_straight", "straight_profile"):
        field = "face"
    return humanize_trait(field, token)


# ---------------------------------------------------------------------------
#  Feeding / climate / price / schemes
# ---------------------------------------------------------------------------

FEED_PLANS = {
    "small": {"green": (8, 12), "dry": (2, 3), "concentrate": (1.0, 1.5),
              "mineral_g": (40, 60), "water": (30, 45)},
    "medium": {"green": (15, 20), "dry": (4, 5), "concentrate": (1.5, 2.5),
               "mineral_g": (50, 75), "water": (40, 60)},
    "large": {"green": (22, 30), "dry": (5, 7), "concentrate": (2.0, 3.0),
              "mineral_g": (60, 90), "water": (55, 80)},
}


def _size_class(traits, species, slug):
    size = (traits.get((species, slug)) or {}).get("size")
    if size == "small":
        return "small"
    if size == "small_medium":
        return "small"
    if size in ("medium_large", "large"):
        return "large"
    if size == "medium":
        return "medium"
    return "large" if species == "Buffalo" else "medium"


def _feeding(size_class, species, utility, data_confidence):
    plan = dict(FEED_PLANS[size_class])
    if species == "Buffalo":
        plan["green"] = (plan["green"][0] + 3, plan["green"][1] + 4)
        plan["concentrate"] = (plan["concentrate"][0] + 0.5,
                               plan["concentrate"][1] + 0.5)
        plan["water"] = (plan["water"][0] + 10, plan["water"][1] + 15)

    note = ("Split concentrate into 2-3 feeds; add 0.4 kg concentrate per litre of "
            "milk above 5 L/day.")
    if utility == DRAUGHT:
        note = ("On working days add 0.5-1 kg concentrate and allow 2 rest days a week; "
                "feed roughage first, then concentrate.")
    elif utility == DWARF:
        note = ("Small body — avoid overfeeding; excess concentrate causes fatty liver "
                "and fertility problems in dwarf breeds.")

    return {
        "green_fodder_kg_per_day": f"{plan['green'][0]} - {plan['green'][1]}",
        "dry_fodder_kg_per_day": f"{plan['dry'][0]} - {plan['dry'][1]}",
        "concentrate_kg_per_day": f"{plan['concentrate'][0]} - {plan['concentrate'][1]}",
        "mineral_mixture_g_per_day": f"{plan['mineral_g'][0]} - {plan['mineral_g'][1]}",
        "water_liters_per_day": f"{plan['water'][0]} - {plan['water'][1]}",
        "notes": note,
        "data_source": data_confidence,
    }


def _climate(slug, species, origin, utility):
    text = (origin or "").lower()
    if any(k in text for k in ("ladakh", "himachal", "uttarakhand", "sikkim", "nagaland")):
        profile = ("High-altitude / cool-humid", "Moderate", "Moderate",
                   "Cold and monsoon protection; hooves soften in wet hill pastures.")
    elif any(k in text for k in ("kerala", "konkan", "assam", "coast", "chilika",
                                 "nilgiri", "odisha")):
        profile = ("Hot-humid coastal", "Moderate", "Moderate",
                   "High humidity raises foot-rot, tick and mastitis risk; provide dry bedding.")
    elif any(k in text for k in ("rajasthan", "kutch", "saurashtra", "thar",
                                 "gujarat", "bikaner", "jodhpur")):
        profile = ("Hot arid / semi-arid", "High", "High",
                   "Tolerates 40-45 °C and scarce water; shade and mineral salt are still essential.")
    elif any(k in text for k in ("punjab", "haryana", "uttar pradesh", "bihar",
                                 "madhya pradesh", "chhattisgarh")):
        profile = ("Hot summer, cool winter (irrigated plains)", "High", "Moderate",
                   "Watch heat stress in May-June and cold stress in December-January.")
    elif any(k in text for k in ("maharashtra", "karnataka", "andhra", "telangana",
                                 "tamil nadu")):
        profile = ("Hot semi-arid / tropical savanna", "High", "Moderate",
                   "Provide shade and water in summer; monitor ticks after monsoon.")
    else:
        profile = ("Tropical Indian plains", "Moderate", "Moderate",
                   "General tropical management; adjust shade and water to local climate.")

    heat, tick, parasites, note = profile
    note = CLIMATE_NOTES.get(slug, note)
    return {
        "heat_tolerance": heat,
        "tick_parasite_resistance": tick,
        "disease_notes": note,
        "suitable_for_smallholders": True,
        "housing": ("Shade + dry floor; 3-4 m² covered area per adult, more for buffalo."
                    if species == "Buffalo" else
                    "Shade + dry floor; 2.5-3.5 m² covered area per adult."),
    }


GOVT_SCHEMES = {
    "base": [
        "Rashtriya Gokul Mission (RGM) — indigenous breed conservation, AI and Gokul Gram",
        "National Livestock Mission (NLM) — fodder development and entrepreneurship support",
        "NADCP — free FMD and brucellosis vaccination",
        "Kisan Credit Card (KCC-AH) — working capital for fodder, feed and veterinary care",
    ],
    "milch": ["Dairy Infrastructure Development Fund (DIDF) — dairy processing and chilling"],
    "buffalo": ["Dairy Infrastructure Development Fund (DIDF) — dairy processing and chilling"],
    "rare": ["NBAGR / ICAR breed conservation unit — registration and elite germplasm support"],
    "dwarf": ["State livestock development board — small-holder and backyard dairy support"],
}


def _schemes(species, utility, data_confidence):
    schemes = list(GOVT_SCHEMES["base"])
    if species == "Buffalo" or utility == MILCH:
        schemes.insert(1, GOVT_SCHEMES["buffalo"][0])
    if utility == DWARF:
        schemes.append(GOVT_SCHEMES["dwarf"][0])
    if data_confidence == "regional_default":
        schemes.append(GOVT_SCHEMES["rare"][0])
    return schemes[:6]


# ---------------------------------------------------------------------------
#  Build
# ---------------------------------------------------------------------------

def _build_entry(slug, species, traits):
    table = BREED_DATA_CATTLE if species == "Cattle" else BREED_DATA_BUFFALO
    data = table.get(slug)
    confidence = "curated"
    note = ""
    if data:
        utility, daily, lactation, fat, price, a2, confidence, note = data
    else:
        utility = DUAL if species == "Cattle" else MILCH
        daily = (4, 7) if species == "Cattle" else (5, 8)
        lactation = (800, 1400) if species == "Cattle" else (1000, 1600)
        fat = (4.5, 5.5) if species == "Cattle" else (6.5, 8.0)
        price = (30000, 65000) if species == "Cattle" else (45000, 90000)
        a2 = True
        confidence = "regional_default"
        note = "Rare or newly documented class; figures are regional estimates."

    origin = ORIGIN_OVERRIDES.get((species, slug)) or ORIGIN_REGION.get(slug, "India")
    size_class = _size_class(traits, species, slug)
    display = _display_name(slug)

    local = {"en": display}
    local.update(LOCAL_NAMES.get(slug, {}))
    local.setdefault("hi", display)
    local.setdefault("ta", display)

    a2_text = ("A2/A2 typical (zebu/buffalo) — confirm by lab test"
               if a2 else "A1/A2 mixed (Bos taurus type) — confirm by lab test")

    return {
        "slug": slug,
        "display_name": display,
        "species": species,
        "local_names": local,
        "origin_region": origin,
        "utility_type": utility,
        "milk_yield": {
            "daily_liters": f"{daily[0]} - {daily[1]} L/day",
            "daily_liters_range": [daily[0], daily[1]],
            "lactation_yield": f"{lactation[0]} - {lactation[1]} kg",
            "lactation_kg_range": [lactation[0], lactation[1]],
            "lactation_length_days": "280 - 305 (cattle)" if species == "Cattle"
                                     else "300 - 320 (buffalo)",
            "fat_percentage": f"{fat[0]:.1f}% - {fat[1]:.1f}%",
            "fat_range": [fat[0], fat[1]],
            "beta_casein": a2_text,
        },
        "physical_markers": _markers(slug, species, traits, {"note": note}),
        "feeding_and_care": _feeding(size_class, species, utility, confidence),
        "climate_and_disease_resistance": _climate(slug, species, origin, utility),
        "estimated_market_price_inr": f"₹{price[0]:,} - ₹{price[1]:,}",
        "price_range_inr": [price[0], price[1]],
        "govt_schemes": _schemes(species, utility, confidence),
        "data_confidence": confidence,
        "note": note,
        "size_class": size_class,
    }


def build_knowledge(traits=None):
    """Generate the full knowledge dict for every configured breed."""
    traits = _load_traits() if traits is None else traits
    breeds = {"Cattle": {}, "Buffalo": {}}
    for species, slugs in (
        ("Cattle", CATTLE_BREEDS + EXTRA_CATTLE_BREEDS),
        ("Buffalo", BUFFALO_BREEDS),
    ):
        for slug in slugs:
            breeds[species][slug] = _build_entry(slug, species, traits)
    return {
        "_meta": {
            "schema_version": SCHEMA_VERSION,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source": "src/breed_knowledge.py",
            "breed_count": len(CATTLE_BREEDS) + len(BUFFALO_BREEDS),
            "extra_crossbred_classes": list(EXTRA_CATTLE_BREEDS),
            "data_policy": ("Curated entries follow NBAGR/ICAR breed descriptions and Indian "
                            "feeding norms; 'regional_default' entries are regional estimates. "
                            "Prices, yields and vaccination timings vary — confirm locally."),
        },
        "Cattle": breeds["Cattle"],
        "Buffalo": breeds["Buffalo"],
    }


# ---------------------------------------------------------------------------
#  Persistence
# ---------------------------------------------------------------------------

_CACHE = None


def ensure_knowledge_file(path=KNOWLEDGE_FILE, force=False):
    """Create (or complete) data/breed_knowledge.json. Returns the loaded dict.

    Existing files are preserved: only missing species/breed keys are added, so
    hand-edited numbers survive. ``force=True`` rewrites everything.
    """
    generated = build_knowledge()
    if force or not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(generated, f, ensure_ascii=False, indent=2)
        return generated

    try:
        with open(path, encoding="utf-8") as f:
            current = json.load(f)
    except Exception:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(generated, f, ensure_ascii=False, indent=2)
        return generated

    changed = False
    current.setdefault("_meta", generated["_meta"])
    for species in ("Cattle", "Buffalo"):
        section = current.setdefault(species, {})
        for slug, entry in generated[species].items():
            if slug not in section:
                section[slug] = entry
                changed = True
    if changed:
        current["_meta"]["breed_count"] = sum(
            len(current.get(s, {})) for s in ("Cattle", "Buffalo"))
        with open(path, "w", encoding="utf-8") as f:
            json.dump(current, f, ensure_ascii=False, indent=2)
    return current


def load_knowledge(path=KNOWLEDGE_FILE, force_reload=False):
    """Load (and lazily create) the knowledge base. Cached per process."""
    global _CACHE
    if _CACHE is not None and not force_reload:
        return _CACHE
    _CACHE = ensure_knowledge_file(path)
    return _CACHE


def reload_knowledge(path=KNOWLEDGE_FILE):
    """Drop the cache and re-read the JSON (after manual edits)."""
    return load_knowledge(path, force_reload=True)


def list_breeds():
    """Return [(species, slug, display_name)] for every configured breed."""
    kb = load_knowledge()
    out = []
    for species in ("Cattle", "Buffalo"):
        for slug, entry in kb.get(species, {}).items():
            out.append((species, slug, entry["display_name"]))
    return out


def get_breed_knowledge(species, breed):
    """Look up one breed; tolerant of aliases, case, spaces and hyphens."""
    kb = load_knowledge()
    species_key = _normalize_species(species)
    slug = _normalize_slug(breed)
    section = kb.get(species_key, {})
    if slug in section:
        return section[slug]
    # A slug that exists only under the other species (e.g. bargur) is a caller
    # mistake -- return None so the API can 404 instead of serving wrong data.
    return None


# ---------------------------------------------------------------------------
#  Advisory engine
# ---------------------------------------------------------------------------

DAIRY_PRICE_PER_LITRE = {"Cattle": (32, 42), "Buffalo": (45, 58)}

DISCLAIMER = (
    "General guidance for informed decisions, not a veterinary prescription. "
    "Yields, prices and disease risk vary by farm, feeding and local market — "
    "confirm with your veterinarian or State Animal Husbandry Department."
)

VACCINATION_BASE = [
    "FMD — first dose at 4-6 months, then every 6 months (NADCP)",
    "HS (Haemorrhagic Septicaemia) — annual, before the monsoon",
    "BQ (Black Quarter) — annual, before the monsoon",
    "Brucellosis — single dose for female calves at 4-8 months (Brucella abortus S19/RB51)",
]


def _norm_gender(gender):
    text = str(gender or "").strip().lower()
    if "female" in text or text in ("f", "cow", "buffalo cow"):
        return "Female"
    if "male" in text or text in ("m", "bull", "ox", "buffalo bull"):
        return "Male"
    return None


def _norm_age(age_group):
    text = str(age_group or "").strip().lower()
    if "calf" in text or "<1" in text or "< 1" in text:
        return "Calf"
    if "young" in text or "sub" in text or "heifer" in text:
        return "Young"
    if "adult" in text or "mature" in text:
        return "Adult"
    return None


def _fmt_range(pair, unit=""):
    return f"{pair[0]:g} - {pair[1]:g}{unit}"


def _milk_block(entry, species):
    milk = entry["milk_yield"]
    lo, hi = milk["daily_liters_range"]
    price_lo, price_hi = DAIRY_PRICE_PER_LITRE[species]
    monthly_lo = int(lo * 30 * price_lo)
    monthly_hi = int(hi * 30 * price_hi)
    return {
        "daily_yield": milk["daily_liters"],
        "lactation_yield": milk["lactation_yield"],
        "fat": milk["fat_percentage"],
        "lactation_length": milk["lactation_length_days"],
        "beta_casein": milk["beta_casein"],
        "estimated_monthly_income_inr": f"₹{monthly_lo:,} - ₹{monthly_hi:,} (gross, farm-gate)",
        "income_basis": f"{lo}-{hi} L/day × 30 days × ₹{price_lo}-{price_hi}/L",
    }


def _afc(entry, species):
    slug = entry["slug"]
    if species == "Buffalo":
        return "30 - 42 months (target 36 months for Murrah-type breeds)"
    if slug in ("ayrshire", "holstein_friesian", "jersey"):
        return "24 - 27 months (crossbred/temperate type)"
    if entry["utility_type"] == DWARF:
        return "30 - 36 months (small body; do not breed before 200 kg)"
    return "28 - 36 months (target 30-32 months for indigenous zebu)"


def _base_advisory(entry, species, gender, age_group):
    feed = entry["feeding_and_care"]
    climate = entry["climate_and_disease_resistance"]
    return {
        "breed": {
            "species": species,
            "slug": entry["slug"],
            "display_name": entry["display_name"],
            "local_names": entry["local_names"],
            "origin_region": entry["origin_region"],
            "utility_type": entry["utility_type"],
            "data_confidence": entry["data_confidence"],
        },
        "context": {"gender": gender, "age_group": age_group},
        "physical_markers": entry["physical_markers"],
        "nutrition": {
            "green_fodder": f"{feed['green_fodder_kg_per_day']} kg/day",
            "dry_fodder": f"{feed['dry_fodder_kg_per_day']} kg/day",
            "concentrate": f"{feed['concentrate_kg_per_day']} kg/day",
            "mineral_mixture": f"{feed['mineral_mixture_g_per_day']} g/day",
            "water": f"{feed['water_liters_per_day']} L/day",
            "notes": feed["notes"],
        },
        "health": {
            "vaccination": list(VACCINATION_BASE),
            "vector_control": climate["disease_notes"],
            "housing": climate["housing"],
        },
        "economics": {
            "market_price": entry["estimated_market_price_inr"],
            "price_range_inr": entry["price_range_inr"],
            "schemes": entry["govt_schemes"],
        },
        "climate": {
            "heat_tolerance": climate["heat_tolerance"],
            "tick_parasite_resistance": climate["tick_parasite_resistance"],
            "suitable_for_smallholders": climate["suitable_for_smallholders"],
        },
        "warning": (f"{entry['note']} " if entry.get("note") else ""),
        "disclaimer": DISCLAIMER,
        "method": "knowledge_base",
    }


def _female_adult(entry, species, adv):
    milk = entry["milk_yield"]
    block = _milk_block(entry, species)
    lo, hi = milk["daily_liters_range"]
    extra_lo = max(0.0, (lo - 5) * 0.4)
    extra_hi = max(0.0, (hi - 5) * 0.4)
    adv["headline"] = (
        f"{entry['display_name']} adult female — protect lactation: "
        f"{milk['daily_liters']} and {milk['fat_percentage']} fat are achievable "
        "with steady feeding and clean milking."
    )
    adv["priority_actions"] = [
        "Milk 2-3 times a day at fixed times; feed concentrate after milking, not before.",
        (f"Top up concentrate by 0.4 kg per litre above 5 L/day "
         f"(≈{extra_lo:.1f}-{extra_hi:.1f} kg extra at this breed's {lo}-{hi} L yield)."),
        "Offer 50-75 g mineral mixture + 30-50 g common salt daily; free-choice calcium during peak lactation.",
        "Prevent mastitis: wash and dry udder, post-milking teat dip, clean bedding, full milking.",
        "Dry off 60 days before calving; do not milk the dry period.",
    ]
    adv["production"] = {"milk": block, "draught": None}
    adv["health"]["watch_for"] = [
        "Mastitis — check quarters daily, use CMT every month",
        "Milk fever / hypocalcaemia around calving",
        "Ketosis in early lactation if appetite drops",
        "Repeat breeding beyond 3 services — get a vet examination",
    ]
    adv["breeding"] = {
        "heat_cycle": "18 - 21 days (cattle)" if species == "Cattle" else "21 - 24 days (buffalo)",
        "service_window": "12 - 18 hours after standing heat (cattle); 12 - 24 h (buffalo)",
        "calving_interval_target": "12 - 14 months",
        "advice": ("Inseminate only on a confirmed standing heat; keep a calving/heat register. "
                   "Buffalo heats are silent — watch for mucus and restlessness."),
    }
    adv["warnings"] = [
        "Do not feed high-concentrate diets without roughage — acidosis lowers milk and fertility.",
        "Discard milk according to the withdrawal period after any antibiotic treatment.",
    ]
    adv["summary"] = ("Adult female: lactation management is the priority. Track daily yield, "
                      "body condition and udder health; re-check pregnancy by 45-60 days.")
    return adv


def _female_young(entry, species, adv):
    adv["headline"] = (
        f"{entry['display_name']} female {('heifer' if species == 'Cattle' else 'buffalo heifer')} "
        "— grow her to breeding weight before the first AI."
    )
    adv["priority_actions"] = [
        "Feed a growth ration: good green fodder + 1-1.5 kg concentrate + mineral mixture daily.",
        f"Target first calving at {_afc(entry, species)}.",
        "Weigh or use a heart-girth tape monthly; heifers should gain 400-600 g/day.",
        "Watch for first heat signs (restlessness, mucus, mounting) and record the date.",
        "Deworm quarterly and vaccinate FMD/HS/BQ before breeding.",
    ]
    adv["production"] = {"milk": None, "draught": None}
    adv["health"]["watch_for"] = [
        "Stunted growth from underfeeding — the most common cause of late first calving",
        "External parasites and anaemia",
        "Brucellosis vaccination status (female calves 4-8 months)",
    ]
    adv["breeding"] = {
        "age_at_first_calving": _afc(entry, species),
        "breeding_weight": ("250 - 300 kg for indigenous cattle; 300 - 350 kg for crossbreds"
                            if species == "Cattle" else
                            "300 - 380 kg for buffalo heifers"),
        "advice": ("Do not rush breeding — a heifer that calves below target weight stays "
                   "unproductive for life."),
    }
    adv["warnings"] = [
        "Avoid high-energy 'fattening' rations; they lay down fat in the udder and reduce future yield.",
    ]
    adv["summary"] = ("Young female: growth and age-at-first-calving decide lifetime productivity. "
                      "Aim for steady gain, not fattening.")
    return adv


def _male_adult(entry, species, adv):
    adv["headline"] = (
        f"{entry['display_name']} adult male — value comes from draught power, "
        "breeding/semen quality and sound feet, not milk."
    )
    adv["priority_actions"] = [
        "Assess work stamina gradually: start with light loads, build over 2-3 weeks.",
        "Feed 0.5-1 kg extra concentrate on working days; give 2 rest days a week.",
        "Check hooves monthly; trim overgrown horn/hoof and treat interdigital wounds early.",
        "Use a properly padded yoke/harness; check neck and shoulder for sores after work.",
        "If kept as a breeding bull, record services and get a semen quality test annually.",
    ]
    adv["production"] = {
        "milk": None,
        "draught": {
            "work_capacity": ("4 - 6 hours/day on light soil; 2 - 3 hours on heavy clay"
                              if entry["utility_type"] == DRAUGHT else
                              "3 - 4 hours/day of light work; this breed is primarily dairy"),
            "load": "Approx. 1/4 to 1/3 of body weight on a cart, less on waterlogged soil",
            "note": ("Milk yield is deliberately hidden for males — it is not the relevant "
                     "metric. Breeding-bull/semen value depends on pedigree and progeny testing."),
        },
    }
    adv["health"]["watch_for"] = [
        "Yoke/harness sores and shoulder abscesses",
        "Hoof overgrowth and interdigital dermatitis in wet yards",
        "Aggression during rut — handle with a nose rope and secure enclosure",
    ]
    adv["breeding"] = {
        "bull_management": ("Use a registered breeding bull; one bull per 25-30 cows in natural "
                            "service, or collect semen at an AI centre."),
        "semen_value": "₹100 - ₹500 per AI dose for a proven indigenous bull (varies by facility)",
    }
    adv["warnings"] = [
        "Do not work a bull to exhaustion — heat stress and dehydration lower semen quality for weeks.",
    ]
    adv["summary"] = ("Adult male: manage work load, feet and body condition. "
                      "Semen/breeding value is the main economic output for good bulls.")
    return adv


def _male_young(entry, species, adv):
    adv["headline"] = (
        f"{entry['display_name']} young male — decide early between breeding bull, "
        "draught ox and sale."
    )
    adv["priority_actions"] = [
        "Feed for steady growth (0.5-0.7 kg/day gain); do not fatten before maturity.",
        "Decide by 12-18 months: retain as breeding bull only if pedigree and conformation are good.",
        "If castrating for draught, use a qualified vet and follow aftercare/antibiotics.",
        "Handle daily from a young age — a tractable bullock is safer and worth more.",
        "Vaccinate FMD/HS/BQ and deworm quarterly.",
    ]
    adv["production"] = {"milk": None, "draught": None}
    adv["health"]["watch_for"] = [
        "Wound infection after castration",
        "Naval infection in young calves",
        "Poor growth from inadequate concentrate",
    ]
    adv["breeding"] = {
        "selection": ("Keep a bull calf only if the dam yields well and the calf has a straight "
                      "back, sound feet and breed-typical horns."),
        "castration_age": "6 - 12 months (veterinary procedure)",
    }
    adv["warnings"] = [
        "Uncastrated males left with the herd can cause injuries and unwanted pregnancies.",
    ]
    adv["summary"] = ("Young male: nutrition plus an early breeding/draught decision. "
                      "Conformation and temperament set the sale value.")
    return adv


def _calf(entry, species, adv, gender):
    feed = entry["feeding_and_care"]
    adv["headline"] = (
        f"{entry['display_name']} calf — the first 90 days decide growth, "
        "immunity and lifetime yield."
    )
    adv["priority_actions"] = [
        "Colostrum: 1-2 L within the first 1-2 hours, then 10% of body weight/day for 3 days.",
        "Deworm on day 10, day 30 and day 60, then every 3 months.",
        "Start calf starter (18-20% protein) from week 2; wean at 8-12 weeks when eating 1 kg/day.",
        "Vaccinate: FMD at 4-6 months, HS and BQ at 6 months (then annual); brucellosis for female calves at 4-8 months.",
        "Provide clean water, dry bedding and shade from day one.",
    ]
    adv["production"] = {"milk": None, "draught": None}
    adv["nutrition"] = {
        "colostrum": "1 - 2 L within 1 - 2 hours of birth (critical)",
        "milk": "10% of body weight/day, split into 2-3 feeds, for the first 8-12 weeks",
        "calf_starter": "0.5 - 1.5 kg/day from week 2 (18-20% crude protein)",
        "green_fodder": "Introduce soft green fodder from week 3-4",
        "water": "Free-choice clean water from day 1",
        "notes": "Never feed adult concentrate or stale milk; wash buckets daily.",
    }
    adv["health"]["watch_for"] = [
        "Scours (diarrhoea) — isolate and give oral rehydration immediately",
        "Pneumonia — draft-free housing, check breathing after feeding",
        "Naval infection in the first week",
        "Coccidiosis in wet, dirty pens",
    ]
    adv["health"]["deworming_schedule"] = ["Day 10", "Day 30", "Day 60", "Then every 3 months"]
    adv["health"]["vaccination_calendar"] = {
        "FMD": "4-6 months, then every 6 months",
        "HS": "6 months, then annual before monsoon",
        "BQ": "6 months, then annual before monsoon",
        "Brucellosis": "Female calves only, 4-8 months, single dose",
    }
    adv["breeding"] = {"note": "No breeding decisions at this stage."}
    adv["warnings"] = [
        "Colostrum cannot be replaced by milk or milk replacer — without it the calf has almost no immunity.",
        "Do not wean before the calf reliably eats 1 kg of starter per day.",
    ]
    adv["summary"] = (
        f"Calf care: colostrum, clean water, starter ration, deworming at day 10/30/60 and the "
        f"FMD/HS/BQ calendar. {'Female calves also need brucellosis vaccination at 4-8 months.' if gender == 'Female' else ''}"
    ).strip()
    return adv


def _general(entry, species, adv, gender):
    adv["headline"] = (
        f"{entry['display_name']} ({entry['utility_type']}) — {entry['origin_region']}."
    )
    adv["priority_actions"] = [
        f"Typical yield: {entry['milk_yield']['daily_liters']} at {entry['milk_yield']['fat_percentage']} fat.",
        f"Feed {entry['feeding_and_care']['green_fodder_kg_per_day']} kg green + "
        f"{entry['feeding_and_care']['dry_fodder_kg_per_day']} kg dry fodder daily.",
        "Vaccinate for FMD, HS and BQ; deworm every 3 months.",
        "Check the physical markers below to confirm the AI prediction.",
    ]
    adv["production"] = {
        "milk": entry["milk_yield"] if gender != "Male" else None,
        "draught": None,
    }
    adv["breeding"] = {"age_at_first_calving": _afc(entry, species)}
    adv["warnings"] = ["Set gender and age group to get stage-specific advice."]
    adv["summary"] = ("Select the animal's gender and age group for lactation, heifer, bull or "
                      "calf-specific guidance.")
    return adv


def get_farmer_advisory(species, breed_slug, gender=None, age_group=None):
    """Context-aware advisory for one breed.

    ``gender``    : "Female" / "Male" (or the API's display labels).
    ``age_group`` : "Calf" / "Young" / "Adult" (or the API's display labels).

    Returns a dict with ``headline``, ``priority_actions``, ``nutrition``,
    ``health``, ``production``, ``breeding``, ``economics``, ``schemes``,
    ``warnings`` and ``disclaimer``. Pure dict assembly — no I/O, no model.
    """
    entry = get_breed_knowledge(species, breed_slug)
    if entry is None:
        return {"error": f"Unknown breed '{breed_slug}' for species '{species}'"}

    species_key = _normalize_species(species)
    g = _norm_gender(gender)
    a = _norm_age(age_group)

    adv = _base_advisory(entry, species_key, g, a)

    if a == "Calf":
        adv = _calf(entry, species_key, adv, g)
    elif g == "Female" and a == "Adult":
        adv = _female_adult(entry, species_key, adv)
    elif g == "Female" and a == "Young":
        adv = _female_young(entry, species_key, adv)
    elif g == "Male" and a == "Adult":
        adv = _male_adult(entry, species_key, adv)
    elif g == "Male" and a == "Young":
        adv = _male_young(entry, species_key, adv)
    else:
        adv = _general(entry, species_key, adv, g)

    adv["schemes"] = entry["govt_schemes"]
    adv["context"] = {"gender": g, "age_group": a}
    return adv


def knowledge_stats():
    kb = load_knowledge()
    curated = sum(1 for sp in ("Cattle", "Buffalo")
                  for e in kb[sp].values() if e["data_confidence"] == "curated")
    total = sum(len(kb[sp]) for sp in ("Cattle", "Buffalo"))
    return {"total": total, "curated": curated,
            "regional_default": total - curated, "file": KNOWLEDGE_FILE}


def _main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
    parser = argparse.ArgumentParser(description="Breed knowledge base utility")
    parser.add_argument("--regenerate", action="store_true",
                        help="rewrite data/breed_knowledge.json from this module")
    parser.add_argument("--stats", action="store_true", help="print coverage counts")
    parser.add_argument("--show", metavar="BREED", help="print one breed entry")
    parser.add_argument("--species", default="Cattle", help="species for --show")
    parser.add_argument("--advisory", nargs=2, metavar=("GENDER", "AGE"),
                        help="print an advisory, e.g. --advisory Female Adult")
    args = parser.parse_args(argv)

    if args.regenerate:
        data = ensure_knowledge_file(force=True)
        print(f"Wrote {KNOWLEDGE_FILE} ({data['_meta']['breed_count']} breeds)")
        return 0
    if args.stats:
        print(json.dumps(knowledge_stats(), indent=2))
        return 0
    if args.show:
        entry = get_breed_knowledge(args.species, args.show)
        if entry is None:
            print(f"Unknown breed: {args.show}", file=sys.stderr)
            return 1
        if args.advisory:
            print(json.dumps(get_farmer_advisory(args.species, args.show,
                                                 *args.advisory),
                             ensure_ascii=False, indent=2))
        else:
            print(json.dumps(entry, ensure_ascii=False, indent=2))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(_main())
