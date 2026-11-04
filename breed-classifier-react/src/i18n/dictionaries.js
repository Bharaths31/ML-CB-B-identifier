/**
 * Dynamic localization dictionaries.
 *
 * The backend returns English payload values (species, gender, age, traits,
 * breed names, advisory prose). These maps let the React tree render the same
 * payload in Tamil/Hindi without another API call.
 *
 * Breed keys are the canonical class-map slugs; lookup also accepts display
 * names ("Red Sindhi"), aliases ("bhadawari") and case/space variants.
 */

const B = (en, ta, hi) => ({ en, ta, hi });

export const BREED_NAMES_I18N = {
  // ---------------------------------------------------------------- cattle
  amritmahal: B("Amrit Mahal", "அமிர்தமகால்", "अमृतमहल"),
  ayrshire: B("Ayrshire", "ஐர்ஷயர்", "आयरशायर"),
  bachaur: B("Bachaur", "பசௌர்", "बछौर"),
  badri: B("Badri", "பத்ரி", "बद्री"),
  bargur: B("Bargur", "பர்கூர்", "बरगुर"),
  bhelai: B("Bhelai", "பேலாய்", "भेलाई"),
  dagri: B("Dagri", "டாக்ரி", "डागरी"),
  dangi: B("Dangi", "டாங்கி", "डांगी"),
  deoni: B("Deoni", "தேவனி", "देवनी"),
  gangatari: B("Gangatari", "கங்காதரி", "गंगातरी"),
  gaolao: B("Gaolao", "காவ்லாவ்", "गाओलाओ"),
  ghumsari: B("Ghumsari", "கும்சாரி", "घुमसारी"),
  gir: B("Gir", "கிர்", "गिर"),
  hallikar: B("Hallikar", "ஹல்லிகர்", "हल्लीकर"),
  hariana: B("Hariana", "ஹரியானா", "हरियाणा"),
  himachali_pahari: B("Himachali Pahari", "இமாச்சலி பஹாரி", "हिमाचली पहाड़ी"),
  kangayam: B("Kangayam", "காங்கேயம்", "कांगायम"),
  kankrej: B("Kankrej", "காங்க்ரேஜ்", "कांकरेज"),
  kenkatha: B("Kenkatha", "கேன்கதா", "केनकथा"),
  khariar: B("Khariar", "காரியார்", "खरियार"),
  kherigarh: B("Kherigarh", "கேரிகர்", "खेरीगढ़"),
  khillari: B("Khillari", "கில்லாரி", "खिल्लारी"),
  konkan_kapila: B("Konkan Kapila", "கொங்கன் கபிலா", "कोंकण कपिला"),
  kosali: B("Kosali", "கோசலி", "कोसली"),
  krishna_valley: B("Krishna Valley", "கிருஷ்ணா வேலி", "कृष्णा वैली"),
  ladakhi: B("Ladakhi", "லடாக்கி", "लद्दाखी"),
  lakhimi: B("Lakhimi", "லக்ஷ்மி", "लखिमी"),
  malnad_gidda: B("Malnad Gidda", "மல்னாடு கிட்டா", "मालनाड गिड्डा"),
  malvi: B("Malvi", "மால்வி", "मालवी"),
  mewati: B("Mewati", "மேவாதி", "मेवाती"),
  motu: B("Motu", "மோட்டு", "मोटू"),
  nagori: B("Nagori", "நாகோரி", "नागोरी"),
  nari: B("Nari", "நாரி", "नारी"),
  nimari: B("Nimari", "நிமாரி", "निमाड़ी"),
  ongole: B("Ongole", "ஓங்கோல்", "ओंगोल"),
  poda_thirupu: B("Poda Thirupu", "பொடா திருப்பு", "पोडा तिरुपु"),
  ponwar: B("Ponwar", "பொன்வார்", "पोनवार"),
  pulikulam: B("Pulikulam", "புலிகுளம்", "पुलिकुलम"),
  punganur: B("Punganur", "புங்கனூர்", "पुंगनूर"),
  purnea: B("Purnea", "பூர்ணியா", "पूर्णिया"),
  rathi: B("Rathi", "ராத்தி", "राठी"),
  red_kandhari: B("Red Kandhari", "சிவப்பு கந்தாரி", "लाल कंधारी"),
  red_sindhi: B("Red Sindhi", "சிவப்பு சிந்தி", "लाल सिंधी"),
  sahiwal: B("Sahiwal", "சாஹிவால்", "साहीवाल"),
  shweta_kapila: B("Shweta Kapila", "ஸ்வேதா கபிலா", "श्वेता कपिला"),
  siri: B("Siri", "சிரி", "सिरी"),
  tharparkar: B("Tharparkar", "தார்பார்கர்", "थारपारकर"),
  thutho: B("Thutho", "துதோ", "थुथो"),
  umblachery: B("Umblachery", "உம்பளச்சேரி", "उम्बलाचेरी"),
  vechur: B("Vechur", "வெச்சூர்", "वेचुर"),
  // --------------------------------------------------------------- buffalo
  banni: B("Banni", "பன்னி", "बन्नी"),
  bhadwari: B("Bhadwari", "பதாவரி", "भदावरी"),
  chhattisgarhi: B("Chhattisgarhi", "சத்தீஸ்கரி", "छत्तीसगढ़ी"),
  chilika: B("Chilika", "சிலிகா", "चिलिका"),
  gojri: B("Gojri", "கோஜ்ரி", "गोजरी"),
  jaffarabadi: B("Jaffarabadi", "ஜாஃபராபாதி", "जाफराबादी"),
  kalahandi: B("Kalahandi", "காளாஹண்டி", "कालाहांडी"),
  luit: B("Luit", "லுயித்", "लुइत"),
  marathwada: B("Marathwada", "மராத்வாடா", "मराठवाड़ा"),
  mehsana: B("Mehsana", "மெஹ்சானா", "मेहसाणा"),
  murrah: B("Murrah", "முர்ரா", "मुर्रा"),
  nagpuri: B("Nagpuri", "நாக்புரி", "नागपुरी"),
  nili_ravi: B("Nili-Ravi", "நீலி ராவி", "नीली रावी"),
  other_buffalo: B("Other / Mixed Buffalo", "கலப்பு எருமை", "मिश्रित भैंस"),
  pandharpuri: B("Pandharpuri", "பந்தர்புரி", "पंढरपुरी"),
  surti: B("Surti", "சூர்தி", "सुरती"),
  toda: B("Toda", "தோடா", "टोडा"),
  // ------------------------------------------------------------- crossbred
  holstein_friesian: B("Holstein Friesian", "ஹோல்ஸ்டீன் பிரீஸியன்", "होल्स्टीन फ्रीजियन"),
  jersey: B("Jersey", "ஜெர்சி", "जर्सी"),
};

const BREED_ALIASES = {
  belahi: "bhelai", gangatiri: "gangatari", ghumusari: "ghumsari",
  khillar: "khillari", poda_thurpu: "poda_thirupu", bhadawari: "bhadwari",
  marathwadi: "marathwada", hf_cross: "holstein_friesian",
  holstein: "holstein_friesian", jersey_cross: "jersey",
  alambadi: "other_buffalo", mixed_buffalo: "other_buffalo",
};

export function normalizeBreedKey(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[()-]/g, " ")
    .replace(/\s+/g, "_")
    .replace(/^_+|_+$/g, "");
}

export function localizeBreed(value, lang = "en") {
  const key = normalizeBreedKey(value);
  const canonical = BREED_ALIASES[key] || key;
  const entry = BREED_NAMES_I18N[canonical];
  if (entry) return entry[lang] || entry.en;
  return String(value || "").replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

/* ------------------------------------------------------------------ traits */

export const TRAIT_FIELDS_I18N = {
  hump: B("Hump", "திமில்", "कूबड़"),
  horn: B("Horns", "கொம்பு", "सींग"),
  coat: B("Coat", "உடல் நிறம்", "रंग"),
  ear: B("Ears", "காது", "कान"),
  dewlap: B("Dewlap", "தாடைத் தோல் (அலைதாடி)", "गलकंबल"),
  face: B("Face profile", "முக வடிவம்", "चेहरा"),
  size: B("Body size", "உடல் அமைப்பு", "शरीर का आकार"),
  origin: B("Origin", "தாயகம் / பூர்வீகம்", "उत्पत्ति स्थल"),
};

export const TRAIT_VALUES_I18N = {
  "Absent / Flat": B("Absent / Flat", "இல்லை / தட்டையான", "अनुपस्थित / सपाट"),
  Small: B("Small", "சிறிய", "छोटा"),
  Medium: B("Medium", "நடுத்தர", "मध्यम"),
  "Large / Prominent": B("Large / Prominent", "பெரிய / தெளிவான", "बड़ा / प्रमुख"),
  "Well-Developed (Cervicothoracic)": B("Well-Developed", "நன்கு வளர்ந்த", "सुविकसित"),
  "Large (Cervicothoracic)": B("Large", "பெரிய", "बड़ा"),
  "Absent / Polled": B("Absent / Polled", "இல்லை / கொம்பற்ற", "अनुपस्थित / सींग रहित"),
  Short: B("Short", "குறுகிய", "छोटा"),
  "Short, Upward": B("Short, Upward", "குறுகிய, மேல்நோக்கி", "छोटा, ऊपर"),
  "Short, Curved": B("Short, Curved", "குறுகிய, வளைந்த", "छोटा, मुड़ा"),
  "Short, Curled": B("Short, Curled", "குறுகிய, சுருண்ட", "छोटा, मुड़ा"),
  "Short & Thick": B("Short & Thick", "குறுகிய, தடிமனான", "छोटा और मोटा"),
  "Short & Stumpy": B("Short & Stumpy", "குறுகிய, குட்டையான", "छोटा और ठूंठा"),
  "Short, Backward": B("Short, Backward", "குறுகிய, பின்னோக்கி", "छोटा, पीछे"),
  Stumpy: B("Stumpy", "குட்டையான", "ठूंठा"),
  "Medium, Curved": B("Medium, Curved", "நடுத்தர, வளைந்த", "मध्यम, मुड़ा"),
  "Medium, Curved Upward": B("Medium, Curved Upward", "நடுத்தர, மேல்நோக்கி வளைந்த", "मध्यम, ऊपर मुड़ा"),
  "Medium, Upward Curved": B("Medium, Upward Curved", "நடுத்தர, மேல் வளைவு", "मध्यम, ऊपर मुड़ा"),
  "Long, Curved": B("Long, Curved", "நீண்ட, வளைந்த", "लंबा, मुड़ा"),
  "Long & Flat": B("Long & Flat", "நீண்ட, தட்டையான", "लंबा और सपाट"),
  "Long, Backward Curved": B("Long, Backward Curved", "நீண்ட, பின்னோக்கி வளைந்த", "लंबा, पीछे मुड़ा"),
  "Curved Backward": B("Curved Backward", "பின்னோக்கி வளைந்த", "पीछे मुड़ा"),
  "Forward Curved": B("Forward Curved", "முன்னோக்கி வளைந்த", "आगे मुड़ा"),
  "Flat & Curved": B("Flat & Curved", "தட்டையான, வளைந்த", "सपाट और मुड़ा"),
  "Upward Curved": B("Upward Curved", "மேல்நோக்கி வளைந்த", "ऊपर मुड़ा"),
  "Heavy & Drooping": B("Heavy & Drooping", "கனமான, தொங்கும்", "भारी और लटकता"),
  "Lyre-Shaped": B("Lyre-Shaped", "யாழ் வடிவ", "वीणा आकार"),
  "Semicircular (Half-moon)": B("Semicircular", "அரை வட்டம் (பிறை)", "अर्धवृत्त (अर्धचंद्र)"),
  "Sickle-Shaped": B("Sickle-Shaped", "அரிவாள் வடிவ", "दरांती आकार"),
  "Tightly Curled": B("Tightly Curled", "இறுக்கமாக சுருண்ட", "कुंडलाकार / कसकर मुड़े सींग"),
  Black: B("Black", "கருப்பு", "काला"),
  "Jet Black": B("Jet Black", "கருங்கருப்பு", "गहरा काला"),
  White: B("White", "வெள்ளை", "सफ़ेद"),
  Grey: B("Grey", "சாம்பல்", "धूसर"),
  "Light Grey": B("Light Grey", "வெளிர் சாம்பல்", "हल्का धूसर"),
  "Silver Grey": B("Silver Grey", "வெள்ளி சாம்பல்", "रजत धूसर"),
  "White-Grey": B("White-Grey", "வெள்ளை-சாம்பல்", "सफ़ेद-धूसर"),
  "Greyish White": B("Greyish White", "சாம்பல் வெள்ளை", "धूसर सफ़ेद"),
  "Greyish Black": B("Greyish Black", "சாம்பல் கருப்பு", "धूसर काला"),
  "Black-Grey": B("Black-Grey", "கருப்பு-சாம்பல்", "काला-धूसर"),
  "Black / Brown / Grey": B("Black / Brown / Grey", "கருப்பு / பழுப்பு / சாம்பல்", "काला / भूरा / धूसर"),
  "Black & White": B("Black & White", "கருப்பு-வெள்ளை", "काला और सफ़ेद"),
  "Brown with White Patches": B("Brown with White Patches", "பழுப்பு-வெள்ளைத் திட்டுகள்", "भूरा सफ़ेद धब्बों के साथ"),
  Fawn: B("Fawn", "மான் நிறம்", "बादामी / हल्का पीला"),
  "Fawn & White": B("Fawn & White", "மான் நிறம்-வெள்ளை", "बादामी और सफ़ेद"),
  "Red & White": B("Red & White", "சிவப்பு-வெள்ளை", "लाल और सफ़ेद"),
  "Red-Brown": B("Red-Brown", "செம்பழுப்பு", "लाल-भूरा"),
  "Red Mottled": B("Red Mottled", "சிவப்பு புள்ளிகள்", "लाल चितकबरा"),
  "Solid Brown": B("Solid Brown", "ஒரே சீரான பழுப்பு", "एकसमान भूरा रंग"),
  Spotted: B("Spotted", "புள்ளிகள்", "धब्बेदार"),
  "Long & Pendulous": B("Long & Pendulous", "நீளமான தொங்கு", "लंबे और लटकते"),
  "Medium, Horizontal": B("Medium, Horizontal", "நடுத்தர, கிடைமட்ட", "मध्यम, क्षैतिज"),
  "Medium, Pendulous": B("Medium, Pendulous", "நடுத்தர, தொங்கு", "मध्यम, लटकते"),
  Drooping: B("Drooping", "தொங்கும்", "लटकते"),
  "Short, Drooping": B("Short, Drooping", "குறுகிய, தொங்கு", "छोटे, लटकते"),
  "Small & Erect": B("Small & Erect", "சிறிய, நேராக", "छोटे और खड़े"),
  "Small, Horizontal": B("Small, Horizontal", "சிறிய, கிடைமட்ட", "छोटे, क्षैतिज"),
  "Small & Pointed": B("Small & Pointed", "சிறிய, கூரான", "छोटे और नुकीले"),
  Minimal: B("Minimal", "மிகக் குறைவு", "न्यूनतम"),
  Moderate: B("Moderate", "நடுத்தர", "मध्यम"),
  "Moderate & Pendulous": B("Moderate & Pendulous", "நடுத்தர, தொங்கு", "मध्यम और लटकता"),
  "Large & Pendulous": B("Large & Pendulous", "பெரிய, தொங்கு", "बड़ा और लटकता"),
  "Long & Narrow": B("Long & Narrow", "நீளமான, குறுகிய", "लंबा और संकरा"),
  "Broad & Convex": B("Broad & Convex", "அகலமான, குவிந்த", "चौड़ा और उभरा"),
  "Broad & Concave": B("Broad & Concave", "அகலமான, குழிவான", "चौड़ा और धँसा"),
  "Broad & Flat": B("Broad & Flat", "அகலமான, தட்டையான", "चौड़ा और सपाट"),
  "Long & Convex": B("Long & Convex", "நீளமான, குவிந்த", "लंबा और उभरा"),
  "Narrow & Straight": B("Narrow & Straight", "குறுகிய, நேரான", "संकरा और सीधा"),
  "Short & Concave": B("Short & Concave", "குறுகிய, குழிவான", "छोटा और धँसा"),
  "Short & Straight": B("Short & Straight", "குறுகிய, நேரான", "छोटा और सीधा"),
  "Straight Profile": B("Straight Profile", "நேரான வடிவம்", "सीधा प्रोफ़ाइल"),
  "Small / Compact": B("Small / Compact", "சிறிய / கச்சிதமான", "छोटा / सुगठित"),
  "Small-Medium": B("Small-Medium", "சிறிய-நடுத்தர", "छोटा-मध्यम"),
  "Medium-Large": B("Medium-Large", "நடுத்தர-பெரிய", "मध्यम-बड़ा"),
  Large: B("Large", "பெரிய", "बड़ा"),
  // Added morphology variants
  Absent: B("Absent", "இல்லை", "अनुपस्थित"),
  Prominent: B("Prominent", "தெளிவான திமில்", "प्रमुख कूबड़"),
  "Very large": B("Very large", "மிகப் பெரிய", "बहुत बड़ा"),
  "Moderate, compact": B("Moderate, compact", "நடுத்தர, கச்சிதமான திமில்", "मध्यम, सुगठित"),
  "Medium to large": B("Medium to large", "நடுத்தர-பெரிய", "मध्यम से बड़ा"),
  "Very small (dwarf)": B("Very small (dwarf)", "மிகச் சிறிய (குள்ள இனம்)", "अति छोटा (बौना)"),
  "Very small": B("Very small", "மிகச் சிறிய", "अति छोटा"),
  "Large and pendulous": B("Large and pendulous", "பெரிய, தொங்கும் தாடைத் தோல்", "बड़ा और लटकता गलकंबल"),
  "Loose and pendulous": B("Loose and pendulous", "தளர்வான, தொங்கும் தாடைத் தோல்", "ढीला और लटकता"),
  "Moderate to large": B("Moderate to large", "நடுத்தர முதல் பெரிய தாடைத் தோல்", "मध्यम से बड़ा"),
  Tight: B("Tight", "இறுக்கமான (சிறிய) தாடைத் தோல்", "सख्त / छोटा"),
  Long: B("Long", "நீண்ட", "लंबा"),
  "Long, drooping": B("Long, drooping", "நீண்ட, தொங்கும் காதுகள்", "लंबे, लटकते कान"),
  "Long, pendulous, folded": B("Long, pendulous, folded", "நீண்ட, தொங்கும் மடிந்த காதுகள்", "लंबे, लटकते, मुड़े कान"),
  "Long, tapering": B("Long, tapering", "நீண்ட, கூம்பும் காதுகள்", "लंबे, नुकीले कान"),
  "Medium, drooping": B("Medium, drooping", "நடுத்தர, தொங்கும் காதுகள்", "मध्यम, लटकते कान"),
  Straight: B("Straight", "நேரான", "सीधा"),
  Broad: B("Broad", "அகலமான", "चौड़ा"),
  "Broad forehead": B("Broad forehead", "அகன்ற நெற்றி", "चौड़ा माथा"),
  "Convex forehead, long face": B("Convex forehead, long face", "குவிந்த நெற்றி, நீண்ட முகம்", "उभरा माथा, लंबा चेहरा"),
  "Long, straight": B("Long, straight", "நீண்ட, நேரான முகம்", "लंबा और सीधा चेहरा"),
  "Short, broad forehead": B("Short, broad forehead", "குட்டையான, அகன்ற நெற்றி", "छोटा, चौड़ा माथा"),
  "Short, broad": B("Short, broad", "குட்டையான, அகன்ற முகம்", "छोटा, चौड़ा चेहरा"),
  Variable: B("Variable", "மாறுபடக்கூடியது", "परिवर्तनीय"),
  "Long, backward and upward": B("Long, backward and upward", "நீண்ட, பின்னோக்கி மற்றும் மேல்நோக்கிய கொம்புகள்", "लंबे, पीछे और ऊपर मुड़े सींग"),
  "Long, curved outward and back": B("Long, curved outward and back", "நீண்ட, வெளிப்புறம் வளைந்து பின்னோக்கிய கொம்புகள்", "लंबे, बाहर और पीछे मुड़े सींग"),
  "Long, flat, backward": B("Long, flat, backward", "நீண்ட, தட்டையான, பின்னோக்கிய கொம்புகள்", "लंबे, सपाट, पीछे मुड़े सींग"),
  "Long, flat, drooping": B("Long, flat, drooping", "நீண்ட, தட்டையான, கீழ்நோக்கி தொங்கும் கொம்புகள்", "लंबे, सपाट, नीचे लटकते सींग"),
  "Long, flat, upward": B("Long, flat, upward", "நீண்ட, தட்டையான, மேல்நோக்கிய கொம்புகள்", "लंबे, सपाट, ऊपर मुड़े सींग"),
  "Long, outward": B("Long, outward", "நீண்ட, வெளிப்புற கொம்புகள்", "लंबे, बाहर की ओर सींग"),
  "Long, sweeping": B("Long, sweeping", "நீண்ட, பரந்த வளைவு கொம்புகள்", "लंबे, फैले हुए सींग"),
  "Long, tapering, upward": B("Long, tapering, upward", "நீண்ட, கூரான, மேல்நோக்கிய கொம்புகள்", "लंबे, नुकीले, ऊपर मुड़े सींग"),
  "Lyre-shaped, curving up and in": B("Lyre-shaped, curving up and in", "யாழ் வடிவம், மேல் மற்றும் உள்நோக்கி வளைந்தவை", "वीणा आकार, ऊपर और अंदर मुड़े"),
  "Medium, backward": B("Medium, backward", "நடுத்தர, பின்னோக்கிய கொம்புகள்", "मध्यम, पीछे मुड़े सींग"),
  "Medium, outward": B("Medium, outward", "நடுத்தர, வெளிப்புற கொம்புகள்", "मध्यम, बाहर की ओर सींग"),
  "Short, curved upward": B("Short, curved upward", "குறுகிய, மேல்நோக்கி வளைந்தவை", "छोटे, ऊपर मुड़े सींग"),
  "Short, curving outward": B("Short, curving outward", "குறுகிய, வெளிப்புறம் வளைந்தவை", "छोटे, बाहर मुड़े सींग"),
  "Short, outward": B("Short, outward", "குறுகிய, வெளிப்புற கொம்புகள்", "छोटे, बाहर की ओर सींग"),
  "Short, stout, outward": B("Short, stout, outward", "குறுகிய, தடிமனான வெளிப்புற கொம்புகள்", "छोटे, मोटे, बाहर की ओर सींग"),
  "Short, thick, outward": B("Short, thick, outward", "குறுகிய, தடித்த வெளிப்புற கொம்புகள்", "छोटे, मोटे, बाहर की ओर सींग"),
  "Sickle-shaped, curved back": B("Sickle-shaped, curved back", "அரிவாள் வடிவம், பின்னோக்கி வளைந்தவை", "दरांती आकार, पीछे मुड़े"),
  "Small, curved": B("Small, curved", "சிறிய, வளைந்த கொம்புகள்", "छोटे, मुड़े सींग"),
  "Spiral, backward": B("Spiral, backward", "சுருள், பின்னோக்கிய கொம்புகள்", "घुमावदार, पीछे मुड़े सींग"),
  "Tightly curled, spiral": B("Tightly curled, spiral", "இறுக்கமாக சுருண்ட கொம்புகள்", "कसकर मुड़े, चक्राकार सींग"),
  "Very long, sword-shaped": B("Very long, sword-shaped", "மிக நீண்ட, வாள் வடிவ கொம்புகள்", "अति लंबे, तलवार आकार सींग"),
  "Black and white or brown": B("Black and white or brown", "கருப்பு-வெள்ளை அல்லது பழுப்பு", "काला-सफ़ेद या भूरा"),
  "Black and white patches": B("Black and white patches", "கருப்பு-வெள்ளைத் திட்டுகள்", "काले और सफ़ेद धब्बे"),
  "Black and white": B("Black and white", "கருப்பு மற்றும் வெள்ளை", "काला और सफ़ेद"),
  "Black to brown": B("Black to brown", "கருப்பு முதல் பழுப்பு வரை", "काला से भूरा"),
  "Black with copper sheen": B("Black with copper sheen", "செம்பு பளபளப்புடன் கருப்பு நிறம்", "तांबे जैसी चमक के साथ काला"),
  "Black with white face/leg markings": B("Black with white face/leg markings", "முக/கால் வெள்ளை அடையாளங்களுடன் கருப்பு", "सफ़ेद चेहरे/पैरों के साथ काला"),
  "Black, brown or grey": B("Black, brown or grey", "கருப்பு, பழுப்பு அல்லது சாம்பல்", "काला, भूरा या धूसर"),
  "Deep red to dark brown": B("Deep red to dark brown", "அடர் சிவப்பு முதல் அடர் பழுப்பு வரை", "गहरा लाल से गहरा भूरा"),
  "Grey to white": B("Grey to white", "சாம்பல் முதல் வெள்ளை வரை", "धूसर से सफ़ेद"),
  "Grey with white patches": B("Grey with white patches", "வெள்ளைத் திட்டுகளுடன் சாம்பல்", "सफ़ेद धब्बों के साथ धूसर"),
  "Red and white patches": B("Red and white patches", "சிவப்பு-வெள்ளைத் திட்டுகள்", "लाल और सफ़ेद धब्बे"),
  "Red with white patches": B("Red with white patches", "வெள்ளைத் திட்டுகளுடன் சிவப்பு", "सफ़ेद धब्बों के साथ लाल"),
  "Red, brown or spotted": B("Red, brown or spotted", "சிவப்பு, பழுப்பு அல்லது புள்ளிகள்", "लाल, भूरा या चितकबरा"),
  "Reddish brown with white speckles": B("Reddish brown with white speckles", "வெள்ளிப் புள்ளிகளுடன் செம்பழுப்பு", "सफ़ेद बूंदों के साथ लाल-भूरा"),
  "Reddish brown": B("Reddish brown", "செம்பழுப்பு நிறம்", "लाल-भूरा रंग"),
  "Reddish brown, sometimes white patches": B("Reddish brown, sometimes white patches", "செம்பழுப்பு, சில சமயம் வெள்ளைத் திட்டுகள்", "लाल-भूरा, कभी-कभी सफ़ेद धब्बे"),
  "Silver grey to iron grey": B("Silver grey to iron grey", "வெள்ளி சாம்பல் முதல் இரும்பு சாம்பல் வரை", "चांदी जैसा धूसर से गहरा धूसर"),
  "White to fawn": B("White to fawn", "வெள்ளை முதல் மான் நிறம் வரை", "सफ़ेद से हल्का पीला"),
  "White to grey": B("White to grey", "வெள்ளை முதல் சாம்பல் வரை", "सफ़ेद से धूसर"),
  "White to light grey": B("White to light grey", "வெள்ளை முதல் வெளிர் சாம்பல் வரை", "सफ़ेद से हल्का धूसर"),
  "White, grey or brown": B("White, grey or brown", "வெள்ளை, சாம்பல் அல்லது பழுப்பு", "सफ़ेद, धूसर या भूरा"),
};

export function localizeTraitValue(field, value, lang = "en") {
  if (!value) return "";
  if (lang === "en") return value;
  const direct = TRAIT_VALUES_I18N[value];
  if (direct) return direct[lang] || value;
  const token = String(value).toLowerCase().replace(/[\s/&,()—–-]+/g, "_").replace(/^_+|_+$/g, "");
  const byToken = Object.values(TRAIT_VALUES_I18N).find(
    (entry) => entry.en.toLowerCase().replace(/[\s/&,()—–-]+/g, "_").replace(/^_+|_+$/g, "") === token
  );
  return byToken ? byToken[lang] || value : value;
}

/* ---------------------------------------------------------------- dynamics */

export const DYNAMIC_I18N = {
  species: {
    Cattle: B("Cattle", "மாடு (பசு/காளை)", "गाय / गोवंश"),
    Buffalo: B("Buffalo", "எருமை", "भैंस"),
    cattle: B("Cattle", "மாடு (பசு/காளை)", "गाय / गोवंश"),
    buffalo: B("Buffalo", "எருமை", "भैंस"),
    Unknown: B("Unknown", "தெரியவில்லை", "अज्ञात"),
  },
  gender: {
    female: B("Female (Cow)", "பெண் (பசு)", "मादा (गाय/भैंस)"),
    male: B("Male (Bull/Ox)", "ஆண் (காளை / எருது)", "नर (बैल/सांड)"),
    Female: B("Female", "பெண்", "मादा"),
    Male: B("Male", "ஆண்", "नर"),
  },
  age: {
    calf: B("Calf (< 1 yr)", "கன்று (< 1 வருடம்)", "बछड़ा / पाड़ा (< 1 वर्ष)"),
    young: B("Young / Heifer (1 - 3 yrs)", "இளங்கன்று / கிடாரி (1 - 3 வருடம்)", "युवा / कलोर (1 - 3 वर्ष)"),
    adult: B("Adult (> 3 yrs)", "வளர்ந்த மாடு (> 3 வருடம்)", "वयस्क (> 3 वर्ष)"),
    Calf: B("Calf", "கன்று", "बछड़ा"),
    Young: B("Young", "இளைய", "युवा"),
    Adult: B("Adult", "வளர்ந்த", "वयस्क"),
  },
  utility: {
    "Milch (Dairy)": B("Milch (Dairy)", "கறவை மாடு (பால் பண்ணை)", "दुधारू (डेयरी)"),
    "Draught (Work)": B("Draught (Work)", "இழுவை / உழவு இனம்", "भारवाही नस्ल (कार्य)"),
    "Dual-Purpose": B("Dual-Purpose", "இருநோக்கு இனம் (பால் & உழவு)", "द्विकाजी नस्ल (दूध व काम)"),
    "Ornamental / Dwarf Dairy": B("Ornamental / Dwarf Dairy", "அலங்கார / குள்ள கறவை இனம்", "सजावटी / बौना दुधारू"),
  },
  quality: {
    blurry: B("Blurry", "மங்கலானது", "धुंधला"),
    too_dark: B("Too dark", "மிக இருட்டு", "बहुत अंधेरा"),
    overexposed: B("Too bright", "மிக வெளிச்சம்", "बहुत चमकीला"),
    low_resolution: B("Low resolution", "குறைந்த தெளிவு", "कम रेज़ोल्यूशन"),
    extreme_aspect: B("Extreme frame", "மிக நீளமான சட்டகம்", "अत्यधिक फ़्रेम"),
    unreadable: B("Unreadable file", "படிக்க முடியாத கோப்பு", "अपठनीय फ़ाइल"),
  },
  ood: {
    not_animal: B("No animal detected", "விலங்கு இல்லை", "जानवर नहीं मिला"),
    unclear_features: B("Breed features unclear", "இன அம்சங்கள் தெளிவில்லை", "नस्ल अस्पष्ट"),
    poor_quality: B("Improve the photo", "படத்தை மேம்படுத்தவும்", "फ़ोटो सुधारें"),
    ok: B("Good", "நல்லது", "अच्छा"),
  },
  climate: {
    High: B("High", "உயர்", "उच्च"),
    Moderate: B("Moderate", "மிதமான", "मध्यम"),
    Low: B("Low", "குறைவு", "कम"),
  },
};

export function localizeDynamic(kind, value, lang = "en") {
  const table = DYNAMIC_I18N[kind];
  if (!table || value == null) return value;
  let entry = table[value];
  if (!entry) {
    const lower = String(value).toLowerCase().trim();
    const hitKey = Object.keys(table).find((k) => k.toLowerCase() === lower);
    if (hitKey) entry = table[hitKey];
  }
  return entry ? entry[lang] || entry.en : value;
}

/* ---------------------------------------------------------------- advisory */

const ADVISORY = {
  ta: {
    headlineFemaleAdult: (breed, milk, fat) =>
      `${breed} வளர்ந்த பசு — ${milk} பால், ${fat} கொழுப்பு சரியான தீவனத்தால் சாத்தியம்.`,
    headlineFemaleYoung: (breed) =>
      `${breed} இளம் பெண் கன்று — முதல் கருவூட்டலுக்கு முன் சரியான எடைக்கு வளர்க்கவும்.`,
    headlineMaleAdult: (breed) =>
      `${breed} வளர்ந்த காளை — மதிப்பு உழைப்புத் திறன், இனப்பெருக்கம் மற்றும் ஆரோக்கியமான குளம்புகளில் உள்ளது; பால் அல்ல.`,
    headlineMaleYoung: (breed) =>
      `${breed} இளம் காளை — இனப்பெருக்கக் காளையா, உழைப்பு மாடா அல்லது விற்பனையா என்பதை முன்கூட்டியே முடிவு செய்யவும்.`,
    headlineCalf: (breed) =>
      `${breed} கன்று — முதல் 90 நாட்கள் வளர்ச்சி, நோய் எதிர்ப்பு மற்றும் வாழ்நாள் பால் உற்பத்தியை தீர்மானிக்கும்.`,
    headlineGeneral: (breed, utility) => `${breed} (${utility}) — இனத் தகவல் மற்றும் பராமரிப்பு வழிகாட்டி.`,
    actionsFemaleAdult: (extra) => [
      "தினமும் 2-3 முறை நிலையான நேரத்தில் பால் கறக்கவும்; கறந்த பிறகு தீவனம் கொடுக்கவும்.",
      `5 லிட்டருக்கு மேல் ஒவ்வொரு லிட்டருக்கும் 0.4 கிலோ தீவனம் கூடுதலாக (இந்த இனத்திற்கு சுமார் ${extra} கிலோ).`,
      "தினமும் 50-75 கிராம் தாது உப்பு கலவை, 30-50 கிராம் சாதாரண உப்பு; பால் சுரப்பு காலத்தில் கால்சியம்.",
      "மடிநோய் (மஸ்டிடிஸ்) தடுப்பு: மடியை கழுவி உலர்த்தவும், கறந்த பிறகு காம்பு மூழ்கவும், உலர்ந்த படுக்கை.",
      "கன்று ஈனுவதற்கு 60 நாட்கள் முன் கறவையை நிறுத்தவும் (வறட்சிக் காலம்).",
    ],
    actionsFemaleYoung: (afc) => [
      "வளர்ச்சிக்கு நல்ல பச்சைத் தீவனம் + 1-1.5 கிலோ தீவனம் + தாது உப்பு கலவை தினமும்.",
      `முதல் ஈற்றுக்கான வயது இலக்கு: ${afc}.`,
      "மாதம் ஒருமுறை எடை அல்லது மார்பு அளவு எடுக்கவும்; நாளொன்றுக்கு 400-600 கிராம் வளர்ச்சி.",
      "முதல் பருவ அறிகுறிகளை (சினைப் பருவம்: அமைதியின்மை, சளி, மற்றவற்றின் மீது ஏறுதல்) கவனித்து தேதியைப் பதிவு செய்யவும்.",
      "கருவூட்டலுக்கு முன் காலாண்டு குடல் புழு நீக்கம் மற்றும் FMD/HS/BQ தடுப்பூசி.",
    ],
    actionsMaleAdult: [
      "உழைப்புத் திறனை படிப்படியாக வளர்க்கவும்; இலகுவான சுமையில் தொடங்கி 2-3 வாரங்களில் அதிகரிக்கவும்.",
      "வேலை நாட்களில் 0.5-1 கிலோ கூடுதல் தீவனம்; வாரத்திற்கு 2 ஓய்வு நாட்கள்.",
      "மாதம் ஒருமுறை குளம்புகளை சரிபார்க்கவும்; காயங்களுக்கு உடனடி சிகிச்சை.",
      "மென்மையான தோல் பட்டை/நுகத்தடி பயன்படுத்தவும்; கழுத்து மற்றும் தோளில் புண்களை சரிபார்க்கவும்.",
      "இனப்பெருக்கக் காளையாக இருந்தால் சேவைகளைப் பதிவு செய்து ஆண்டுதோறும் விந்து பரிசோதனை செய்யவும்.",
    ],
    actionsMaleYoung: [
      "நிலையான வளர்ச்சிக்கு தீவனம் (நாளொன்றுக்கு 0.5-0.7 கிலோ); முதிர்ச்சிக்கு முன் கொழுப்பேற்ற வேண்டாம்.",
      "12-18 மாதங்களில் முடிவு: இனப்பெருக்கக் காளையா, உழைப்பு மாடா, விற்பனையா.",
      "உழைப்புக்காக வளர்த்தால் தகுதியான கால்நடை மருத்துவரால் விதை நீக்கம் / ஆண்மை நீக்கம் (அடிப்பு) செய்யவும்.",
      "சிறு வயதிலிருந்தே தினமும் கையாளவும் — பழக்கமான மாடு பாதுகாப்பானது, அதிக விலை.",
      "FMD/HS/BQ தடுப்பூசி மற்றும் காலாண்டு புழு நீக்கம்.",
    ],
    actionsCalf: [
      "சீம்பால் (கொலஸ்ட்ரம்): பிறந்த 1-2 மணி நேரத்தில் 1-2 லிட்டர், பின்னர் உடல் எடையின் 10% தினமும் 3 நாட்கள்.",
      "10, 30, 60 நாட்களில் புழு நீக்கம்; பின்னர் 3 மாதங்களுக்கு ஒருமுறை.",
      "2 வது வாரத்திலிருந்து கன்று தீவனம் (18-20% புரதம்); 8-12 வாரங்களில் தாய்ப்பால் நிறுத்தம்.",
      "தடுப்பூசி: 4-6 மாதங்களில் FMD, 6 மாதங்களில் HS/BQ (பின்னர் ஆண்டுதோறும்); பெண் கன்றுகளுக்கு 4-8 மாதங்களில் புருசெல்லோசிஸ்.",
      "சுத்தமான நீர், உலர்ந்த படுக்கை, நிழல் எப்போதும்.",
    ],
    vaccination: [
      "FMD — 4-6 மாதங்களில் முதல் தடுப்பூசி, பின்னர் 6 மாதங்களுக்கு ஒருமுறை (NADCP)",
      "தொண்டை அடைப்பான் (HS) — ஆண்டுதோறும், பருவமழைக்கு முன்",
      "சப்பை நோய் (BQ) — ஆண்டுதோறும், பருவமழைக்கு முன்",
      "புருசெல்லோசிஸ் — பெண் கன்றுகளுக்கு 4-8 மாதங்களில் ஒருமுறை",
    ],
    watchAdult: [
      "மடிநோய் (மஸ்டிடிஸ்) — தினமும் மடியை சரிபார்க்கவும், மாதம் ஒருமுறை CMT",
      "கன்று போட்ட பின் கால்சியம் குறைபாடு",
      "பால் சுரப்பின் தொடக்கத்தில் உணவு மறுப்பு",
      "3 கருவூட்டல்களுக்கு மேல் கருத்தரிக்கவில்லை என்றால் மருத்துவர் பரிசோதனை",
    ],
    watchCalf: [
      "வயிற்றுப்போக்கு — உடனடியாக தனிமைப்படுத்தி ORS கொடுக்கவும்",
      "நிமோனியா — காற்றோட்டமான கொட்டகை",
      "தொப்புள் தொற்று முதல் வாரத்தில்",
      "ஈரமான கொட்டகையில் காக்சிடியோசிஸ்",
    ],
    watchMale: [
      "நுகத்தடி/கயிறு புண்கள் மற்றும் தோள் காயங்கள்",
      "ஈரமான தரையில் குளம்பு அழுகல்",
      "இனச்சேர்க்கைப் பருவத்தில் ஆக்ரோஷம் — பாதுகாப்பான கட்டு",
    ],
    deworming: ["10 வது நாள்", "30 வது நாள்", "60 வது நாள்", "பின்னர் 3 மாதங்களுக்கு ஒருமுறை"],
    schemes: [
      ["Rashtriya Gokul Mission", "ராஷ்டிரிய கோகுல் மிஷன் (RGM) — உள்ளூர் இனப் பாதுகாப்பு மற்றும் செயற்கை கருவூட்டல்"],
      ["National Livestock Mission", "தேசிய கால்நடை மிஷன் (NLM) — தீவனம் மற்றும் தொழில்முனைவு ஆதரவு"],
      ["NADCP", "NADCP — FMD மற்றும் புருசெல்லோசிஸ் இலவச தடுப்பூசி"],
      ["Kisan Credit Card", "கிசான் கிரெடிட் கார்டு (KCC-AH) — தீவனம் மற்றும் கால்நடை பராமரிப்பு கடன்"],
      ["Dairy Infrastructure", "DIDF — பால் பதப்படுத்துதல் மற்றும் குளிரூட்டல் உதவி"],
      ["NBAGR", "NBAGR / ICAR — இனப் பாதுகாப்பு மற்றும் உயர்ந்த இனப்பெருக்க ஆதரவு"],
    ],
    schemeFallback: "மாநில கால்நடை வளர்ச்சி வாரிய ஆதரவு",
    disclaimer:
      "இது பொதுவான வழிகாட்டுதல் மட்டுமே; கால்நடை மருத்துவர் அல்லது மாநில கால்நடை துறையுடன் உறுதிப்படுத்தவும்.",
    feedNote: "தீவனத்தை 2-3 வேளைகளாகப் பிரிக்கவும்; 5 லிட்டருக்கு மேல் பால் எனில் ஒவ்வொரு லிட்டருக்கும் 0.4 கிலோ தீவனம்.",
    regionalDefault: "இந்த இனத்திற்கு வெளியிடப்பட்ட தரவு குறைவு — மதிப்புகள் பிராந்திய மதிப்பீடுகள்.",
    milkLabel: "பால்",
    draughtNote: "ஆண் விலங்கிற்கு பால் அளவு முதன்மையானது அல்ல; உழைப்பு மற்றும் இனப்பெருக்க மதிப்பே முக்கியம்.",
  },
  hi: {
    headlineFemaleAdult: (breed, milk, fat) =>
      `${breed} वयस्क मादा — सही आहार से ${milk} दूध और ${fat} वसा संभव है।`,
    headlineFemaleYoung: (breed) =>
      `${breed} युवा मादा — पहले गर्भाधान से पहले सही वजन तक बढ़ाएँ।`,
    headlineMaleAdult: (breed) =>
      `${breed} वयस्क नर — मूल्य भारवाहन क्षमता, प्रजनन और स्वस्थ खुरों में है, दूध में नहीं।`,
    headlineMaleYoung: (breed) =>
      `${breed} युवा नर — समय रहते तय करें: प्रजनन बैल, भारवाहन बैल या बिक्री।`,
    headlineCalf: (breed) =>
      `${breed} बछड़ा — पहले 90 दिन वृद्धि, रोग-प्रतिरोधक क्षमता और जीवनभर का दूध तय करते हैं।`,
    headlineGeneral: (breed, utility) => `${breed} (${utility}) — नस्ल जानकारी और देखभाल मार्गदर्शन।`,
    actionsFemaleAdult: (extra) => [
      "दिन में 2-3 बार निश्चित समय पर दूध निकालें; दूध के बाद दाना दें।",
      `5 लीटर से अधिक पर प्रति लीटर 0.4 किग्रा अतिरिक्त दाना (इस नस्ल के लिए लगभग ${extra} किग्रा)।`,
      "रोज़ 50-75 ग्राम खनिज मिश्रण और 30-50 ग्राम नमक; दूध देने के शिखर पर कैल्शियम।",
      "थनैला रोग (मास्टाइटिस) से बचाव: थन धोकर सुखाएँ, दूध के बाद टीट डिप, साफ़ बिछावन, पूरा दूध निकालें।",
      "ब्याने से 60 दिन पहले दूध निकालना बंद करें।",
    ],
    actionsFemaleYoung: (afc) => [
      "वृद्धि के लिए अच्छा हरा चारा + 1-1.5 किग्रा दाना + खनिज मिश्रण रोज़।",
      `पहले ब्याने का लक्ष्य: ${afc}.`,
      "महीने में एक बार वजन या छाती मापें; रोज़ 400-600 ग्राम वृद्धि।",
      "पहले गर्मी के लक्षण (बेचैनी, स्राव, दूसरों पर चढ़ना) देखें और तारीख नोट करें।",
      "गर्भाधान से पहले हर तीन माह कृमिनाशन और FMD/HS/BQ टीका।",
    ],
    actionsMaleAdult: [
      "भार क्षमता धीरे-धीरे बढ़ाएँ; हल्के भार से शुरू कर 2-3 हफ्तों में बढ़ाएँ।",
      "काम के दिन 0.5-1 किग्रा अतिरिक्त दाना; हफ्ते में 2 दिन आराम।",
      "महीने में एक बार खुर जाँचें; घाव का तुरंत इलाज करें।",
      "नरम गद्देदार जुआ/हार्नेस लगाएँ; गर्दन और कंधे पर घाव देखें।",
      "प्रजनन बैल रखें तो सेवा दर्ज करें और सालाना वीर्य जाँच कराएँ।",
    ],
    actionsMaleYoung: [
      "स्थिर वृद्धि के लिए चारा (रोज़ 0.5-0.7 किग्रा बढ़ोतरी); परिपक्वता से पहले मोटा न करें।",
      "12-18 महीने में तय करें: प्रजनन बैल, भारवाहन बैल या बिक्री।",
      "भारवाहन के लिए बधिया करना हो तो योग्य पशु चिकित्सक से कराएँ।",
      "छोटी उम्र से रोज़ संभालें — शांत पशु सुरक्षित और अधिक मूल्य का होता है।",
      "FMD/HS/BQ टीका और हर तीन माह कृमिनाशन।",
    ],
    actionsCalf: [
      "खीस (कोलोस्ट्रम): जन्म के 1-2 घंटे में 1-2 लीटर, फिर 3 दिन तक शरीर के वजन का 10% खीस पिलाएँ।",
      "10, 30 और 60 दिन पर कृमिनाशन; फिर हर 3 महीने।",
      "दूसरे हफ्ते से काफ स्टार्टर (18-20% प्रोटीन); 8-12 हफ्ते में दूध छुड़ाएँ।",
      "टीका: 4-6 माह में FMD, 6 माह में HS/BQ (फिर सालाना); मादा बछड़ों को 4-8 माह में ब्रुसेलोसिस।",
      "साफ़ पानी, सूखा बिछावन और छाया हमेशा।",
    ],
    vaccination: [
      "FMD — 4-6 माह में पहला टीका, फिर हर 6 माह (NADCP)",
      "HS (गलाघोंटू) — सालाना, मानसून से पहले",
      "BQ (लंगड़ा बुखार) — सालाना, मानसून से पहले",
      "ब्रुसेलोसिस — मादा बछड़ों को 4-8 माह में एक बार",
    ],
    watchAdult: [
      "थनैला रोग (मास्टाइटिस) — रोज़ थन जाँचें, महीने में CMT",
      "ब्याने के बाद कैल्शियम की कमी",
      "दूध शुरू होते समय भूख कम लगना",
      "3 बार से अधिक गर्भाधान में सफलता न हो तो पशु चिकित्सक जाँच",
    ],
    watchCalf: [
      "दस्त — तुरंत अलग करें और ORS दें",
      "निमोनिया — हवादार बाड़ा",
      "पहले हफ्ते नाभि संक्रमण",
      "गीले बाड़े में कोक्सीडियोसिस",
    ],
    watchMale: [
      "जुआ/रस्सी के घाव और कंधे की चोट",
      "गीले फर्श पर खुर सड़न",
      "सांड के मौसम में आक्रामकता — सुरक्षित बाँधना",
    ],
    deworming: ["10वाँ दिन", "30वाँ दिन", "60वाँ दिन", "फिर हर 3 महीने"],
    schemes: [
      ["Rashtriya Gokul Mission", "राष्ट्रीय गोकुल मिशन (RGM) — देशी नस्ल संरक्षण और कृत्रिम गर्भाधान"],
      ["National Livestock Mission", "राष्ट्रीय पशुधन मिशन (NLM) — चारा और उद्यमिता सहायता"],
      ["NADCP", "NADCP — FMD और ब्रुसेलोसिस मुफ्त टीकाकरण"],
      ["Kisan Credit Card", "किसान क्रेडिट कार्ड (KCC-AH) — चारा और पशु देखभाल ऋण"],
      ["Dairy Infrastructure", "DIDF — दुग्ध प्रसंस्करण और शीतलन सहायता"],
      ["NBAGR", "NBAGR / ICAR — नस्ल संरक्षण और उन्नत प्रजनन सहायता"],
    ],
    schemeFallback: "राज्य पशुपालन विकास बोर्ड सहायता",
    disclaimer:
      "यह केवल सामान्य मार्गदर्शन है; पशु चिकित्सक या राज्य पशुपालन विभाग से पुष्टि करें।",
    feedNote: "दाना 2-3 बार में बाँटें; 5 लीटर से अधिक दूध पर प्रति लीटर 0.4 किग्रा अतिरिक्त दाना।",
    regionalDefault: "इस नस्ल पर प्रकाशित डेटा सीमित है — आँकड़े क्षेत्रीय अनुमान हैं।",
    milkLabel: "दूध",
    draughtNote: "नर के लिए दूध मुख्य मापदंड नहीं है; भारवाहन और प्रजनन मूल्य महत्वपूर्ण है।",
  },
};

export function localizeScheme(scheme, lang) {
  if (lang === "en" || !scheme) return scheme;
  const table = ADVISORY[lang]?.schemes || [];
  const hit = table.find(([needle]) => String(scheme).toLowerCase().includes(needle.toLowerCase()));
  return hit ? hit[1] : ADVISORY[lang]?.schemeFallback || scheme;
}

export function renderLocalizedAdvisory(advisoryData, lang = "en", helpers = {}) {
  if (!advisoryData || lang === "en") return advisoryData;
  return localizeAdvisory(advisoryData, lang, helpers);
}

export function localizeAdvisory(advisory, lang, helpers = {}) {
  if (!advisory || advisory.error || advisory.skipped) return advisory;
  const table = ADVISORY[lang];
  if (!table) return advisory;

  const { tBreed = (x) => x, tDynamic = (k, v) => v } = helpers;
  const breed = tBreed(advisory.breed?.slug || advisory.breed?.display_name || "");
  const utility = tDynamic("utility", advisory.breed?.utility_type);
  const context = advisory.context || {};
  const g = String(context.gender || "").toLowerCase();
  const a = String(context.age_group || "").toLowerCase();
  const milk = advisory.production?.milk;
  const daily = milk?.daily_yield || "";
  const fat = milk?.fat || "";
  const lo = parseFloat(String(daily).split("-")[0]) || 0;
  const hi = parseFloat(String(daily).split("-")[1]) || lo;
  const extra = `${Math.max(0, (lo - 5) * 0.4).toFixed(1)}-${Math.max(0, (hi - 5) * 0.4).toFixed(1)}`;

  let headline;
  let actions;
  if (a === "calf") {
    headline = table.headlineCalf(breed);
    actions = table.actionsCalf;
  } else if (g === "female" && a === "adult") {
    headline = table.headlineFemaleAdult(breed, daily, fat);
    actions = table.actionsFemaleAdult(extra);
  } else if (g === "female") {
    headline = table.headlineFemaleYoung(breed);
    const afc = advisory.breeding?.age_at_first_calving || "";
    actions = table.actionsFemaleYoung(afc);
  } else if (g === "male" && a === "adult") {
    headline = table.headlineMaleAdult(breed);
    actions = table.actionsMaleAdult;
  } else if (g === "male") {
    headline = table.headlineMaleYoung(breed);
    actions = table.actionsMaleYoung;
  } else {
    headline = table.headlineGeneral(breed, utility);
    actions = [];
  }

  const watchKey = a === "calf" ? "watchCalf" : g === "male" ? "watchMale" : "watchAdult";
  const health = advisory.health || {};
  const localizedHealth = {
    ...health,
    vaccination: table.vaccination,
    watch_for: table[watchKey],
    deworming_schedule: a === "calf" ? table.deworming : health.deworming_schedule,
  };

  return {
    ...advisory,
    headline,
    priority_actions: actions.length ? actions : advisory.priority_actions,
    nutrition: { ...(advisory.nutrition || {}), notes: table.feedNote },
    health: localizedHealth,
    economics: {
      ...(advisory.economics || {}),
      schemes: (advisory.economics?.schemes || advisory.schemes || []).map((s) =>
        localizeScheme(s, lang)
      ),
    },
    schemes: (advisory.schemes || []).map((s) => localizeScheme(s, lang)),
    warnings: a === "calf"
      ? []
      : g === "male"
      ? [table.draughtNote]
      : [table.watchAdult[0], table.watchAdult[1]],
    disclaimer: table.disclaimer,
    localized: true,
  };
}
