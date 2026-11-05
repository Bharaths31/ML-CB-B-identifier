"use strict";

const LANGS = ["en", "ta", "hi"];

const NOT_AVAILABLE = {
  en: "Not specified",
  ta: "குறிப்பிடப்படவில்லை",
  hi: "उपलब्ध नहीं",
};

const UTILITY = {
  en: { Milch: "Milch (Dairy)", Draught: "Draught (Work)", Dual: "Dual purpose" },
  ta: { Milch: "பால் தரும்", Draught: "உழவு", Dual: "இருநோக்கம்" },
  hi: { Milch: "दुधारू", Draught: "भारवाही", Dual: "द्विउद्देशीय" },
};

const SCHEMES = {
  en: {
    rgm: {
      name: "Rashtriya Gokul Mission (RGM)",
      description: "Indigenous breed conservation, AI and Gokul Gram support.",
    },
    nlm: {
      name: "National Livestock Mission (NLM)",
      description: "Fodder development, entrepreneurship and livestock support.",
    },
    nadcp: {
      name: "NADCP",
      description: "Free FMD and brucellosis vaccination under the national programme.",
    },
    kcc: {
      name: "Kisan Credit Card (KCC-AH)",
      description: "Working capital for fodder, feed and veterinary care.",
    },
    nedp: {
      name: "Dairy Entrepreneurship Development Programme",
      description: "Support for small dairy units and equipment.",
    },
    insurance: {
      name: "Livestock insurance",
      description: "Protection against accidental death and eligible disease losses.",
    },
  },
  ta: {
    rgm: {
      name: "ராஷ்டிரிய கோகுல் மிஷன் (RGM)",
      description: "உள்ளூர் இனப் பாதுகாப்பு, செயற்கை விந்து ஊட்டம் மற்றும் கோகுல் கிராம் ஆதரவு.",
    },
    nlm: {
      name: "தேசிய கால்நடை மிஷன் (NLM)",
      description: "தீவன மேம்பாடு, தொழில்முனைவு மற்றும் கால்நடை ஆதரவு.",
    },
    nadcp: {
      name: "NADCP",
      description: "தேசிய திட்டத்தின் கீழ் இலவச FMD மற்றும் புருசெல்லோசிஸ் தடுப்பூசி.",
    },
    kcc: {
      name: "கிசான் கிரெடிட் கார்டு (KCC-AH)",
      description: "தீவனம், உணவு மற்றும் கால்நடை சிகிச்சைக்கான செயல்பாட்டு மூலதனம்.",
    },
    nedp: {
      name: "பால் தொழில்முனைவு மேம்பாட்டுத் திட்டம்",
      description: "சிறு பால் பண்ணைகள் மற்றும் உபகரணங்களுக்கான ஆதரவு.",
    },
    insurance: {
      name: "கால்நடை காப்பீடு",
      description: "விபத்து இறப்பு மற்றும் தகுதியான நோய் இழப்புகளுக்கு பாதுகாப்பு.",
    },
  },
  hi: {
    rgm: {
      name: "राष्ट्रीय गोकुल मिशन (RGM)",
      description: "देशी नस्ल संरक्षण, कृत्रिम गर्भाधान और गोकुल ग्राम सहायता।",
    },
    nlm: {
      name: "राष्ट्रीय पशुधन मिशन (NLM)",
      description: "चारा विकास, उद्यमिता और पशुधन सहायता।",
    },
    nadcp: {
      name: "NADCP",
      description: "राष्ट्रीय कार्यक्रम के तहत मुफ्त FMD और ब्रुसेलोसिस टीकाकरण।",
    },
    kcc: {
      name: "किसान क्रेडिट कार्ड (KCC-AH)",
      description: "चारा, फीड और पशु चिकित्सा के लिए कार्यशील पूंजी।",
    },
    nedp: {
      name: "डेयरी उद्यमिता विकास कार्यक्रम",
      description: "छोटी डेयरी इकाइयों और उपकरणों के लिए सहायता।",
    },
    insurance: {
      name: "पशुधन बीमा",
      description: "दुर्घटना मृत्यु और पात्र रोग नुकसान के लिए सुरक्षा।",
    },
  },
};

const COPY = {
  en: {
    labels: {
      breed: "Breed",
      origin: "Native tract",
      utility: "Utility",
      milk: "Milk yield",
      fat: "Fat",
      market: "Typical market price",
      markers: "Visual markers",
      schemes: "Useful schemes",
      context: "Advisory context",
      gender: "Gender",
      age: "Age group",
      milkVisible: "Milk yield shown",
      draughtVisible: "Draught power shown",
      source: "Source",
      disclaimer: "This is general guidance, not a veterinary diagnosis. Confirm feed, prices and vaccination timing with your local veterinarian or extension officer.",
    },
    gender: { Female: "Female", Male: "Male", Unknown: "Not specified" },
    age: { Calf: "Calf (< 1 year)", Young: "Young (1–3 years)", Adult: "Adult (> 3 years)" },
    titles: {
      Female_Adult: "Lactation nutrition and udder care",
      Male_Adult: "Draught power and breeding value",
      Calf: "Calf growth and vaccination",
      Young: "Growth and first breeding",
      General_Adult: "General care and health",
    },
    summaries: {
      Female_Adult: (c) =>
        `${c.breedName} is a ${c.utility.toLowerCase()} breed. This card focuses on lactation feed, daily milk, calcium and mastitis prevention.`,
      Male_Adult: (c) =>
        `${c.breedName} is a ${c.utility.toLowerCase()} breed. This card focuses on work nutrition, breeding value and bull health; milk yield is hidden.`,
      Calf: (c) =>
        `${c.breedName} calf care: colostrum, starter feed, deworming and the FMD/brucellosis vaccination schedule.`,
      Young: (c) =>
        `${c.breedName} is growing. Focus on balanced growth, first breeding age and routine deworming/vaccination.`,
      General_Adult: (c) =>
        `General care for ${c.breedName}: balanced feed, clean water, shade, hoof care and routine vaccination.`,
    },
    cards: {
      lactation_diet: {
        title: "Lactation diet",
        body: (c) =>
          `Provide ${c.feeding.green} kg green fodder, ${c.feeding.dry} kg dry fodder and ${c.feeding.concentrate} kg concentrate per day. Split concentrate into 2–3 feeds and give ${c.feeding.water} litres of clean water.`,
      },
      milk_yield: {
        title: "Daily milk and fat",
        body: (c) =>
          `Typical yield for this breed is ${c.milk.daily} with ${c.milk.fat} fat. Actual yield depends on parity, feed, water and milking routine.`,
      },
      calcium: {
        title: "Calcium and minerals",
        body: (c) =>
          `Give ${c.feeding.mineral} g mineral mixture daily. During peak lactation add a calcium/phosphorus supplement and keep the mineral mixture dry and free-flowing.`,
      },
      mastitis: {
        title: "Mastitis prevention",
        body: () =>
          "Check the udder before and after milking. Wash hands, use clean dry bedding, dip teats after milking and strip the udder fully. Heat, pain, hard quarters or clots need a vet the same day.",
      },
      breeding: {
        title: "Heat detection and breeding",
        body: () =>
          "Watch for standing heat every 18–21 days. Inseminate 12–18 hours after standing heat. Aim for a 12–15 month calving interval and keep AI/calving records.",
      },
      work_diet: {
        title: "Work nutrition",
        body: (c) =>
          `On working days add 0.5–1 kg concentrate to the normal ration (${c.feeding.green} kg green fodder, ${c.feeding.dry} kg dry fodder). Feed roughage first and allow 2 rest days each week.`,
      },
      draught: {
        title: "Draught power",
        body: () =>
          "Use a well-fitted yoke, work in the cool hours, provide shade and water at every rest, and avoid heavy work for 2–3 weeks after illness or vaccination.",
      },
      breeding_value: {
        title: "Breeding value",
        body: () =>
          "Keep records of dam, sire, growth, temperament and work performance. Use a proven AI bull and avoid breeding a bull with known hereditary defects.",
      },
      bull_health: {
        title: "Bull health",
        body: () =>
          "Deworm every 6 months, vaccinate for FMD and HS/BQ as per the local schedule, check hooves monthly and control ticks after the monsoon.",
      },
      colostrum: {
        title: "Colostrum in the first hour",
        body: () =>
          "Feed colostrum within the first hour and continue for the first 3 days — about 10% of body weight per day. Colostrum is the calf's first disease protection.",
      },
      starter: {
        title: "Calf starter feed",
        body: () =>
          "Introduce calf starter at 1–2 weeks with clean water and good hay. Wean at 8–12 weeks when the calf eats 1–1.5 kg starter per day.",
      },
      deworming: {
        title: "Deworming",
        body: () =>
          "Deworm at 2–3 weeks and repeat every 4–6 weeks until 6 months, then every 3–6 months. Use the dose recommended by your veterinarian.",
      },
      vaccination: {
        title: "Vaccination schedule",
        body: () =>
          "FMD: first dose at 4–6 months, booster 1 month later, then every 6 months. Brucellosis: vaccinate female calves once at 4–8 months. Follow the local veterinary schedule.",
      },
      calf_housing: {
        title: "Calf housing",
        body: () =>
          "Keep calves in a dry, draft-free pen with 1.5–2 m² per calf, clean bedding and fresh water. Separate sick calves immediately.",
      },
      growth_diet: {
        title: "Growth diet",
        body: (c) =>
          `Feed ${c.feeding.green} kg green fodder, ${c.feeding.dry} kg dry fodder and ${c.feeding.concentrate} kg concentrate per day, plus ${c.feeding.mineral} g mineral mixture.`,
      },
      puberty: {
        title: "First breeding",
        body: () =>
          "Female: plan first AI at 18–24 months and 250–300 kg body weight. Male: evaluate growth, feet and temperament before using for breeding.",
      },
      young_health: {
        title: "Routine health",
        body: () =>
          "Deworm every 3–6 months. Vaccinate for FMD, HS and BQ as per the local schedule. Check for ticks, lameness and poor growth every month.",
      },
      general_feed: {
        title: "Balanced feed",
        body: (c) =>
          `Provide ${c.feeding.green} kg green fodder, ${c.feeding.dry} kg dry fodder, ${c.feeding.concentrate} kg concentrate and ${c.feeding.mineral} g mineral mixture per day.`,
      },
      general_health: {
        title: "Routine health",
        body: () =>
          "Provide clean water, shade and dry flooring. Deworm every 6 months, vaccinate for FMD and HS/BQ, and check hooves and ticks monthly.",
      },
    },
    vaccination: {
      title: "Vaccination and deworming timeline",
      items: [
        { age: "0–3 days", item: "Colostrum — 10% of body weight per day", note: "First protection" },
        { age: "2–3 weeks", item: "First deworming", note: "Repeat every 4–6 weeks" },
        { age: "4–6 months", item: "FMD first dose", note: "Booster after 1 month" },
        { age: "4–8 months", item: "Brucellosis (female calves)", note: "Single dose" },
        { age: "6 months", item: "HS and BQ vaccination", note: "As per local schedule" },
      ],
    },
    schemes: {
      title: "Schemes you can use",
      note: "Ask your local veterinary office for the current eligibility and subsidy rates.",
    },
    disclaimer:
      "This is general guidance, not a veterinary diagnosis. Confirm feed, prices and vaccination timing with your local veterinarian or extension officer.",
  },
  ta: {
    labels: {
      breed: "இனம்",
      origin: "தாயகப் பகுதி",
      utility: "பயன்பாடு",
      milk: "பால் உற்பத்தி",
      fat: "கொழுப்பு",
      market: "வழக்கமான சந்தை விலை",
      markers: "அடையாள அம்சங்கள்",
      schemes: "பயனுள்ள திட்டங்கள்",
      context: "ஆலோசனை சூழல்",
      gender: "பாலினம்",
      age: "வயது குழு",
      milkVisible: "பால் உற்பத்தி காட்டப்படுகிறது",
      draughtVisible: "உழவு திறன் காட்டப்படுகிறது",
      source: "மூலம்",
      disclaimer: "இது பொதுவான வழிகாட்டல்; கால்நடை மருத்துவ ஆலோசனை அல்ல. தீவனம், விலை மற்றும் தடுப்பூசி நேரத்தை உங்கள் கால்நடை மருத்துவரிடம் உறுதிசெய்யவும்.",
    },
    gender: { Female: "பெண்", Male: "ஆண்", Unknown: "குறிப்பிடப்படவில்லை" },
    age: { Calf: "கன்று (1 வயதுக்கு கீழ்)", Young: "இளம் (1–3 வயது)", Adult: "முதிர்ந்த (3 வயதுக்கு மேல்)" },
    titles: {
      Female_Adult: "பால் உற்பத்தி ஊட்டம் மற்றும் மடி பராமரிப்பு",
      Male_Adult: "உழவு திறன் மற்றும் இனப்பெருக்க மதிப்பு",
      Calf: "கன்று வளர்ச்சி மற்றும் தடுப்பூசி",
      Young: "வளர்ச்சி மற்றும் முதல் இனப்பெருக்கம்",
      General_Adult: "பொது பராமரிப்பு மற்றும் ஆரோக்கியம்",
    },
    summaries: {
      Female_Adult: (c) =>
        `${c.breedName} ஒரு ${c.utility} இனம். இந்த அட்டை பால் உற்பத்தி ஊட்டம், தினசரி பால், கால்சியம் மற்றும் மடி அழற்சி தடுப்பில் கவனம் செலுத்துகிறது.`,
      Male_Adult: (c) =>
        `${c.breedName} ஒரு ${c.utility} இனம். இந்த அட்டை உழவு ஊட்டம், இனப்பெருக்க மதிப்பு மற்றும் காளை ஆரோக்கியத்தில் கவனம் செலுத்துகிறது; பால் உற்பத்தி மறைக்கப்பட்டுள்ளது.`,
      Calf: (c) =>
        `${c.breedName} கன்று பராமரிப்பு: சீம்பால், தொடக்கத் தீவனம், கிருமி நீக்கம் மற்றும் FMD/புருசெல்லோசிஸ் தடுப்பூசி அட்டவணை.`,
      Young: (c) =>
        `${c.breedName} வளர்ந்து வருகிறது. சமநிலை வளர்ச்சி, முதல் இனப்பெருக்க வயது மற்றும் வழக்கமான கிருமி நீக்கம்/தடுப்பூசியில் கவனம் செலுத்துங்கள்.`,
      General_Adult: (c) =>
        `${c.breedName} பொது பராமரிப்பு: சமநிலை தீவனம், சுத்தமான நீர், நிழல், குளம்பு பராமரிப்பு மற்றும் வழக்கமான தடுப்பூசி.`,
    },
    cards: {
      lactation_diet: {
        title: "பால் உற்பத்தி தீவனம்",
        body: (c) =>
          `தினமும் ${c.feeding.green} கிலோ பச்சைத் தீவனம், ${c.feeding.dry} கிலோ உலர் தீவனம், ${c.feeding.concentrate} கிலோ செறிவூட்டப்பட்ட தீவனம் கொடுக்கவும். செறிவூட்டலை 2–3 வேளைகளாகப் பிரித்து, ${c.feeding.water} லிட்டர் சுத்தமான நீர் வழங்கவும்.`,
      },
      milk_yield: {
        title: "தினசரி பால் மற்றும் கொழுப்பு",
        body: (c) =>
          `இந்த இனத்திற்கு வழக்கமான உற்பத்தி ${c.milk.daily}, கொழுப்பு ${c.milk.fat}. உண்மையான உற்பத்தி பருவம், தீவனம், நீர் மற்றும் பால் கறக்கும் முறையைப் பொறுத்தது.`,
      },
      calcium: {
        title: "கால்சியம் மற்றும் தாதுக்கள்",
        body: (c) =>
          `தினமும் ${c.feeding.mineral} கிராம் தாது கலவை கொடுக்கவும். அதிக பால் உற்பத்தி காலத்தில் கால்சியம்/பாஸ்பரஸ் கூடுதல் கொடுக்கவும்; தாது கலவையை உலர்வாக வைக்கவும்.`,
      },
      mastitis: {
        title: "மடி அழற்சி தடுப்பு",
        body: () =>
          "பால் கறப்பதற்கு முன்னும் பின்னும் மடியைப் பரிசோதிக்கவும். கைகளைக் கழுவி, சுத்தமான உலர் படுக்கை பயன்படுத்தி, கறந்த பின் காம்புகளை டிப் செய்து, மடியை முழுமையாகக் கறக்கவும். வெப்பம், வலி, கடினமான கால் அல்லது உறைதல் இருந்தால் அதே நாள் மருத்துவரை அணுகவும்.",
      },
      breeding: {
        title: "சூடு கண்டறிதல் மற்றும் இனப்பெருக்கம்",
        body: () =>
          "18–21 நாட்களுக்கு ஒருமுறை சூடு அறிகுறிகளைக் கவனிக்கவும். சூடு கண்ட 12–18 மணி நேரத்தில் செயற்கை விந்து ஊட்டம் செய்யவும். 12–15 மாத இடைவெளியை இலக்காகக் கொள்ளவும்.",
      },
      work_diet: {
        title: "உழவு தீவனம்",
        body: (c) =>
          `உழவு நாட்களில் வழக்கமான தீவனத்துடன் 0.5–1 கிலோ செறிவூட்டல் சேர்க்கவும் (${c.feeding.green} கிலோ பச்சைத் தீவனம், ${c.feeding.dry} கிலோ உலர் தீவனம்). முதலில் உலர் தீவனம் கொடுத்து, வாரத்திற்கு 2 ஓய்வு நாட்கள் அளிக்கவும்.`,
      },
      draught: {
        title: "உழவு திறன்",
        body: () =>
          "நன்கு பொருந்தும் நுகத்தடி பயன்படுத்தவும், குளிர்ந்த நேரத்தில் உழவு செய்யவும், ஓய்வின் போது நிழலும் நீரும் வழங்கவும், நோய்/தடுப்பூசிக்குப் பின் 2–3 வாரம் கடின உழைப்பைத் தவிர்க்கவும்.",
      },
      breeding_value: {
        title: "இனப்பெருக்க மதிப்பு",
        body: () =>
          "தாய், தந்தை, வளர்ச்சி, குணம் மற்றும் உழைப்பு பதிவுகளை வைத்திருக்கவும். சான்றளிக்கப்பட்ட செயற்கை விந்து காளையைப் பயன்படுத்தவும்; பரம்பரை குறைகள் உள்ள காளையைத் தவிர்க்கவும்.",
      },
      bull_health: {
        title: "காளை ஆரோக்கியம்",
        body: () =>
          "6 மாதத்திற்கு ஒருமுறை கிருமி நீக்கம் செய்யவும், உள்ளூர் அட்டவணைப்படி FMD மற்றும் HS/BQ தடுப்பூசி போடவும், மாதம் ஒருமுறை குளம்புகளைப் பரிசோதிக்கவும், பருவமழைக்குப் பின் உண்ணிகளைக் கட்டுப்படுத்தவும்.",
      },
      colostrum: {
        title: "முதல் மணி நேர சீம்பால்",
        body: () =>
          "பிறந்த முதல் மணி நேரத்தில் சீம்பால் கொடுக்கவும்; முதல் 3 நாட்களும் தொடரவும் — உடல் எடையின் 10% வரை தினமும். சீம்பால் கன்றின் முதல் நோய் பாதுகாப்பு.",
      },
      starter: {
        title: "கன்று தொடக்கத் தீவனம்",
        body: () =>
          "1–2 வார வயதில் சுத்தமான நீர் மற்றும் நல்ல வைக்கோலுடன் கன்று தொடக்கத் தீவனத்தை அறிமுகப்படுத்தவும். தினமும் 1–1.5 கிலோ தொடக்கத் தீவனம் சாப்பிடும் போது 8–12 வாரத்தில் பால் மறிக்கவும்.",
      },
      deworming: {
        title: "கிருமி நீக்கம்",
        body: () =>
          "2–3 வார வயதில் கிருமி நீக்கம் செய்து, 6 மாதம் வரை 4–6 வாரத்திற்கு ஒருமுறை மீண்டும் செய்யவும்; பின்னர் 3–6 மாதத்திற்கு ஒருமுறை. மருத்துவர் பரிந்துரைத்த அளவைப் பயன்படுத்தவும்.",
      },
      vaccination: {
        title: "தடுப்பூசி அட்டவணை",
        body: () =>
          "FMD: 4–6 மாதத்தில் முதல் டோஸ், 1 மாதம் கழித்து பூஸ்டர், பின்னர் 6 மாதத்திற்கு ஒருமுறை. புருசெல்லோசிஸ்: பெண் கன்றுகளுக்கு 4–8 மாதத்தில் ஒருமுறை. உள்ளூர் கால்நடை அட்டவணையைப் பின்பற்றவும்.",
      },
      calf_housing: {
        title: "கன்று கொட்டகை",
        body: () =>
          "கன்றுகளுக்கு ஒன்றுக்கு 1.5–2 சதுர மீட்டர் இடம், சுத்தமான படுக்கை, சுத்தமான நீர் மற்றும் காற்றோட்டமான உலர் கொட்டகை வழங்கவும். நோய்வாய்ப்பட்ட கன்றுகளை உடனே தனிமைப்படுத்தவும்.",
      },
      growth_diet: {
        title: "வளர்ச்சி தீவனம்",
        body: (c) =>
          `தினமும் ${c.feeding.green} கிலோ பச்சைத் தீவனம், ${c.feeding.dry} கிலோ உலர் தீவனம், ${c.feeding.concentrate} கிலோ செறிவூட்டல் மற்றும் ${c.feeding.mineral} கிராம் தாது கலவை கொடுக்கவும்.`,
      },
      puberty: {
        title: "முதல் இனப்பெருக்கம்",
        body: () =>
          "பெண்: 18–24 மாத வயது மற்றும் 250–300 கிலோ எடையில் முதல் செயற்கை விந்து ஊட்டத்தைத் திட்டமிடவும். ஆண்: இனப்பெருக்கத்திற்கு முன் வளர்ச்சி, குளம்பு மற்றும் குணத்தை மதிப்பிடவும்.",
      },
      young_health: {
        title: "வழக்கமான ஆரோக்கியம்",
        body: () =>
          "3–6 மாதத்திற்கு ஒருமுறை கிருமி நீக்கம் செய்யவும். உள்ளூர் அட்டவணைப்படி FMD, HS, BQ தடுப்பூசி போடவும். உண்ணிகள், நொண்டி மற்றும் மோசமான வளர்ச்சியை மாதம் ஒருமுறை பரிசோதிக்கவும்.",
      },
      general_feed: {
        title: "சமநிலை தீவனம்",
        body: (c) =>
          `தினமும் ${c.feeding.green} கிலோ பச்சைத் தீவனம், ${c.feeding.dry} கிலோ உலர் தீவனம், ${c.feeding.concentrate} கிலோ செறிவூட்டல் மற்றும் ${c.feeding.mineral} கிராம் தாது கலவை வழங்கவும்.`,
      },
      general_health: {
        title: "வழக்கமான ஆரோக்கியம்",
        body: () =>
          "சுத்தமான நீர், நிழல் மற்றும் உலர் தளம் வழங்கவும். 6 மாதத்திற்கு ஒருமுறை கிருமி நீக்கம், FMD மற்றும் HS/BQ தடுப்பூசி, மாதம் ஒருமுறை குளம்பு மற்றும் உண்ணி பரிசோதனை செய்யவும்.",
      },
    },
    vaccination: {
      title: "தடுப்பூசி மற்றும் கிருமி நீக்க அட்டவணை",
      items: [
        { age: "0–3 நாட்கள்", item: "சீம்பால் — உடல் எடையின் 10% தினமும்", note: "முதல் பாதுகாப்பு" },
        { age: "2–3 வாரங்கள்", item: "முதல் கிருமி நீக்கம்", note: "4–6 வாரத்திற்கு ஒருமுறை" },
        { age: "4–6 மாதங்கள்", item: "FMD முதல் டோஸ்", note: "1 மாதம் கழித்து பூஸ்டர்" },
        { age: "4–8 மாதங்கள்", item: "புருசெல்லோசிஸ் (பெண் கன்றுகள்)", note: "ஒருமுறை மட்டும்" },
        { age: "6 மாதங்கள்", item: "HS மற்றும் BQ தடுப்பூசி", note: "உள்ளூர் அட்டவணைப்படி" },
      ],
    },
    schemes: {
      title: "நீங்கள் பயன்படுத்தக்கூடிய திட்டங்கள்",
      note: "தற்போதைய தகுதி மற்றும் மானிய விகிதங்களுக்கு உள்ளூர் கால்நடை அலுவலகத்தை அணுகவும்.",
    },
    disclaimer:
      "இது பொதுவான வழிகாட்டல்; கால்நடை மருத்துவ ஆலோசனை அல்ல. தீவனம், விலை மற்றும் தடுப்பூசி நேரத்தை உங்கள் கால்நடை மருத்துவரிடம் உறுதிசெய்யவும்.",
  },
  hi: {
    labels: {
      breed: "नस्ल",
      origin: "मूल क्षेत्र",
      utility: "उपयोग",
      milk: "दूध उत्पादन",
      fat: "वसा",
      market: "सामान्य बाजार मूल्य",
      markers: "पहचान के लक्षण",
      schemes: "उपयोगी योजनाएँ",
      context: "सलाह का संदर्भ",
      gender: "लिंग",
      age: "आयु वर्ग",
      milkVisible: "दूध उत्पादन दिखाया गया",
      draughtVisible: "भारवाही क्षमता दिखाई गई",
      source: "स्रोत",
      disclaimer: "यह सामान्य मार्गदर्शन है, पशु चिकित्सकीय निदान नहीं। चारा, कीमत और टीकाकरण समय की पुष्टि अपने पशु चिकित्सक या विस्तार अधिकारी से करें।",
    },
    gender: { Female: "मादा", Male: "नर", Unknown: "उपलब्ध नहीं" },
    age: { Calf: "बछड़ा (1 वर्ष से कम)", Young: "युवा (1–3 वर्ष)", Adult: "वयस्क (3 वर्ष से अधिक)" },
    titles: {
      Female_Adult: "दुग्ध उत्पादन पोषण और थन देखभाल",
      Male_Adult: "भारवाही क्षमता और प्रजनन मूल्य",
      Calf: "बछड़े की वृद्धि और टीकाकरण",
      Young: "वृद्धि और पहला प्रजनन",
      General_Adult: "सामान्य देखभाल और स्वास्थ्य",
    },
    summaries: {
      Female_Adult: (c) =>
        `${c.breedName} ${c.utility} नस्ल है। यह कार्ड दुग्ध पोषण, दैनिक दूध, कैल्शियम और थनशोथ से बचाव पर केंद्रित है।`,
      Male_Adult: (c) =>
        `${c.breedName} ${c.utility} नस्ल है। यह कार्ड कार्य पोषण, प्रजनन मूल्य और सांड स्वास्थ्य पर केंद्रित है; दूध उत्पादन छिपाया गया है।`,
      Calf: (c) =>
        `${c.breedName} बछड़ा देखभाल: कोलोस्ट्रम, स्टार्टर फीड, कृमिनाशन और FMD/ब्रुसेलोसिस टीकाकरण कार्यक्रम।`,
      Young: (c) =>
        `${c.breedName} बढ़ रहा है। संतुलित वृद्धि, पहले प्रजनन की आयु और नियमित कृमिनाशन/टीकाकरण पर ध्यान दें।`,
      General_Adult: (c) =>
        `${c.breedName} की सामान्य देखभाल: संतुलित चारा, स्वच्छ पानी, छाया, खुर देखभाल और नियमित टीकाकरण।`,
    },
    cards: {
      lactation_diet: {
        title: "दुग्ध उत्पादन आहार",
        body: (c) =>
          `प्रतिदिन ${c.feeding.green} किलो हरा चारा, ${c.feeding.dry} किलो सूखा चारा और ${c.feeding.concentrate} किलो दाना दें। दाने को 2–3 बार में बाँटें और ${c.feeding.water} लीटर स्वच्छ पानी दें।`,
      },
      milk_yield: {
        title: "दैनिक दूध और वसा",
        body: (c) =>
          `इस नस्ल का सामान्य उत्पादन ${c.milk.daily} और वसा ${c.milk.fat} है। वास्तविक उत्पादन ब्यात, चारे, पानी और दुहने के तरीके पर निर्भर करता है।`,
      },
      calcium: {
        title: "कैल्शियम और खनिज",
        body: (c) =>
          `प्रतिदिन ${c.feeding.mineral} ग्राम खनिज मिश्रण दें। अधिक दूध के समय कैल्शियम/फॉस्फोरस पूरक दें और खनिज मिश्रण सूखा रखें।`,
      },
      mastitis: {
        title: "थनशोथ से बचाव",
        body: () =>
          "दूहने से पहले और बाद में थन जाँचें। हाथ धोएँ, सूखी साफ बिछावन रखें, दूध के बाद थन डिप करें और थन पूरा खाली करें। गर्मी, दर्द, सख्त हिस्सा या थक्का दिखे तो उसी दिन पशु चिकित्सक से मिलें।",
      },
      breeding: {
        title: "गर्मी पहचान और प्रजनन",
        body: () =>
          "हर 18–21 दिन में गर्मी के लक्षण देखें। खड़ी गर्मी के 12–18 घंटे बाद कृत्रिम गर्भाधान कराएँ। 12–15 महीने का ब्यांत अंतराल रखें और रिकॉर्ड रखें।",
      },
      work_diet: {
        title: "कार्य आहार",
        body: (c) =>
          `कार्य दिवसों में सामान्य राशन (${c.feeding.green} किलो हरा चारा, ${c.feeding.dry} किलो सूखा चारा) के साथ 0.5–1 किलो दाना बढ़ाएँ। पहले रूक्ष चारा दें और सप्ताह में 2 दिन आराम दें।`,
      },
      draught: {
        title: "भारवाही क्षमता",
        body: () =>
          "अच्छी तरह फिट जुआ प्रयोग करें, ठंडे समय में काम कराएँ, हर आराम पर छाया और पानी दें, बीमारी या टीकाकरण के बाद 2–3 सप्ताह भारी काम से बचें।",
      },
      breeding_value: {
        title: "प्रजनन मूल्य",
        body: () =>
          "माँ, पिता, वृद्धि, स्वभाव और कार्य प्रदर्शन का रिकॉर्ड रखें। प्रमाणित कृत्रिम गर्भाधान सांड का उपयोग करें; ज्ञात आनुवंशिक दोष वाले सांड से बचें।",
      },
      bull_health: {
        title: "सांड स्वास्थ्य",
        body: () =>
          "हर 6 महीने में कृमिनाशन, स्थानीय कार्यक्रम के अनुसार FMD और HS/BQ टीका, हर महीने खुर जाँच और मानसून के बाद कीट नियंत्रण करें।",
      },
      colostrum: {
        title: "पहले घंटे में कोलोस्ट्रम",
        body: () =>
          "जन्म के पहले घंटे में कोलोस्ट्रम दें और पहले 3 दिन जारी रखें — शरीर के वजन का लगभग 10% प्रतिदिन। कोलोस्ट्रम बछड़े की पहली रोग सुरक्षा है।",
      },
      starter: {
        title: "बछड़ा स्टार्टर फीड",
        body: () =>
          "1–2 सप्ताह में स्वच्छ पानी और अच्छे भूसे के साथ स्टार्टर शुरू करें। जब बछड़ा प्रतिदिन 1–1.5 किलो स्टार्टर खाए, तब 8–12 सप्ताह में दूध छुड़ाएँ।",
      },
      deworming: {
        title: "कृमिनाशन",
        body: () =>
          "2–3 सप्ताह में कृमिनाशन करें और 6 महीने तक हर 4–6 सप्ताह में दोहराएँ; फिर हर 3–6 महीने में। पशु चिकित्सक की सलाह से खुराक दें।",
      },
      vaccination: {
        title: "टीकाकरण कार्यक्रम",
        body: () =>
          "FMD: 4–6 महीने में पहली खुराक, 1 महीने बाद बूस्टर, फिर हर 6 महीने में। ब्रुसेलोसिस: मादा बछड़ों को 4–8 महीने में एक बार। स्थानीय पशु चिकित्सा कार्यक्रम का पालन करें।",
      },
      calf_housing: {
        title: "बछड़ा आवास",
        body: () =>
          "प्रति बछड़ा 1.5–2 वर्ग मीटर सूखा, हवादार स्थान, साफ बिछावन और ताजा पानी रखें। बीमार बछड़े को तुरंत अलग करें।",
      },
      growth_diet: {
        title: "वृद्धि आहार",
        body: (c) =>
          `प्रतिदिन ${c.feeding.green} किलो हरा चारा, ${c.feeding.dry} किलो सूखा चारा, ${c.feeding.concentrate} किलो दाना और ${c.feeding.mineral} ग्राम खनिज मिश्रण दें।`,
      },
      puberty: {
        title: "पहला प्रजनन",
        body: () =>
          "मादा: 18–24 महीने और 250–300 किलो वजन पर पहला कृत्रिम गर्भाधान योजना बनाएँ। नर: प्रजनन से पहले वृद्धि, खुर और स्वभाव का मूल्यांकन करें।",
      },
      young_health: {
        title: "नियमित स्वास्थ्य",
        body: () =>
          "हर 3–6 महीने में कृमिनाशन। स्थानीय कार्यक्रम के अनुसार FMD, HS, BQ टीका। हर महीने कीट, लंगड़ापन और कम वृद्धि की जाँच करें।",
      },
      general_feed: {
        title: "संतुलित चारा",
        body: (c) =>
          `प्रतिदिन ${c.feeding.green} किलो हरा चारा, ${c.feeding.dry} किलो सूखा चारा, ${c.feeding.concentrate} किलो दाना और ${c.feeding.mineral} ग्राम खनिज मिश्रण दें।`,
      },
      general_health: {
        title: "नियमित स्वास्थ्य",
        body: () =>
          "स्वच्छ पानी, छाया और सूखा फर्श दें। हर 6 महीने में कृमिनाशन, FMD और HS/BQ टीका, हर महीने खुर और कीट जाँच करें।",
      },
    },
    vaccination: {
      title: "टीकाकरण और कृमिनाशन समय-सारणी",
      items: [
        { age: "0–3 दिन", item: "कोलोस्ट्रम — शरीर के वजन का 10% प्रतिदिन", note: "पहली सुरक्षा" },
        { age: "2–3 सप्ताह", item: "पहला कृमिनाशन", note: "हर 4–6 सप्ताह में दोहराएँ" },
        { age: "4–6 महीने", item: "FMD पहली खुराक", note: "1 महीने बाद बूस्टर" },
        { age: "4–8 महीने", item: "ब्रुसेलोसिस (मादा बछड़े)", note: "एक बार" },
        { age: "6 महीने", item: "HS और BQ टीका", note: "स्थानीय कार्यक्रम के अनुसार" },
      ],
    },
    schemes: {
      title: "योजनाएँ जिनका लाभ ले सकते हैं",
      note: "वर्तमान पात्रता और सब्सिडी दर के लिए अपने स्थानीय पशु चिकित्सा कार्यालय से संपर्क करें।",
    },
    disclaimer:
      "यह सामान्य मार्गदर्शन है, पशु चिकित्सकीय निदान नहीं। चारा, कीमत और टीकाकरण समय की पुष्टि अपने पशु चिकित्सक या विस्तार अधिकारी से करें।",
  },
};

function resolveLang(value) {
  const lang = String(value || "en").toLowerCase().slice(0, 2);
  return LANGS.includes(lang) ? lang : "en";
}

function formatPrice(entry, lang) {
  const range = entry.price_range_inr;
  if (Array.isArray(range) && range.length === 2 && range.every(Number.isFinite)) {
    const locale = lang === "ta" ? "ta-IN" : lang === "hi" ? "hi-IN" : "en-IN";
    const fmt = new Intl.NumberFormat(locale, { maximumFractionDigits: 0 });
    return `₹${fmt.format(range[0])} – ₹${fmt.format(range[1])}`;
  }
  return entry.market_price || NOT_AVAILABLE[lang];
}

function orNA(value, lang) {
  return value || NOT_AVAILABLE[lang];
}

function localizedSchemes(codes, lang) {
  const copy = SCHEMES[lang] || SCHEMES.en;
  return (codes || []).map((code) => ({
    code,
    ...(copy[code] || SCHEMES.en[code] || {
      name: code,
      description: "",
    }),
  }));
}

function contextFor(entry, lang, gender, age) {
  const copy = COPY[lang] || COPY.en;
  const name = entry.local_names?.[lang] || entry.name;
  const utility = UTILITY[lang]?.[entry.utility] || entry.utility;
  const feeding = {
    green: orNA(entry.feeding?.green_fodder_kg_per_day, lang),
    dry: orNA(entry.feeding?.dry_fodder_kg_per_day, lang),
    concentrate: orNA(entry.feeding?.concentrate_kg_per_day, lang),
    mineral: orNA(entry.feeding?.mineral_mixture_g_per_day, lang),
    water: orNA(entry.feeding?.water_liters_per_day, lang),
  };
  const milk = {
    daily: orNA(entry.milk_yield?.daily_liters, lang),
    fat: orNA(entry.milk_yield?.fat_percentage, lang),
    lactation: orNA(entry.milk_yield?.lactation_yield, lang),
  };
  return {
    breedName: name,
    utility,
    feeding,
    milk,
    marketPrice: formatPrice(entry, lang),
    gender,
    age,
    copy,
  };
}

function card(copy, id, context) {
  const definition = copy.cards[id];
  if (!definition) return null;
  return {
    id,
    title: definition.title,
    body: definition.body(context),
  };
}

function buildAdvisory({ entry, gender, age, lang }) {
  const copy = COPY[lang] || COPY.en;
  const context = contextFor(entry, lang, gender, age);
  const cards = [];
  let milkYieldVisible = false;
  let draughtPowerVisible = false;
  let titleKey;

  if (age === "Calf") {
    titleKey = "Calf";
    cards.push(card(copy, "colostrum", context));
    cards.push(card(copy, "starter", context));
    cards.push(card(copy, "deworming", context));
    cards.push(card(copy, "vaccination", context));
    cards.push(card(copy, "calf_housing", context));
  } else if (age === "Young") {
    titleKey = "Young";
    cards.push(card(copy, "growth_diet", context));
    cards.push(card(copy, "puberty", context));
    cards.push(card(copy, "deworming", context));
    cards.push(card(copy, "young_health", context));
  } else if (gender === "Female") {
    titleKey = "Female_Adult";
    milkYieldVisible = Boolean(entry.milk_yield);
    cards.push(card(copy, "lactation_diet", context));
    if (milkYieldVisible) cards.push(card(copy, "milk_yield", context));
    cards.push(card(copy, "calcium", context));
    cards.push(card(copy, "mastitis", context));
    cards.push(card(copy, "breeding", context));
  } else if (gender === "Male") {
    titleKey = "Male_Adult";
    draughtPowerVisible = true;
    cards.push(card(copy, "work_diet", context));
    cards.push(card(copy, "draught", context));
    cards.push(card(copy, "breeding_value", context));
    cards.push(card(copy, "bull_health", context));
  } else {
    titleKey = "General_Adult";
    cards.push(card(copy, "general_feed", context));
    cards.push(card(copy, "general_health", context));
  }

  const vaccination =
    age === "Calf" || age === "Young"
      ? {
          title: copy.vaccination.title,
          items: copy.vaccination.items,
        }
      : null;

  return {
    title: copy.titles[titleKey],
    summary: copy.summaries[titleKey](context),
    context: {
      gender: copy.gender[gender] || copy.gender.Unknown,
      age: copy.age[age] || copy.age.Adult,
      milk_yield_visible: milkYieldVisible,
      draught_power_visible: draughtPowerVisible,
    },
    cards: cards.filter(Boolean),
    ...(vaccination ? { vaccination_schedule: vaccination } : {}),
    schemes: {
      title: copy.schemes.title,
      note: copy.schemes.note,
      items: localizedSchemes(entry.scheme_codes, lang),
    },
    disclaimer: copy.disclaimer,
  };
}

function breedFacts(entry, lang) {
  const copy = COPY[lang] || COPY.en;
  return {
    name: entry.local_names?.[lang] || entry.name,
    local_names: entry.local_names,
    species: entry.species,
    origin: entry.origin,
    utility: UTILITY[lang]?.[entry.utility] || entry.utility,
    utility_type: UTILITY[lang]?.[entry.utility] || entry.utility,
    milk_yield: entry.milk_yield
      ? {
          daily_liters: entry.milk_yield.daily_liters,
          lactation_yield: entry.milk_yield.lactation_yield,
          fat_percentage: entry.milk_yield.fat_percentage,
        }
      : null,
    physical_markers: entry.physical_markers,
    feeding: entry.feeding,
    climate: entry.climate,
    market_price: formatPrice(entry, lang),
    price_range_inr: entry.price_range_inr,
    schemes: localizedSchemes(entry.scheme_codes, lang),
    labels: copy.labels,
  };
}

module.exports = {
  LANGS,
  COPY,
  UTILITY,
  SCHEMES,
  resolveLang,
  buildAdvisory,
  breedFacts,
  formatPrice,
};
