import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import tempfile
import os
from PIL import Image
from keras_cv.layers import RandomCutout
# Page Configuration
st.set_page_config(
    page_title="Bird Sound Classifier",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful interface
st.markdown("""
    <style>
    :root {
        --primary: #4CAF50;
        --secondary: #2C3E50;
        --accent: #E8F4F8;
    }
    .main {
        background-color: #F8F9FA;
        padding: 2rem;
    }
    .result-card {
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        background: white;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-left: 6px solid var(--primary);
        transition: all 0.3s ease;
    }
    .result-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 24px rgba(0,0,0,0.12);
    }
    .bird-name {
        color: var(--secondary);
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .bird-sci-name {
        color: #7F8C8D;
        font-style: italic;
        margin-bottom: 1rem;
    }
    .confidence-meter {
        height: 10px;
        background: #E0E0E0;
        border-radius: 5px;
        margin: 1rem 0;
        overflow: hidden;
    }
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--primary), #8BC34A);
        width: 0%;
        transition: width 1s ease;
    }
    .spectrogram-container {
        background: var(--accent);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1.5rem 0;
    }
    .stButton>button {
        background: var(--primary);
        color: white;
        border-radius: 8px;
        padding: 0.8rem 2rem;
        font-weight: 600;
        border: none;
        font-size: 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: #45a049;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .top-prediction {
        background: rgba(76, 175, 80, 0.1);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Audio Preprocessing Function
def audio_to_melspectrogram(audio_path, sr=22050, n_fft=2048, hop_length=512, n_mels=128, f_min=20, f_max=16000, duration=5, img_size=256):
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=sr, duration=duration)
        
        # Compute mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, fmax=sr//2
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to 0-1 range
        mel_spec_norm = 255*(mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        
        mel_spec_norm =  mel_spec_norm.astype(np.float32)
        mel_image = Image.fromarray(mel_spec_norm)
        mel_image = mel_image.resize((img_size, img_size), Image.LANCZOS)
    
        # Convert to 3-channel image
        mel_image = np.stack([mel_image] * 3, axis=-1)
        return mel_image
    
    except Exception as e:
        st.error(f"Audio processing error: {str(e)}")
        return None

# Load Model and Bird Database
@st.cache_resource
def load_resources():
    try:
        # Load your trained model
        model = tf.keras.models.load_model('model_checkpoint_epochft_05.keras',custom_objects={"RandomCutout": RandomCutout})
    except:
        model = None
        st.error("Could not load the model file")
    
    # Complete bird database with all 150 species
    bird_classes = [
    "asbfly", "ashdro1", "ashpri1", "ashwoo2", "asikoe2", "asiope1", "aspfly1", "aspswi1",
    "barfly1", "barswa", "bcnher", "bkcbul1", "bkrfla1", "bkskit1", "bkwsti", "bladro1",
    "blaeag1", "blakit1", "blhori1", "blnmon1", "blrwar1", "bncwoo3", "brakit1", "brasta1",
    "brcful1", "brfowl1", "brnhao1", "brnshr", "brodro1", "brwjac1", "brwowl1", "btbeat1",
    "bwfshr1", "categr", "chbeat1", "cohcuc1", "comfla1", "comgre", "comior1", "comkin1",
    "commoo3", "commyn", "compea", "comros", "comsan", "comtai1", "copbar1", "crbsun2",
    "cregos1", "crfbar1", "crseag1", "dafbab1", "darter2", "eaywag1", "emedov2", "eucdov",
    "eurbla2", "eurcoo", "forwag1", "gargan", "gloibi", "goflea1", "graher1", "grbeat1",
    "grecou1", "greegr", "grefla1", "grehor1", "grejun2", "grenig1", "grewar3", "grnsan",
    "grnwar1", "grtdro1", "gryfra", "grynig2", "grywag", "gybpri1", "gyhcaf1", "heswoo1",
    "hoopoe", "houcro1", "houspa", "inbrob1", "indpit1", "indrob1", "indrol2", "indtit1",
    "ingori1", "inpher1", "insbab1", "insowl1", "integr", "isbduc1", "jerbus2", "junbab2",
    "junmyn1", "junowl1", "kenplo1", "kerlau2", "labcro1", "laudov1", "lblwar1", "lesyel1",
    "lewduc1", "lirplo", "litegr", "litgre1", "litspi1", "litswi1", "lobsun2", "maghor2",
    "malpar1", "maltro1", "malwoo1", "marsan", "mawthr1", "moipig1", "nilfly2", "niwpig1",
    "nutman", "orihob2", "oripip1", "pabflo1", "paisto1", "piebus1", "piekin1", "placuc3",
    "plaflo1", "plapri1", "plhpar1", "pomgrp2", "purher1", "pursun3", "pursun4", "purswa3",
    "putbab1", "redspu1", "rerswa1", "revbul", "rewbul", "rewlap1", "rocpig", "rorpar",
    "rossta2", "rufbab3", "ruftre2", "rufwoo2", "rutfly6", "sbeowl1", "scamin3", "shikra1",
    "smamin1", "sohmyn1", "spepic1", "spodov", "spoowl1", "sqtbul1", "stbkin1", "sttwoo1",
    "thbwar1", "tibfly3", "tilwar1", "vefnut1", "vehpar1", "wbbfly1", "wemhar1", "whbbul2",
    "whbsho3", "whbtre1", "whbwag1", "whbwat1", "whbwoo2", "whcbar1", "whiter2", "whrmun",
    "whtkin2", "woosan", "wynlau1", "yebbab1", "yebbul3", "zitcis1"
]
    bird_d = {'asbfly': 'Lesser Whistling-Duck',
 'ashdro1': 'Garganey',
 'ashpri1': 'Indian Spot-billed Duck',
 'ashwoo2': 'Indian Peafowl',
 'asikoe2': 'Red Spurfowl',
 'asiope1': 'Gray Junglefowl',
 'aspfly1': 'Gray Francolin',
 'aspswi1': 'Little Grebe',
 'barfly1': 'Rock Pigeon',
 'barswa': 'Nilgiri Wood-Pigeon',
 'bcnher': 'Eurasian Collared-Dove',
 'bkcbul1': 'Spotted Dove',
 'bkrfla1': 'Laughing Dove',
 'bkskit1': 'Asian Emerald Dove',
 'bkwsti': 'Gray-fronted Green-Pigeon',
 'bladro1': 'Mountain Imperial-Pigeon',
 'blaeag1': 'Greater Coucal',
 'blakit1': 'Asian Koel',
 'blhori1': 'Gray-bellied Cuckoo',
 'blnmon1': 'Common Hawk-Cuckoo',
 'blrwar1': 'Great Eared-Nightjar',
 'bncwoo3': 'Jungle Nightjar',
 'brakit1': 'Little Swift',
 'brasta1': 'Asian Palm-Swift',
 'brcful1': 'Eurasian Moorhen',
 'brfowl1': 'Eurasian Coot',
 'brnhao1': 'Gray-headed Swamphen',
 'brnshr': 'White-breasted Waterhen',
 'brodro1': 'Black-winged Stilt',
 'brwjac1': 'Red-wattled Lapwing',
 'brwowl1': 'Kentish Plover',
 'btbeat1': 'Little Ringed Plover',
 'bwfshr1': 'Bronze-winged Jacana',
 'categr': 'Common Sandpiper',
 'chbeat1': 'Green Sandpiper',
 'cohcuc1': 'Common Greenshank',
 'comfla1': 'Marsh Sandpiper',
 'comgre': 'Wood Sandpiper',
 'comior1': 'Whiskered Tern',
 'comkin1': 'Asian Openbill',
 'commoo3': 'Painted Stork',
 'commyn': 'Oriental Darter',
 'compea': 'Gray Heron',
 'comros': 'Purple Heron',
 'comsan': 'Great Egret',
 'comtai1': 'Intermediate Egret',
 'copbar1': 'Little Egret',
 'crbsun2': 'Cattle Egret',
 'cregos1': 'Indian Pond-Heron',
 'crfbar1': 'Black-crowned Night-Heron',
 'crseag1': 'Glossy Ibis',
 'dafbab1': 'Black-winged Kite',
 'darter2': 'Oriental Honey-buzzard',
 'eaywag1': 'Crested Serpent-Eagle',
 'emedov2': 'Black Eagle',
 'eucdov': 'Eurasian Marsh-Harrier',
 'eurbla2': 'Crested Goshawk',
 'eurcoo': 'Shikra',
 'forwag1': 'Black Kite',
 'gargan': 'Brahminy Kite',
 'gloibi': 'Indian Scops-Owl',
 'goflea1': 'Spot-bellied Eagle-Owl',
 'graher1': 'Brown Fish-Owl',
 'grbeat1': 'Jungle Owlet',
 'grecou1': 'Spotted Owlet',
 'greegr': 'Brown Wood-Owl',
 'grefla1': 'Brown Boobook',
 'grehor1': 'Malabar Trogon',
 'grejun2': 'Eurasian Hoopoe',
 'grenig1': 'Great Hornbill',
 'grewar3': 'Malabar Gray Hornbill',
 'grnsan': 'Common Kingfisher',
 'grnwar1': 'Stork-billed Kingfisher',
 'grtdro1': 'White-throated Kingfisher',
 'gryfra': 'Pied Kingfisher',
 'grynig2': 'Green Bee-eater',
 'grywag': 'Blue-tailed Bee-eater',
 'gybpri1': 'Chestnut-headed Bee-eater',
 'gyhcaf1': 'Indian Roller',
 'heswoo1': 'Malabar Barbet',
 'hoopoe': 'Coppersmith Barbet',
 'houcro1': 'White-cheeked Barbet',
 'houspa': 'Speckled Piculet',
 'inbrob1': 'Heart-spotted Woodpecker',
 'indpit1': 'Brown-capped Pygmy Woodpecker',
 'indrob1': 'Greater Flameback',
 'indrol2': 'Rufous Woodpecker',
 'indtit1': 'Common Flameback',
 'ingori1': 'Black-rumped Flameback',
 'inpher1': 'Lesser Yellownape',
 'insbab1': 'Streak-throated Woodpecker',
 'insowl1': 'White-bellied Woodpecker',
 'integr': 'Rose-ringed Parakeet',
 'isbduc1': 'Plum-headed Parakeet',
 'jerbus2': 'Malabar Parakeet',
 'junbab2': 'Vernal Hanging-Parrot',
 'junmyn1': 'Indian Pitta',
 'junowl1': 'Small Minivet',
 'kenplo1': 'Orange Minivet',
 'kerlau2': 'Indian Golden Oriole',
 'labcro1': 'Black-hooded Oriole',
 'laudov1': 'Ashy Woodswallow',
 'lblwar1': 'Malabar Woodshrike',
 'lesyel1': 'Bar-winged Flycatcher-shrike',
 'lewduc1': 'Common Iora',
 'lirplo': 'Black Drongo',
 'litegr': 'Ashy Drongo',
 'litgre1': 'Bronzed Drongo',
 'litspi1': 'Greater Racket-tailed Drongo',
 'litswi1': 'Black-naped Monarch',
 'lobsun2': 'Indian Paradise-Flycatcher',
 'maghor2': 'Brown Shrike',
 'malpar1': 'Rufous Treepie',
 'maltro1': 'White-bellied Treepie',
 'malwoo1': 'House Crow',
 'marsan': 'Large-billed Crow',
 'mawthr1': 'Gray-headed Canary-Flycatcher',
 'moipig1': 'Indian Yellow Tit',
 'nilfly2': "Jerdon's Bushlark",
 'niwpig1': 'Common Tailorbird',
 'nutman': 'Gray-breasted Prinia',
 'orihob2': 'Ashy Prinia',
 'oripip1': 'Plain Prinia',
 'pabflo1': 'Zitting Cisticola',
 'paisto1': 'Thick-billed Warbler',
 'piebus1': "Blyth's Reed Warbler",
 'piekin1': 'Barn Swallow',
 'placuc3': 'Red-rumped Swallow',
 'plaflo1': 'Flame-throated Bulbul',
 'plapri1': 'Red-vented Bulbul',
 'plhpar1': 'Red-whiskered Bulbul',
 'pomgrp2': 'White-browed Bulbul',
 'purher1': 'Yellow-browed Bulbul',
 'pursun3': 'Square-tailed Bulbul',
 'pursun4': "Tickell's Leaf Warbler",
 'purswa3': 'Green Warbler',
 'putbab1': 'Greenish Warbler',
 'redspu1': 'Large-billed Leaf Warbler',
 'rerswa1': 'Dark-fronted Babbler',
 'revbul': 'Indian Scimitar-Babbler',
 'rewbul': 'Puff-throated Babbler',
 'rewlap1': 'Brown-cheeked Fulvetta',
 'rocpig': 'Palani Laughingthrush',
 'rorpar': 'Rufous Babbler',
 'rossta2': 'Jungle Babbler',
 'rufbab3': 'Yellow-billed Babbler',
 'ruftre2': 'Wayanad Laughingthrush',
 'rufwoo2': 'Velvet-fronted Nuthatch',
 'rutfly6': 'Southern Hill Myna',
 'sbeowl1': 'Rosy Starling',
 'scamin3': 'Brahminy Starling',
 'shikra1': 'Common Myna',
 'smamin1': 'Jungle Myna',
 'sohmyn1': 'Indian Blackbird',
 'spepic1': 'Asian Brown Flycatcher',
 'spodov': 'Indian Robin',
 'spoowl1': 'White-bellied Sholakili',
 'sqtbul1': 'White-bellied Blue Flycatcher',
 'stbkin1': "Tickell's Blue Flycatcher",
 'sttwoo1': 'Nilgiri Flycatcher',
 'thbwar1': 'Indian Blue Robin',
 'tibfly3': 'Malabar Whistling-Thrush',
 'tilwar1': 'Black-and-orange Flycatcher',
 'vefnut1': 'Rusty-tailed Flycatcher',
 'vehpar1': 'Pied Bushchat',
 'wbbfly1': 'Pale-billed Flowerpecker',
 'wemhar1': 'Nilgiri Flowerpecker',
 'whbbul2': 'Purple-rumped Sunbird',
 'whbsho3': 'Crimson-backed Sunbird',
 'whbtre1': 'Purple Sunbird',
 'whbwag1': "Loten's Sunbird",
 'whbwat1': 'Little Spiderhunter',
 'whbwoo2': 'Golden-fronted Leafbird',
 'whcbar1': 'Scaly-breasted Munia',
 'whiter2': 'White-rumped Munia',
 'whrmun': 'House Sparrow',
 'whtkin2': 'Forest Wagtail',
 'woosan': 'Gray Wagtail',
 'wynlau1': 'Western Yellow Wagtail',
 'yebbab1': 'White-browed Wagtail',
 'yebbul3': 'Paddyfield Pipit',
 'zitcis1': 'Common Rosefinch'}
    
    bird_info = {
    "Lesser Whistling-Duck": {
        "code": "leswhd",
        "sci_name": "Dendrocygna javanica",
        "info": "Nocturnal duck with distinctive whistling calls."
    },
    "Garganey": {
        "code": "gargan",
        "sci_name": "Spatula querquedula",
        "info": "Small dabbling duck that migrates long distances."
    },
    "Indian Spot-billed Duck": {
        "code": "indspbd",
        "sci_name": "Anas poecilorhyncha",
        "info": "Common freshwater duck with distinctive yellow-tipped bill."
    },
    "Indian Peafowl": {
        "code": "indpeaf",
        "sci_name": "Pavo cristatus",
        "info": "National bird of India, known for its iridescent plumage and elaborate courtship display."
    },
    "Red Spurfowl": {
        "code": "redspfl",
        "sci_name": "Galloperdix spadicea",
        "info": "Ground-dwelling bird with reddish plumage and spurred legs."
    },
    "Gray Junglefowl": {
        "code": "gryjngf",
        "sci_name": "Gallus sonneratii",
        "info": "Wild ancestor of domestic chickens, endemic to southern India."
    },
    "Gray Francolin": {
        "code": "gryfrcl",
        "sci_name": "Francolinus pondicerianus",
        "info": "Common game bird with a loud, distinctive call."
    },
    "Little Grebe": {
        "code": "litgreb",
        "sci_name": "Tachybaptus ruficollis",
        "info": "Small water bird known for its diving ability."
    },
    "Rock Pigeon": {
        "code": "rocpgon",
        "sci_name": "Columba livia",
        "info": "Ubiquitous urban bird, often seen in cities worldwide."
    },
    "Nilgiri Wood-Pigeon": {
        "code": "nilwpgn",
        "sci_name": "Columba elphinstonii",
        "info": "Large pigeon endemic to the Western Ghats of India."
    },
    "Eurasian Collared-Dove": {
        "code": "eurcldv",
        "sci_name": "Streptopelia decaocto",
        "info": "Pale dove with a black neck collar, common in urban areas."
    },
    "Spotted Dove": {
        "code": "spotdov",
        "sci_name": "Spilopelia chinensis",
        "info": "Dove with distinctive spotted neck patch."
    },
    "Laughing Dove": {
        "code": "lghdove",
        "sci_name": "Spilopelia senegalensis",
        "info": "Small, pinkish dove with a cheerful cooing call."
    },
    "Asian Emerald Dove": {
        "code": "asemrdv",
        "sci_name": "Chalcophaps indica",
        "info": "Bright green-winged dove found in forested areas."
    },
    "Gray-fronted Green-Pigeon": {
        "code": "gryfgpg",
        "sci_name": "Treron affinis",
        "info": "Colorful fruit-eating pigeon native to the Western Ghats."
    },
    "Mountain Imperial-Pigeon": {
        "code": "mtimpgn",
        "sci_name": "Ducula badia",
        "info": "Large forest pigeon with a deep booming call."
    },
    "Greater Coucal": {
        "code": "grtcouc",
        "sci_name": "Centropus sinensis",
        "info": "Large, crow-like bird with coppery wings and deep calls."
    },
    "Asian Koel": {
        "code": "asikoel",
        "sci_name": "Eudynamys scolopaceus",
        "info": "Famous for its melodious calls, often heard in the early morning."
    },
    "Gray-bellied Cuckoo": {
        "code": "grybcuc",
        "sci_name": "Cacomantis passerinus",
        "info": "Small cuckoo with a distinctive repetitive call."
    },
    "Common Hawk-Cuckoo": {
        "code": "comhkcu",
        "sci_name": "Hierococcyx varius",
        "info": "Also known as the 'brainfever bird' due to its loud calls."
    },
    "Great Eared-Nightjar": {
        "code": "gretnjr",
        "sci_name": "Lyncornis macrotis",
        "info": "Large nocturnal bird with prominent ear tufts."
    },
    "Jungle Nightjar": {
        "code": "jnglnjr",
        "sci_name": "Caprimulgus indicus",
        "info": "Camouflaged nocturnal bird found in wooded habitats."
    },
    "Little Swift": {
        "code": "litswft",
        "sci_name": "Apus affinis",
        "info": "Small, fast-flying bird often seen in flocks."
    },
    "Asian Palm-Swift": {
        "code": "aspmswf",
        "sci_name": "Cypsiurus balasiensis",
        "info": "Slender swift that nests in palm fronds."
    },
    "Eurasian Moorhen": {
        "code": "eurmoor",
        "sci_name": "Gallinula chloropus",
        "info": "Common waterbird with a red frontal shield."
    },
    "Eurasian Coot": {
        "code": "eurcoot",
        "sci_name": "Fulica atra",
        "info": "Black waterbird with a distinctive white beak."
    },
    "Gray-headed Swamphen": {
        "code": "gryhsph",
        "sci_name": "Porphyrio poliocephalus",
        "info": "Large purple-blue bird found in wetlands."
    },
    "White-breasted Waterhen": {
        "code": "whtbwhn",
        "sci_name": "Amaurornis phoenicurus",
        "info": "Bold waterbird often seen darting through vegetation."
    },
    "Black-winged Stilt": {
        "code": "blkwstl",
        "sci_name": "Himantopus himantopus",
        "info": "Wader with long pink legs and contrasting black-and-white plumage."
    },
    "Red-wattled Lapwing": {
        "code": "rdwtlap",
        "sci_name": "Vanellus indicus",
        "info": "Noisy bird known for its loud 'did-he-do-it' call."
    },
    "Kentish Plover": {
        "code": "kntplov",
        "sci_name": "Charadrius alexandrinus",
        "info": "Small shorebird often found on sandy beaches."
    },
    "Little Ringed Plover": {
        "code": "ltrplov",
        "sci_name": "Charadrius dubius",
        "info": "Tiny wader with a distinctive yellow eye-ring."
    },
    "Bronze-winged Jacana": {
        "code": "brwjaca",
        "sci_name": "Metopidius indicus",
        "info": "Wetland bird with long toes, allowing it to walk on floating vegetation."
    },
    "Common Sandpiper": {
        "code": "comsand",
        "sci_name": "Actitis hypoleucos",
        "info": "Small wader with a distinctive bobbing motion."
    },
    "Green Sandpiper": {
        "code": "grnsand",
        "sci_name": "Tringa ochropus",
        "info": "Solitary wader often found near freshwater bodies."
    },
    "Common Greenshank": {
        "code": "comgrsh",
        "sci_name": "Tringa nebularia",
        "info": "Tall wader with greenish legs and a slightly upturned bill."
    },
    "Marsh Sandpiper": {
        "code": "mrshsnd",
        "sci_name": "Tringa stagnatilis",
        "info": "Elegant wader with long legs and a slender bill."
    },
    "Wood Sandpiper": {
        "code": "wodsand",
        "sci_name": "Tringa glareola",
        "info": "Small wader with a speckled back and yellowish legs."
    },
    "Whiskered Tern": {
        "code": "whsktrn",
        "sci_name": "Chlidonias hybrida",
        "info": "Medium-sized tern with a black cap and white cheeks."
    },
    "Asian Openbill": {
        "code": "asopbil",
        "sci_name": "Anastomus oscitans",
        "info": "Stork with a distinctive gap between its mandibles."
    },
    "Painted Stork": {
        "code": "pntstork",
        "sci_name": "Mycteria leucocephala",
        "info": "Large stork with striking pink tertial feathers."
    },
    "Oriental Darter": {
        "code": "oridart",
        "sci_name": "Anhinga melanogaster",
        "info": "Snake-like bird that swims with only its neck above water."
    },
    "Gray Heron": {
        "code": "gryheron",
        "sci_name": "Ardea cinerea",
        "info": "Tall, long-necked heron commonly found near water bodies."
    },
    "Purple Heron": {
        "code": "prplhern",
        "sci_name": "Ardea purpurea",
        "info": "Slender heron with reddish-brown plumage."
    },
    "Great Egret": {
        "code": "grtegret",
        "sci_name": "Ardea alba",
        "info": "Large white egret with a yellow bill and black legs."
    },
    "Intermediate Egret": {
        "code": "intregt",
        "sci_name": "Ardea intermedia",
        "info": "Medium-sized egret with yellow bill and black legs."
    },
    "Little Egret": {
        "code": "litegrt",
        "sci_name": "Egretta garzetta",
        "info": "Small white egret with black bill and yellow feet."
    },
    "Cattle Egret": {
        "code": "cattegt",
        "sci_name": "Bubulcus ibis",
        "info": "Often seen near livestock, feeding on insects stirred up by animals."
    },
    "Indian Pond-Heron": {
        "code": "indphrn",
        "sci_name": "Ardeola grayii",
        "info": "Stocky heron with a distinctive breeding plumage."
    },
    "Black-crowned Night-Heron": {
        "code": "blcnnhn",
        "sci_name": "Nycticorax nycticorax",
        "info": "Nocturnal heron with a black crown and back."
    },
    "Glossy Ibis": {
        "code": "glosibs",
        "sci_name": "Plegadis falcinellus",
        "info": "Wader with iridescent dark plumage and down-curved bill."
    },
    "Black-winged Kite": {
        "code": "blwikit",
        "sci_name": "Elanus caeruleus",
        "info": "Small raptor with striking red eyes and hovering flight."
    },
    "Oriental Honey-buzzard": {
        "code": "orhnbzd",
        "sci_name": "Pernis ptilorhynchus",
        "info": "Specializes in feeding on honeycombs and larvae."
    },
    "Crested Serpent-Eagle": {
        "code": "crseagl",
        "sci_name": "Spilornis cheela",
        "info": "Medium-sized raptor with a distinctive call and crest."
    },
    "Black Eagle": {
        "code": "blkeagl",
        "sci_name": "Ictinaetus malaiensis",
        "info": "Large eagle known for its soaring flight over forests."
    },
    "Eurasian Marsh-Harrier": {
        "code": "eurmhar",
        "sci_name": "Circus aeruginosus",
        "info": "Prefers wetlands and is often seen gliding low over reed beds."
    },
    "Crested Goshawk": {
        "code": "crgoshk",
        "sci_name": "Accipiter trivirgatus",
        "info": "Forest-dwelling raptor with a prominent crest."
    },
    "Shikra": {
        "code": "shikrap",
        "sci_name": "Accipiter badius",
        "info": "Small hawk commonly found in urban and rural areas."
    },
    "Black Kite": {
        "code": "blkkite",
        "sci_name": "Milvus migrans",
        "info": "Scavenger often seen soaring over cities and towns."
    },
    "Brahminy Kite": {
        "code": "brhkite",
        "sci_name": "Haliastur indus",
        "info": "Raptor with a contrasting white head and chestnut body."
    },
    "Indian Scops-Owl": {
        "code": "indscow",
        "sci_name": "Otus bakkamoena",
        "info": "Small owl with excellent camouflage against tree bark."
    },
    "Spot-bellied Eagle-Owl": {
        "code": "spbeowl",
        "sci_name": "Bubo nipalensis",
        "info": "Large owl with distinctive spots on its belly."
    },
    "Brown Fish-Owl": {
        "code": "brfshow",
        "sci_name": "Ketupa zeylonensis",
        "info": "Prefers habitats near water bodies and feeds on fish."
    },
    "Jungle Owlet": {
        "code": "jnglowl",
        "sci_name": "Glaucidium radiatum",
        "info": "Small, diurnal owl found in forested areas."
    },
    "Spotted Owlet": {
        "code": "spotowl",
        "sci_name": "Athene brama",
        "info": "Commonly seen in urban areas, active during dusk and dawn."
    },
    "Brown Wood-Owl": {
        "code": "brwdowl",
        "sci_name": "Strix leptogrammica",
        "info": "Large owl with a deep hooting call, inhabits dense forests."
    },
    "Brown Boobook": {
        "code": "brboobk",
        "sci_name": "Ninox scutulata",
        "info": "Nocturnal owl with a distinctive two-note call."
    },
    "Malabar Trogon": {
        "code": "mlbtrog",
        "sci_name": "Harpactes fasciatus",
        "info": "Colorful bird with a preference for dense forests."
    },
    "Eurasian Hoopoe": {
        "code": "eurhoop",
        "sci_name": "Upupa epops",
        "info": "Recognizable by its crown of feathers and unique call."
    },
    "Great Hornbill": {
        "code": "grthorn",
        "sci_name": "Buceros bicornis",
        "info": "Large hornbill with a prominent casque on its bill."
    },
    "Malabar Gray Hornbill": {
        "code": "mlbghrn",
        "sci_name": "Ocyceros griseus",
        "info": "Endemic to the Western Ghats, prefers dense forests."
    },
    "Common Kingfisher": {
        "code": "comkngf",
        "sci_name": "Alcedo atthis",
        "info": "Small, brightly colored kingfisher often seen near water."
    },
    "Stork-billed Kingfisher": {
        "code": "stbkngf",
        "sci_name": "Pelargopsis capensis",
        "info": "Large kingfisher with a massive red bill."
    },
    "White-throated Kingfisher": {
        "code": "whtkngf",
        "sci_name": "Halcyon smyrnensis",
        "info": "Common kingfisher with a bright blue back and white throat."
    },
    "Pied Kingfisher": {
        "code": "piedkng",
        "sci_name": "Ceryle rudis",
        "info": "Black and white kingfisher known for its hovering flight."
    },
    "Green Bee-eater": {
        "code": "grnbeat",
        "sci_name": "Merops orientalis",
        "info": "Slender bird with bright green plumage and elongated tail feathers."
    },
    "Blue-tailed Bee-eater": {
        "code": "bltbeat",
        "sci_name": "Merops philippinus",
        "info": "Colorful bee-eater with a distinctive blue tail."
    },
    "Chestnut-headed Bee-eater": {
        "code": "chhbeat",
        "sci_name": "Merops leschenaulti",
        "info": "Bee-eater with a rich chestnut-colored head and green body."
    },
    "Indian Roller": {
        "code": "indroll",
        "sci_name": "Coracias benghalensis",
        "info": "Vibrant bird known for its acrobatic flight displays."
    },
    "Malabar Barbet": {
        "code": "mlbbarb",
        "sci_name": "Psilopogon malabaricus",
        "info": "Endemic to the Western Ghats, with a repetitive call."
    },
    "Coppersmith Barbet": {
        "code": "copbarb",
        "sci_name": "Psilopogon haemacephalus",
        "info": "Named for its metallic 'tuk-tuk' call resembling a coppersmith at work."
    },
    "White-cheeked Barbet": {
        "code": "wchbarb",
        "sci_name": "Psilopogon viridis",
        "info": "Green barbet with distinctive white cheek patches."
    },
    "Speckled Piculet": {
        "code": "spkpicu",
        "sci_name": "Picumnus innominatus",
        "info": "Tiny woodpecker with speckled plumage."
    },
    "Heart-spotted Woodpecker": {
        "code": "hspwood",
        "sci_name": "Hemicircus canente",
        "info": "Small woodpecker with heart-shaped spots on its back."
    },
    "Brown-capped Pygmy Woodpecker": {
        "code": "bcpygwp",
        "sci_name": "Yungipicus nanus",
        "info": "One of the smallest woodpeckers, with a brown cap."
    },
    "Greater Flameback": {
        "code": "grflmbk",
        "sci_name": "Chrysocolaptes guttacristatus",
        "info": "Large woodpecker with golden-yellow back and loud call."
    },
    "Rufous Woodpecker": {
        "code": "rufwood",
        "sci_name": "Micropternus brachyurus",
        "info": "Unique among woodpeckers for its termite mound nesting."
    },
    "Common Flameback": {
        "code": "comflmb",
        "sci_name": "Dinopium javanense",
        "info": "Medium-sized woodpecker with a striking golden back."
    },
    "Black-rumped Flameback": {
        "code": "brflmbk",
        "sci_name": "Dinopium benghalense",
        "info": "Widespread woodpecker with a black rump and golden back."
    },
    "Lesser Yellownape": {
        "code": "lsylnap",
        "sci_name": "Picus chlorolophus",
        "info": "Small woodpecker with a"
    },
    "Streak-throated Woodpecker": {
        "code": "stthwp",
        "sci_name": "Picus xanthopygaeus",
        "info": "Medium-sized woodpecker with streaked throat and greenish back."
    },
    "White-bellied Woodpecker": {
        "code": "wbwdpk",
        "sci_name": "Dryocopus javensis",
        "info": "Large woodpecker with striking white belly and loud calls."
    },
    "Rose-ringed Parakeet": {
        "code": "rrpar",
        "sci_name": "Psittacula krameri",
        "info": "Common green parakeet with a distinctive rose-colored neck ring."
    },
    "Plum-headed Parakeet": {
        "code": "phpar",
        "sci_name": "Psittacula cyanocephala",
        "info": "Colorful parakeet with a plum-colored head and green body."
    },
    "Malabar Parakeet": {
        "code": "mlbpar",
        "sci_name": "Psittacula columboides",
        "info": "Endemic to the Western Ghats, with a bluish-gray body and green wings."
    },
    "Vernal Hanging-Parrot": {
        "code": "vhpar",
        "sci_name": "Loriculus vernalis",
        "info": "Small green parrot that sleeps upside down like a bat."
    },
    "Indian Pitta": {
        "code": "indpitta",
        "sci_name": "Pitta brachyura",
        "info": "Colorful ground-dwelling bird known for its seven-colored plumage."
    },
    "Small Minivet": {
        "code": "smminv",
        "sci_name": "Pericrocotus cinnamomeus",
        "info": "Small passerine with bright orange-red underparts in males."
    },
    "Orange Minivet": {
        "code": "orngminv",
        "sci_name": "Pericrocotus flammeus",
        "info": "Striking bird with vivid orange and black plumage."
    },
    "Indian Golden Oriole": {
        "code": "indgor",
        "sci_name": "Oriolus kundoo",
        "info": "Bright yellow oriole with a melodious whistle."
    },
    "Black-hooded Oriole": {
        "code": "bhori",
        "sci_name": "Oriolus xanthornus",
        "info": "Yellow bird with a contrasting black head and throat."
    },
    "Ashy Woodswallow": {
        "code": "ashwsw",
        "sci_name": "Artamus fuscus",
        "info": "Sociable bird with ashy-gray plumage and forked tail."
    },
    "Malabar Woodshrike": {
        "code": "mlbws",
        "sci_name": "Tephrodornis sylvicola",
        "info": "Forest-dwelling bird with grayish upperparts and white underparts."
    },
    "Bar-winged Flycatcher-shrike": {
        "code": "bwfs",
        "sci_name": "Hemipus picatus",
        "info": "Small bird with distinctive white wing bars and active foraging behavior."
    },
    "Common Iora": {
        "code": "comiora",
        "sci_name": "Aegithina tiphia",
        "info": "Bright yellow bird with black upperparts in males during breeding season."
    },
    "Black Drongo": {
        "code": "blkdro",
        "sci_name": "Dicrurus macrocercus",
        "info": "Glossy black bird known for its aggressive behavior towards larger birds."
    },
    "Ashy Drongo": {
        "code": "ashdro",
        "sci_name": "Dicrurus leucophaeus",
        "info": "Grayish drongo with a deeply forked tail and agile flight."
    },
    "Bronzed Drongo": {
        "code": "brndro",
        "sci_name": "Dicrurus aeneus",
        "info": "Small drongo with metallic sheen and acrobatic flight."
    },
    "Greater Racket-tailed Drongo": {
        "code": "grrtdro",
        "sci_name": "Dicrurus paradiseus",
        "info": "Large drongo with elongated tail feathers ending in rackets."
    },
    "Black-naped Monarch": {
        "code": "bnmon",
        "sci_name": "Hypothymis azurea",
        "info": "Graceful flycatcher with a distinctive black nape and azure blue plumage."
    },
    "Indian Paradise-Flycatcher": {
        "code": "ipfly",
        "sci_name": "Terpsiphone paradisi",
        "info": "Elegant bird with long tail streamers and sexual dimorphism in plumage."
    },
    "Brown Shrike": {
        "code": "brnshr",
        "sci_name": "Lanius cristatus",
        "info": "Migratory shrike with a brown back and distinctive eye stripe."
    },
    "Rufous Treepie": {
        "code": "ruftreep",
        "sci_name": "Dendrocitta vagabunda",
        "info": "Noisy bird with a long tail and rufous plumage."
    },
    "White-bellied Treepie": {
        "code": "wbtreep",
        "sci_name": "Dendrocitta leucogastra",
        "info": "Endemic to the Western Ghats, with contrasting black and white plumage."
    },
    "House Crow": {
        "code": "hscrow",
        "sci_name": "Corvus splendens",
        "info": "Common urban bird with a gray neck and scavenging habits."
    },
    "Large-billed Crow": {
        "code": "lbcrow",
        "sci_name": "Corvus macrorhynchos",
        "info": "Robust crow with a large bill and all-black plumage."
    },
    "Gray-headed Canary-Flycatcher": {
        "code": "ghcfly",
        "sci_name": "Culicicapa ceylonensis",
        "info": "Small flycatcher with a gray head and yellow underparts."
    },
    "Indian Yellow Tit": {
        "code": "iytit",
        "sci_name": "Machlolophus aplonotus",
        "info": "Bright yellow tit with a distinctive black crest."
    },
    "Jerdon's Bushlark": {
        "code": "jbuslark",
        "sci_name": "Mirafra affinis",
        "info": "Ground-dwelling bird with a melodious song and streaked plumage."
    },
    "Common Tailorbird": {
        "code": "ctailb",
        "sci_name": "Orthotomus sutorius",
        "info": "Small warbler known for stitching leaves to build its nest."
    },
    "Gray-breasted Prinia": {
        "code": "gbprinia",
        "sci_name": "Prinia hodgsonii",
        "info": "Active warbler with a gray breast and distinctive call."
    },
    "Ashy Prinia": {
        "code": "ashprinia",
        "sci_name": "Prinia socialis",
        "info": "Small bird with ashy upperparts and a jerky tail movement."
    },
    "Plain Prinia": {
        "code": "plprinia",
        "sci_name": "Prinia inornata",
        "info": "Unstreaked warbler with a plain appearance and repetitive song."
    },
    "Zitting Cisticola": {
        "code": "zitcist",
        "sci_name": "Cisticola juncidis",
        "info": "Tiny bird with a distinctive 'zitting' call during flight displays."
    },
    "Thick-billed Warbler": {
        "code": "tbwarb",
        "sci_name": "Arundinax aedon",
        "info": "Large warbler with a robust bill and skulking behavior."
    },
    "Blyth's Reed Warbler": {
        "code": "brwarb",
        "sci_name": "Acrocephalus dumetorum",
        "info": "Migratory warbler with a rich, varied song."
    },
    "Barn Swallow": {
        "code": "barnsw",
        "sci_name": "Hirundo rustica",
        "info": "Graceful swallow with a deeply forked tail and agile flight."
    },
    "Red-rumped Swallow": {
        "code": "rrswallow",
        "sci_name": "Cecropis daurica",
        "info": "Swallow with a reddish rump and mud nest-building habits."
    },
    "Flame-throated Bulbul": {
        "code": "ftbulbul",
        "sci_name": "Rubigula gularis",
        "info": "Endemic bulbul with a fiery orange throat patch."
    },
    "Red-vented Bulbul": {
        "code": "rvbulbul",
        "sci_name": "Pycnonotus cafer",
        "info": "Common bulbul with a red vent and cheerful song."
    },
    "Red-whiskered Bulbul": {
        "code": "rwbulbul",
        "sci_name": "Pycnonotus jocosus",
        "info": "Bulbul with a prominent red patch below the eye and melodious calls."
    },
    "White-browed Bulbul": {
        "code": "wbbulbul",
        "sci_name": "Pycnonotus luteolus",
        "info": "Bulbul with a distinctive white brow and yellowish underparts."
    },
    "Yellow-browed Bulbul": {
        "code": "ybbulbul",
        "sci_name": "Acritillas indica",
        "info": "Bright yellow bulbul with a noticeable yellow brow."
    },
    "Square-tailed Bulbul": {
        "code": "stbulbul",
        "sci_name": "Hypsipetes ganeesa",
        "info": "Bulbul with a square-shaped tail and dark"
    },
     "Streak-throated Woodpecker": {
        "code": "stthwp",
        "sci_name": "Picus xanthopygaeus",
        "info": "Medium-sized green woodpecker with a streaked throat and scaly whitish underparts."
    },
    "White-bellied Woodpecker": {
        "code": "wbwdpk",
        "sci_name": "Dryocopus javensis",
        "info": "One of Asia's largest woodpeckers, known for its striking white belly and loud calls."
    },
    "Rose-ringed Parakeet": {
        "code": "rrpar",
        "sci_name": "Psittacula krameri",
        "info": "Medium-sized green parakeet with a distinctive rose-colored neck ring."
    },
    "Plum-headed Parakeet": {
        "code": "phpar",
        "sci_name": "Psittacula cyanocephala",
        "info": "Colorful parakeet with males having a pinkish-purple head and females a grey head."
    },
    "Malabar Parakeet": {
        "code": "mlbpar",
        "sci_name": "Psittacula columboides",
        "info": "Endemic to the Western Ghats, featuring bluish-grey plumage and a long, elegant tail."
    },
    "Vernal Hanging-Parrot": {
        "code": "vhpar",
        "sci_name": "Loriculus vernalis",
        "info": "Small green parrot known for sleeping upside down and vibrant red rump."
    },
    "Indian Pitta": {
        "code": "indpitta",
        "sci_name": "Pitta brachyura",
        "info": "Vibrant bird with a rainbow of colors, often found in dense undergrowth."
    },
    "Small Minivet": {
        "code": "smminv",
        "sci_name": "Pericrocotus cinnamomeus",
        "info": "Small passerine with bright orange-red underparts in males."
    },
    "Orange Minivet": {
        "code": "orngminv",
        "sci_name": "Pericrocotus flammeus",
        "info": "Striking bird with vivid orange and black plumage."
    },
    "Indian Golden Oriole": {
        "code": "indgor",
        "sci_name": "Oriolus kundoo",
        "info": "Bright yellow oriole with a melodious whistle."
    },
    "Black-hooded Oriole": {
        "code": "bhori",
        "sci_name": "Oriolus xanthornus",
        "info": "Yellow bird with a contrasting black head and throat."
    },
    "Ashy Woodswallow": {
        "code": "ashwsw",
        "sci_name": "Artamus fuscus",
        "info": "Sociable bird with ashy-gray plumage and forked tail."
    },
    "Malabar Woodshrike": {
        "code": "mlbws",
        "sci_name": "Tephrodornis sylvicola",
        "info": "Forest-dwelling bird with grayish upperparts and white underparts."
    },
    "Bar-winged Flycatcher-shrike": {
        "code": "bwfs",
        "sci_name": "Hemipus picatus",
        "info": "Small bird with distinctive white wing bars and active foraging behavior."
    },
    "Common Iora": {
        "code": "comiora",
        "sci_name": "Aegithina tiphia",
        "info": "Bright yellow bird with black upperparts in males during breeding season."
    },
    "Black Drongo": {
        "code": "blkdro",
        "sci_name": "Dicrurus macrocercus",
        "info": "Glossy black bird known for its aggressive behavior towards larger birds."
    },
    "Ashy Drongo": {
        "code": "ashdro",
        "sci_name": "Dicrurus leucophaeus",
        "info": "Grayish drongo with a deeply forked tail and agile flight."
    },
    "Bronzed Drongo": {
        "code": "brndro",
        "sci_name": "Dicrurus aeneus",
        "info": "Small drongo with metallic sheen and acrobatic flight."
    },
    "Greater Racket-tailed Drongo": {
        "code": "grrtdro",
        "sci_name": "Dicrurus paradiseus",
        "info": "Large drongo with elongated tail feathers ending in rackets."
    },
    "Black-naped Monarch": {
        "code": "bnmon",
        "sci_name": "Hypothymis azurea",
        "info": "Graceful flycatcher with a distinctive black nape and azure blue plumage."
    },
    "Indian Paradise-Flycatcher": {
        "code": "ipfly",
        "sci_name": "Terpsiphone paradisi",
        "info": "Elegant bird with long tail streamers and sexual dimorphism in plumage."
    },
    "Brown Shrike": {
        "code": "brnshr",
        "sci_name": "Lanius cristatus",
        "info": "Migratory shrike with a brown back and distinctive eye stripe."
    },
    "Rufous Treepie": {
        "code": "ruftreep",
        "sci_name": "Dendrocitta vagabunda",
        "info": "Noisy bird with a long tail and rufous plumage."
    },
    "White-bellied Treepie": {
        "code": "wbtreep",
        "sci_name": "Dendrocitta leucogastra",
        "info": "Endemic to the Western Ghats, with contrasting black and white plumage."
    },
    "House Crow": {
        "code": "hscrow",
        "sci_name": "Corvus splendens",
        "info": "Common urban bird with a gray neck and scavenging habits."
    },
    "Large-billed Crow": {
        "code": "lbcrow",
        "sci_name": "Corvus macrorhynchos",
        "info": "Robust crow with a large bill and all-black plumage."
    },
    "Gray-headed Canary-Flycatcher": {
        "code": "ghcfly",
        "sci_name": "Culicicapa ceylonensis",
        "info": "Small flycatcher with a gray head and yellow underparts."
    },
    "Indian Yellow Tit": {
        "code": "iytit",
        "sci_name": "Machlolophus aplonotus",
        "info": "Bright yellow tit with a distinctive black crest."
    },
    "Jerdon's Bushlark": {
        "code": "jbuslark",
        "sci_name": "Mirafra affinis",
        "info": "Ground-dwelling bird with a melodious song and streaked plumage."
    },
    "Common Tailorbird": {
        "code": "ctailb",
        "sci_name": "Orthotomus sutorius",
        "info": "Small warbler known for stitching leaves to build its nest."
    },
    "Gray-breasted Prinia": {
        "code": "gbprinia",
        "sci_name": "Prinia hodgsonii",
        "info": "Active warbler with a gray breast and distinctive call."
    },
    "Ashy Prinia": {
        "code": "ashprinia",
        "sci_name": "Prinia socialis",
        "info": "Small bird with ashy upperparts and a jerky tail movement."
    },
    "Plain Prinia": {
        "code": "plprinia",
        "sci_name": "Prinia inornata",
        "info": "Unstreaked warbler with a plain appearance and repetitive song."
    },
    "Zitting Cisticola": {
        "code": "zitcist",
        "sci_name": "Cisticola juncidis",
        "info": "Tiny bird with a distinctive 'zitting' call during flight displays."
    },
    "Thick-billed Warbler": {
        "code": "tbwarb",
        "sci_name": "Arundinax aedon",
        "info": "Large warbler with a robust bill and skulking behavior."
    },
    "Blyth's Reed Warbler": {
        "code": "brwarb",
        "sci_name": "Acrocephalus dumetorum",
        "info": "Migratory warbler with a rich, varied song."
    },
    "Barn Swallow": {
        "code": "barnsw",
        "sci_name": "Hirundo rustica",
        "info": "Graceful swallow with a deeply forked tail and agile flight."
    },
    "Red-rumped Swallow": {
        "code": "rrswallow",
        "sci_name": "Cecropis daurica",
        "info": "Swallow with a reddish rump and mud nest-building habits."
    },
    "Flame-throated Bulbul": {
        "code": "ftbulbul",
        "sci_name": "Rubigula gularis",
        "info": "Endemic bulbul with a fiery orange throat patch."
    },
    "Red-vented Bulbul": {
        "code": "rvbulbul",
        "sci_name": "Pycnonotus cafer",
        "info": "Common bulbul with a red vent and cheerful song."
    },
    "Red-whiskered Bulbul": {
        "code": "rwbulbul",
        "sci_name": "Pycnonotus jocosus",
        "info": "Bulbul with a prominent red patch below the eye and melodious calls."
    },
    "White-browed Bulbul": {
        "code": "wbbulbul",
        "sci_name": "Pycnonotus luteolus",
        "info": "Bulbul with a distinctive white brow and yellowish underparts."
    },
    "Yellow-browed Bulbul": {
        "code": "ybbulbul",
        "sci_name": "Acritillas indica",
        "info": "Bright yellow bulbul with a noticeable yellow brow."
    },
    "Square-tailed Bulbul": {
        "code": "stbulbul",
        "sci_name": "-",
        "info":"-"
    },
     "Lesser Whistling-Duck": {
        "code": "lwduck",
        "sci_name": "Dendrocygna javanica",
        "info": "Noisy duck often found in wetlands, with a distinctive whistling call and chestnut body."
    },
    "Garganey": {
        "code": "garganey",
        "sci_name": "Spatula querquedula",
        "info": "Small migratory duck with a striking white eyebrow in males."
    },
    "Indian Spot-billed Duck": {
        "code": "isbduck",
        "sci_name": "Anas poecilorhyncha",
        "info": "Medium-sized duck with a yellow-tipped black bill and characteristic white-bordered speculum."
    },
    "Indian Peafowl": {
        "code": "indpeafowl",
        "sci_name": "Pavo cristatus",
        "info": "National bird of India known for its iridescent blue plumage and extravagant courtship display."
    },
    "Red Spurfowl": {
        "code": "redspur",
        "sci_name": "Galloperdix spadicea",
        "info": "Rusty-brown spurfowl often heard more than seen in undergrowth habitats."
    },
    "Gray Junglefowl": {
        "code": "gyjunglefowl",
        "sci_name": "Gallus sonneratii",
        "info": "Wild ancestor of domestic chickens with striking gray and yellow plumage."
    },
    "Gray Francolin": {
        "code": "gyfrancolin",
        "sci_name": "Ortygornis pondicerianus",
        "info": "Ground-dwelling bird with a loud, repetitive call and finely barred gray plumage."
    },
    "Little Grebe": {
        "code": "ltgrebe",
        "sci_name": "Tachybaptus ruficollis",
        "info": "Small diving bird with a chestnut neck and excellent swimming ability."
    },
    "Rock Pigeon": {
        "code": "rockpgn",
        "sci_name": "Columba livia",
        "info": "Urban-adapted pigeon with iridescent neck and common around human settlements."
    },
    "Nilgiri Wood-Pigeon": {
        "code": "nlgpgn",
        "sci_name": "Columba elphinstonii",
        "info": "Endemic to the Western Ghats, large pigeon with pinkish breast and scaly nape."
    },
    "Eurasian Collared-Dove": {
        "code": "eurcdove",
        "sci_name": "Streptopelia decaocto",
        "info": "Pale dove with a black collar around the neck, often found in open habitats."
    },
    "Spotted Dove": {
        "code": "spotdove",
        "sci_name": "Spilopelia chinensis",
        "info": "Dove with a spotted black-and-white neck patch and a gentle call."
    },
    "Laughing Dove": {
        "code": "laughdove",
        "sci_name": "Spilopelia senegalensis",
        "info": "Small pinkish dove with a chuckling call and scaly neck pattern."
    },
    "Asian Emerald Dove": {
        "code": "aemdove",
        "sci_name": "Chalcophaps indica",
        "info": "Beautiful dove with iridescent green wings and a preference for forested areas."
    },
    "Gray-fronted Green-Pigeon": {
        "code": "gfgpgn",
        "sci_name": "Treron affinis",
        "info": "Forest-dwelling green pigeon endemic to the Western Ghats with a gray forehead."
    },
    "Mountain Imperial-Pigeon": {
        "code": "mipgn",
        "sci_name": "Ducula badia",
        "info": "Large forest pigeon with a deep call and maroon back."
    },
    "Greater Coucal": {
        "code": "gtcoucal",
        "sci_name": "Centropus sinensis",
        "info": "Large crow-like bird with a coppery back and deep booming call."
    },
    "Asian Koel": {
        "code": "askoel",
        "sci_name": "Eudynamys scolopaceus",
        "info": "Brood parasite known for its loud and repetitive calls."
    },
    "Gray-bellied Cuckoo": {
        "code": "gybcuckoo",
        "sci_name": "Cacomantis passerinus",
        "info": "Small cuckoo with gray underparts and rapid 'pee-pip-pee' call."
    },
    "Common Hawk-Cuckoo": {
        "code": "chcuckoo",
        "sci_name": "Hierococcyx varius",
        "info": "Also known as 'brainfever bird' due to its loud and persistent call."
    },
    "Great Eared-Nightjar": {
        "code": "genjar",
        "sci_name": "Lyncornis macrotis",
        "info": "Large nightjar with ear tufts and booming nocturnal call."
    },
    "Jungle Nightjar": {
        "code": "jngnjar",
        "sci_name": "Caprimulgus indicus",
        "info": "Cryptically colored nightjar, active during dusk and dawn."
    },
    "Little Swift": {
        "code": "ltswift",
        "sci_name": "Apus affinis",
        "info": "Small aerial bird with a white rump and fast flight."
    },
    "Asian Palm-Swift": {
        "code": "apmswift",
        "sci_name": "Cypsiurus balasiensis",
        "info": "Slim swift often seen around palms, known for its speed and agility."
    },
    "Eurasian Moorhen": {
        "code": "eurmoorhen",
        "sci_name": "Gallinula chloropus",
        "info": "Common waterbird with a red frontal shield and yellow-tipped bill."
    },
    "Eurasian Coot": {
        "code": "eurcoot",
        "sci_name": "Fulica atra",
        "info": "Black aquatic bird with a white frontal shield and lobed toes."
    },
    "Gray-headed Swamphen": {
        "code": "ghswamphen",
        "sci_name": "Porphyrio poliocephalus",
        "info": "Large purple-blue rail with a pale gray head and loud calls."
    },
    "White-breasted Waterhen": {
        "code": "wbwaterhen",
        "sci_name": "Amaurornis phoenicurus",
        "info": "Common wetland bird with white face and breast, often seen darting in reeds."
    },
    "Black-winged Stilt": {
        "code": "bwstilt",
        "sci_name": "Himantopus himantopus",
        "info": "Slender wader with long pink legs and contrasting black-and-white plumage."
    },
    "Red-wattled Lapwing": {
        "code": "rwlplover",
        "sci_name": "Vanellus indicus",
        "info": "Loud and striking plover with a red wattle in front of each eye."
    },
    "Kentish Plover": {
        "code": "kplover",
        "sci_name": "Charadrius alexandrinus",
        "info": "Small sandy-colored shorebird often found on beaches and mudflats."
    },
    "Little Ringed Plover": {
        "code": "lrplover",
        "sci_name": "Charadrius dubius",
        "info": "Tiny plover with a yellow eye-ring and sharp two-note call."
    },
    "Bronze-winged Jacana": {
        "code": "bwjacana",
        "sci_name": "Metopidius indicus",
        "info": "Wetland bird with metallic bronze wings and long toes for walking on lily pads."
    },
    "Common Sandpiper": {
        "code": "cmsandpiper",
        "sci_name": "Actitis hypoleucos",
        "info": "Shorebird with a bobbing tail and white crescent between shoulder and chest."
    },
    "Green Sandpiper": {
        "code": "grsandpiper",
        "sci_name": "Tringa ochropus",
        "info": "Solitary wader with dark upperparts and contrasting white rump."
    },
    "Common Greenshank": {
        "code": "cmgreenshank",
        "sci_name": "Tringa nebularia",
        "info": "Tall wader with slightly upturned bill and loud ringing calls."
    },
    "Marsh Sandpiper": {
        "code": "marsandpiper",
        "sci_name": "Tringa stagnatilis",
        "info": "Delicate shorebird with long legs and a needle-thin bill."
    },
    "Wood Sandpiper": {
        "code": "wdsandpiper",
        "sci_name": "Tringa glareola",
        "info": "Small spotted wader that prefers shallow freshwater wetlands."
    },
    "Whiskered Tern": {
        "code": "whtern",
        "sci_name": "Chlidonias hybrida",
        "info": "Graceful marsh tern with breeding black belly and contrasting gray plumage."
    },
    "Asian Openbill": {
        "code": "asopenbill",
        "sci_name": "Anastomus oscitans",
        "info": "Medium-sized stork with a distinctive gap between its mandibles."
    },
    "Painted Stork": {
        "code": "paintstork",
        "sci_name": "Mycteria leucocephala",
        "info": "Large colorful stork with pink tertials and yellow bill."
    },
    "Oriental Darter": {
        "code": "ordarter",
        "sci_name": "Anhinga melanogaster",
        "info": "Also known as the snakebird, it swims with only its head and neck above water."
    },
    "Gray Heron": {
        "code": "gryheron",
        "sci_name": "Ardea cinerea",
        "info": "Tall, elegant heron with a dagger-like bill and gray plumage."
    },
    "Purple Heron": {
        "code": "purpheron",
        "sci_name": "Ardea purpurea",
        "info": "Slender heron with reddish neck stripes and a preference for reed beds."
    },
    "Great Egret": {
        "code": "gtegret",
        "sci_name": "Ardea alba",
        "info": "Large white egret with a yellow bill and elegant stature."
    },
    "Intermediate Egret": {
        "code": "integret",
        "sci_name": "Ardea intermedia",
        "info": "Medium-sized egret with a thick neck and straight yellow bill."
    },
    "Little Egret": {
        "code": "ltegret",
        "sci_name": "Egretta garzetta",
        "info": "Slender white egret with black legs and yellow feet."
    },
    "Cattle Egret": {
        "code": "cattlegret",
        "sci_name": "Bubulcus ibis",
        "info": "Often seen near grazing cattle, this egret has a stocky build and yellow bill."
    },
    "Indian Pond-Heron": {
        "code": "iphheron",
        "sci_name": "Ardeola grayii",
        "info": "Pale heron that blends with surroundings and flushes with sudden white wings."
    },
    "Black-crowned Night-Heron": {
        "code": "bcnheron",
        "sci_name": "Nycticorax nycticorax",
        "info": "Nocturnal heron with a stocky build and distinctive black crown."
    },
    "Glossy Ibis": {
        "code": "glosibis",
        "sci_name": "Plegadis falcinellus",
        "info": "Wading bird with iridescent dark plumage and long curved bill."
    },
    "Black-winged Kite": {
        "code": "bwkite",
        "sci_name": "Elanus caeruleus",
        "info": "Small raptor with white body and striking black shoulder patches."
    },
    "Oriental Honey-buzzard": {
        "code": "ohbuzzard",
        "sci_name": "Pernis ptilorhynchus",
        "info": "Medium-sized raptor known for feeding on bees and wasps."
    },
    "Crested Serpent-Eagle": {
        "code": "cseagle",
        "sci_name": "Spilornis cheela",
        "info": "Forest eagle with prominent crest and yellow eyes."
    },
    "Black Eagle": {
        "code": "bkeagle",
        "sci_name": "Ictinaetus malaiensis",
        "info": "Large dark raptor with broad wings and soaring flight."
    },
    "Eurasian Marsh-Harrier": {
        "code": "emharrier",
        "sci_name": "Circus aeruginosus",
        "info": "Low-flying hawk hunting over wetlands and marshes."
    },
    "Crested Goshawk": {
        "code": "cgoshawk",
        "sci_name": "Accipiter trivirgatus",
        "info": "Small forest hawk with a distinctive crest and sharp hunting skills."
    },
    "Shikra": {
        "code": "shikra",
        "sci_name": "Accipiter badius",
        "info": "Small hawk often seen darting through urban and forested areas."
    },
    "Black Kite": {
        "code": "blkite",
        "sci_name": "Milvus migrans",
        "info": "Common scavenger raptor with forked tail and soaring flight."
    },
    "Brahminy Kite": {
        "code": "brkite",
        "sci_name": "Haliastur indus",
        "info": "Raptor with chestnut body and white head, often seen near water."
    },
    "Indian Scops-Owl": {
        "code": "iscoowl",
        "sci_name": "Otus bakkamoena",
        "info": "Small owl with ear tufts and camouflaged plumage."
    },
    "Spot-bellied Eagle-Owl": {
        "code": "sbeagleowl",
        "sci_name": "Bubo nipalensis",
        "info": "Large owl with spotted belly and deep hooting calls."
    },
    "Brown Fish-Owl": {
        "code": "bfowl",
        "sci_name": "Ketupa zeylonensis",
        "info": "Nocturnal owl that hunts fish along riverbanks."
    },
    "Jungle Owlet": {
        "code": "jungowl",
        "sci_name": "Glaucidium radiatum",
        "info": "Small forest owl with spotted wings and daytime activity."
    },
    "Spotted Owlet": {
        "code": "spotowl",
        "sci_name": "Athene brama",
        "info": "Small owl with white-spotted brown plumage often found near human settlements."
    },
    "Brown Wood-Owl": {
        "code": "bwowl",
        "sci_name": "Strix leptogrammica",
        "info": "Large nocturnal owl with brown streaked feathers and silent flight."
    },
    "Brown Boobook": {
        "code": "bboobook",
        "sci_name": "Ninox scutulata",
        "info": "Medium-sized owl known for its distinctive calls and brown plumage."
    },
    "Malabar Trogon": {
        "code": "maltrogon",
        "sci_name": "Harpactes fasciatus",
        "info": "Colorful forest bird with red underparts and green back."
    },
    "Eurasian Hoopoe": {
        "code": "ehoopoe",
        "sci_name": "Upupa epops",
        "info": "Bird with a striking crest and long curved bill."
    },
    "Great Hornbill": {
        "code": "grhornbill",
        "sci_name": "Buceros bicornis",
        "info": "Large bird with massive casque on bill and loud calls."
    },
    "Malabar Gray Hornbill": {
        "code": "malghornbill",
        "sci_name": "Ocyceros griseus",
        "info": "Medium-sized hornbill endemic to Western Ghats with gray plumage."
    },
    "Common Kingfisher": {
        "code": "cmkingfisher",
        "sci_name": "Alcedo atthis",
        "info": "Small, brightly colored bird with a sharp beak for fishing."
    },
    "Stork-billed Kingfisher": {
        "code": "sbkingfisher",
        "sci_name": "Pelargopsis capensis",
        "info": "Large kingfisher with heavy bill and loud, piercing call."
    },
    "White-throated Kingfisher": {
        "code": "wtkingfisher",
        "sci_name": "Halcyon smyrnensis",
        "info": "Bright blue and brown bird with a conspicuous white throat."
    },
    "Pied Kingfisher": {
        "code": "pkingfisher",
        "sci_name": "Ceryle rudis",
        "info": "Black and white kingfisher known for hovering while hunting fish."
    },
    "Green Bee-eater": {
        "code": "gbbeater",
        "sci_name": "Merops orientalis",
        "info": "Slender bird with vivid green plumage and a curved beak."
    },
    "Blue-tailed Bee-eater": {
        "code": "btbeater",
        "sci_name": "Merops philippinus",
        "info": "Vibrantly colored bee-eater with a distinctive blue tail."
    },
    "Chestnut-headed Bee-eater": {
        "code": "chbeater",
        "sci_name": "Merops leschenaulti",
        "info": "Bee-eater with a chestnut-colored head and bright green body."
    },
    "Indian Roller": {
        "code": "indroller",
        "sci_name": "Coracias benghalensis",
        "info": "Brightly colored bird known for spectacular aerial displays."
    },
    "Malabar Barbet": {
        "code": "malbarbet",
        "sci_name": "Psilopogon malabaricus",
        "info": "Small barbet with green and yellow plumage, endemic to Western Ghats."
    },
    "Coppersmith Barbet": {
        "code": "copbarbet",
        "sci_name": "Psilopogon haemacephalus",
        "info": "Barbet known for its metallic 'coppersmith' call."
    },
    "White-cheeked Barbet": {
        "code": "wcbarbet",
        "sci_name": "Psilopogon viridis",
        "info": "Green barbet with white cheeks found in southern India."
    },
    "Speckled Piculet": {
        "code": "specpiculet",
        "sci_name": "Picumnus innominatus",
        "info": "Tiny woodpecker-like bird with speckled plumage."
    },
    "Heart-spotted Woodpecker": {
        "code": "hswoodpecker",
        "sci_name": "Hemicircus canente",
        "info": "Small woodpecker with heart-shaped spots on wings."
    },
    "Brown-capped Pygmy Woodpecker": {
        "code": "bcppygmy",
        "sci_name": "Yungipicus nanus",
        "info": "Smallest woodpecker with brown cap and spotted underparts."
    },
    "Greater Flameback": {
        "code": "grflameback",
        "sci_name": "Chrysocolaptes guttacristatus",
        "info": "Large woodpecker with bright golden-yellow back and red crest."
    },
    "Rufous Woodpecker": {
        "code": "rufwoodpecker",
        "sci_name": "Micropternus brachyurus",
        "info": "Woodpecker with reddish-brown plumage and loud calls."
    },
    "Common Flameback": {
        "code": "comflameback",
        "sci_name": "Dinopium javanense",
        "info": "Medium woodpecker with striking red crest and golden back."
    },
    "Black-rumped Flameback": {
        "code": "brflameback",
        "sci_name": "Dinopium benghalense",
        "info": "Woodpecker with a black rump and bright yellow back."
    },
    "Lesser Yellownape": {
        "code": "lesyellownape",
        "sci_name": "Picus chlorolophus",
        "info": "Greenish-yellow woodpecker with a prominent yellow nape."
    },
    "Streak-throated Woodpecker": {
        "code": "stwoodpecker",
        "sci_name": "Picus xanthopygaeus",
        "info": "Woodpecker with streaked throat and olive-green plumage."
    },
    "White-bellied Woodpecker": {
        "code": "wbwoodpecker",
        "sci_name": "Dryocopus javensis",
        "info": "Large woodpecker with white belly and red crown."
    },
    "Rose-ringed Parakeet": {
        "code": "rrparakeet",
        "sci_name": "Psittacula krameri",
        "info": "Common parakeet with green plumage and distinctive red neck ring."
    },
    "Plum-headed Parakeet": {
        "code": "phparakeet",
        "sci_name": "Psittacula cyanocephala",
        "info": "Parakeet with striking plum-colored head and green body."
    },
    "Malabar Parakeet": {
        "code": "malparakeet",
        "sci_name": "Psittacula columboides",
        "info": "Endemic parakeet of Western Ghats with deep green and blue hues."
    },
    "Vernal Hanging-Parrot": {
        "code": "vhparrot",
        "sci_name": "Loriculus vernalis",
        "info": "Small, bright green parrot known for hanging upside down."
    },
    "Indian Pitta": {
        "code": "indpitta",
        "sci_name": "Pitta brachyura",
        "info": "Colorful ground bird with a bright green back and orange belly."
    },
    "Small Minivet": {
        "code": "smminivet",
        "sci_name": "Pericrocotus cinnamomeus",
        "info": "Small insectivorous bird with orange and gray plumage."
    },
    "Orange Minivet": {
        "code": "orminivet",
        "sci_name": "Pericrocotus flammeus",
        "info": "Vibrant minivet with striking orange and black colors."
    },
    "Indian Golden Oriole": {
        "code": "igoriole",
        "sci_name": "Oriolus kundoo",
        "info": "Bright yellow bird with black wings and melodious call."
    },
    "Black-hooded Oriole": {
        "code": "bhoriole",
        "sci_name": "Oriolus xanthornus",
        "info": "Yellow bird with a distinctive black hood."
    },
    "Ashy Woodswallow": {
        "code": "ashwoodswallow",
        "sci_name": "Artamus fuscus",
        "info": "Slender bird with ash-gray plumage and sharp flight skills."
    },
    "Malabar Woodshrike": {
        "code": "malwoodshrike",
        "sci_name": "Tephrodornis sylvicola",
        "info": "Grayish woodshrike endemic to the Western Ghats."
    },
    "Bar-winged Flycatcher-shrike": {
        "code": "bwflyshrike",
        "sci_name": "Hemipus picatus",
        "info": "Small black-and-white flycatcher with barred wings."
    },
    "Common Iora": {
        "code": "comiora",
        "sci_name": "Aegithina tiphia",
        "info": "Small bird with bright yellow underparts and melodious calls."
    },
    "Black Drongo": {
        "code": "blkdrongo",
        "sci_name": "Dicrurus macrocercus",
        "info": "Glossy black bird with forked tail and aggressive behavior."
    },
    "Ashy Drongo": {
        "code": "ashdrongo",
        "sci_name": "Dicrurus leucophaeus",
        "info": "Drongo with ashy gray plumage and sharp features."
    },
    "Bronzed Drongo": {
        "code": "brndrongo",
        "sci_name": "Dicrurus aeneus",
        "info": "Dark drongo with a metallic bronze sheen."
    },
    "Greater Racket-tailed Drongo": {
        "code": "grrtdrongo",
        "sci_name": "Dicrurus paradiseus",
        "info": "Distinctive drongo with long racket-shaped tail feathers."
    },
    "Black-naped Monarch": {
        "code": "bknmonarch",
        "sci_name": "Hypothymis azurea",
        "info": "Small bright blue flycatcher with a black nape."
    },
    "Indian Paradise-Flycatcher": {
        "code": "indparadise",
        "sci_name": "Terpsiphone paradisi",
        "info": "Elegant bird with long tail streamers and striking plumage."
    },
    "Brown Shrike": {
        "code": "bnshrike",
        "sci_name": "Lanius cristatus",
        "info": "Small brown bird with hooked beak and masked face."
    },
    "Rufous Treepie": {
        "code": "ruftreepie",
        "sci_name": "Dendrocitta vagabunda",
        "info": "Reddish-brown bird with long tail and loud calls."
    },
    "White-bellied Treepie": {
        "code": "wbtreepie",
        "sci_name": "Dendrocitta leucogastra",
        "info": "Treepie with contrasting white belly and dark upperparts."
    },
    "House Crow": {
        "code": "housecrow",
        "sci_name": "Corvus splendens",
        "info": "Common urban crow with gray neck and black body."
    },
    "Large-billed Crow": {
        "code": "lbcrow",
        "sci_name": "Corvus macrorhynchos",
        "info": "Big crow with strong bill and glossy black plumage."
    },
    "Gray-headed Canary-Flycatcher": {
        "code": "ghcanfly",
        "sci_name": "Culicicapa ceylonensis",
        "info": "Small bird with gray head and bright yellow underparts."
    },
    "Indian Yellow Tit": {
        "code": "indyellowtit",
        "sci_name": "Machlolophus aplonotus",
        "info": "Yellow crested tit endemic to India."
    },
    "Jerdon's Bushlark": {
        "code": "jerdbushlark",
        "sci_name": "Mirafra affinis",
        "info": "Small brown lark known for its melodious song."
    },
    "Common Tailorbird": {
        "code": "comtailorbird",
        "sci_name": "Orthotomus sutorius",
        "info": "Tiny bird famous for sewing leaves to build its nest."
    },
     "Gray-breasted Prinia": {
        "code": "gbprinia",
        "sci_name": "Prinia hodgsonii",
        "info": "Small warbler with grayish breast and lively behavior."
    },
    "Ashy Prinia": {
        "code": "ashyprinia",
        "sci_name": "Prinia socialis",
        "info": "Grayish-brown warbler common in scrub and gardens."
    },
    "Plain Prinia": {
        "code": "plainprinia",
        "sci_name": "Prinia inornata",
        "info": "Small warbler with plain brown upperparts and active tail flicking."
    },
    "Zitting Cisticola": {
        "code": "zittingcisticola",
        "sci_name": "Cisticola juncidis",
        "info": "Tiny warbler known for its distinctive 'zitting' call."
    },
    "Thick-billed Warbler": {
        "code": "thickbillwarbler",
        "sci_name": "Arundinax aedon",
        "info": "Large warbler with robust bill and loud, complex song."
    },
    "Blyth's Reed Warbler": {
        "code": "blythsreedwarbler",
        "sci_name": "Acrocephalus dumetorum",
        "info": "Migratory warbler found in reedbeds and dense vegetation."
    },
    "Barn Swallow": {
        "code": "barnswallow",
        "sci_name": "Hirundo rustica",
        "info": "Famous swallow with deeply forked tail and graceful flight."
    },
    "Red-rumped Swallow": {
        "code": "redrumpswallow",
        "sci_name": "Cecropis daurica",
        "info": "Swallow with distinctive reddish rump and chestnut throat."
    },
    "Flame-throated Bulbul": {
        "code": "flamethroatbulbul",
        "sci_name": "Rubigula gularis",
        "info": "Vibrant bulbul with bright orange throat patch."
    },
    "Red-vented Bulbul": {
        "code": "redventbulbul",
        "sci_name": "Pycnonotus cafer",
        "info": "Common garden bird with distinctive red vent patch."
    },
    "Red-whiskered Bulbul": {
        "code": "redwhiskerbulbul",
        "sci_name": "Pycnonotus jocosus",
        "info": "Bulbul with red cheek patches and crest."
    },
    "White-browed Bulbul": {
        "code": "whitebrowedbulbul",
        "sci_name": "Pycnonotus luteolus",
        "info": "Bulbul with prominent white eyebrow stripe."
    },
    "Yellow-browed Bulbul": {
        "code": "yellowbrowedbulbul",
        "sci_name": "Acritillas indica",
        "info": "Bulbul with yellow eyebrow and olive-green body."
    },
    "Square-tailed Bulbul": {
        "code": "squaretailbulbul",
        "sci_name": "Hypsipetes ganeesa",
        "info": "Bulbul named for its distinctive squared tail."
    },
    "Tickell's Leaf Warbler": {
        "code": "tickellsleafwarbler",
        "sci_name": "Phylloscopus affinis",
        "info": "Small warbler with distinctive call and greenish plumage."
    },
    "Green Warbler": {
        "code": "greenwarbler",
        "sci_name": "Phylloscopus nitidus",
        "info": "Bright green warbler often found in forests."
    },
    "Greenish Warbler": {
        "code": "greenishwarbler",
        "sci_name": "Phylloscopus trochiloides",
        "info": "Migratory warbler with subtle green hues."
    },
    "Large-billed Leaf Warbler": {
        "code": "largebilledleafwarbler",
        "sci_name": "Phylloscopus magnirostris",
        "info": "Warbler with a notably large bill and loud call."
    },
    "Dark-fronted Babbler": {
        "code": "darkfrontbabbler",
        "sci_name": "Dumetia atriceps",
        "info": "Social babbler with dark forehead and noisy calls."
    },
    "Indian Scimitar-Babbler": {
        "code": "indianscimitarbabbler",
        "sci_name": "Pomatorhinus horsfieldii",
        "info": "Babbler with distinctive curved bill and melodious calls."
    },
    "Puff-throated Babbler": {
        "code": "puffthroatbabbler",
        "sci_name": "Pellorneum ruficeps",
        "info": "Brown babbler with puffed throat feathers."
    },
    "Brown-cheeked Fulvetta": {
        "code": "browncheekfulvetta",
        "sci_name": "Alcippe poioicephala",
        "info": "Small bird with brown cheeks and soft calls."
    },
    "Palani Laughingthrush": {
        "code": "palanilaughingthrush",
        "sci_name": "Montecincla fairbanki",
        "info": "Laughingthrush endemic to Palani hills with rufous body."
    },
    "Rufous Babbler": {
        "code": "rufousbabbler",
        "sci_name": "Turdoides subrufus",
        "info": "Reddish-brown babbler with strong social bonds."
    },
    "Jungle Babbler": {
        "code": "junglebabbler",
        "sci_name": "Turdoides striata",
        "info": "Noisy babbler often seen in groups called 'seven sisters'."
    },
    "Yellow-billed Babbler": {
        "code": "yellowbilledbabbler",
        "sci_name": "Turdoides affinis",
        "info": "Babbler with bright yellow bill and chatty nature."
    },
    "Wayanad Laughingthrush": {
        "code": "wayanadlaughingthrush",
        "sci_name": "Montecincla jerdoni",
        "info": "Endemic laughingthrush with dark face and rufous wings."
    },
    "Velvet-fronted Nuthatch": {
        "code": "velvetfrontnuthatch",
        "sci_name": "Sitta frontalis",
        "info": "Small nuthatch with velvet blue head and chestnut belly."
    },
    "Southern Hill Myna": {
        "code": "southernhillmyna",
        "sci_name": "Gracula indica",
        "info": "Glossy black myna known for excellent mimicry."
    },
    "Rosy Starling": {
        "code": "rosystarling",
        "sci_name": "Pastor roseus",
        "info": "Migratory starling with pink body and black wings."
    },
    "Brahminy Starling": {
        "code": "brahminystarling",
        "sci_name": "Sturnia pagodarum",
        "info": "Starling with pale body and dark head patch."
    },
    "Common Myna": {
        "code": "commonmyna",
        "sci_name": "Acridotheres tristis",
        "info": "Familiar urban bird with loud calls and yellow eye patch."
    },
    "Jungle Myna": {
        "code": "junglemyna",
        "sci_name": "Acridotheres fuscus",
        "info": "Myna with brown body and distinct tufted crest."
    },
    "Indian Blackbird": {
        "code": "indianblackbird",
        "sci_name": "Turdus simillimus",
        "info": "Black thrush found in hilly forests of India."
    },
    "Asian Brown Flycatcher": {
        "code": "asianbrownflycatcher",
        "sci_name": "Muscicapa dauurica",
        "info": "Small flycatcher with brown upperparts and pale underparts."
    },
    "Indian Robin": {
        "code": "indianrobin",
        "sci_name": "Copsychus fulicatus",
        "info": "Small bird with black plumage and white wing patches."
    },
    "White-bellied Sholakili": {
        "code": "wbsholakili",
        "sci_name": "Sholicola albiventris",
        "info": "Rare bird from shola forests with white belly."
    },
    "White-bellied Blue Flycatcher": {
        "code": "wbblueflycatcher",
        "sci_name": "Cyornis pallipes",
        "info": "Beautiful blue flycatcher with white underparts."
    },
    "Tickell's Blue Flycatcher": {
        "code": "tickellsblueflycatcher",
        "sci_name": "Cyornis tickelliae",
        "info": "Small flycatcher with bright blue upperparts."
    },
    "Nilgiri Flycatcher": {
        "code": "nilgiriflycatcher",
        "sci_name": "Eumyias albicaudatus",
        "info": "Vibrant blue flycatcher endemic to Nilgiri hills."
    },
    "Indian Blue Robin": {
        "code": "indianbluerobin",
        "sci_name": "Larvivora brunnea",
        "info": "Shy forest bird with blue upperparts and orange underparts."
    },
    "Malabar Whistling-Thrush": {
        "code": "malabarwhistlingthrush",
        "sci_name": "Myophonus horsfieldii",
        "info": "Nocturnal bird known for its loud whistling song."
    },
    "Black-and-orange Flycatcher": {
        "code": "blackandorangeflycatcher",
        "sci_name": "Ficedula nigrorufa",
        "info": "Striking flycatcher with black and orange plumage."
    },
    "Rusty-tailed Flycatcher": {
        "code": "rustytailedflycatcher",
        "sci_name": "Ficedula ruficauda",
        "info": "Small flycatcher with rusty tail and pale underparts."
    },
    "Pied Bushchat": {
        "code": "piedbushchat",
        "sci_name": "Saxicola caprata",
        "info": "Small black and white chat commonly seen in open areas."
    },
    "Pale-billed Flowerpecker": {
        "code": "palebilledflowerpecker",
        "sci_name": "Dicaeum erythrorhynchos",
        "info": "Tiny nectar-feeding bird with pale bill and active behavior."
    },
    "Nilgiri Flowerpecker": {
        "code": "nilgiriflowerpecker",
        "sci_name": "Dicaeum concolor",
        "info": "Small bird with olive-green upperparts endemic to Nilgiri."
    },
    "Purple-rumped Sunbird": {
        "code": "purplerumpedsunbird",
        "sci_name": "Leptocoma zeylonica",
        "info": "Iridescent sunbird with purple rump and rapid flight."
    },
    "Crimson-backed Sunbird": {
        "code": "crimsonbackedsunbird",
        "sci_name": "Leptocoma minima",
        "info": "Small sunbird with brilliant crimson back."
    },
    "Purple Sunbird": {
        "code": "purplesunbird",
        "sci_name": "Cinnyris asiaticus",
        "info": "Common sunbird with metallic purple male plumage."
    },
    "Loten's Sunbird": {
        "code": "lotenssunbird",
        "sci_name": "Cinnyris lotenius",
        "info": "Small sunbird with bright yellow underparts."
    },
    "Little Spiderhunter": {
        "code": "littlespiderhunter",
        "sci_name": "Arachnothera longirostra",
        "info": "Spiderhunter with long curved bill for probing flowers."
    },
    "Golden-fronted Leafbird": {
        "code": "goldenfrontedleafbird",
        "sci_name": "Chloropsis aurifrons",
        "info": "Bright green bird with golden forehead and melodious calls."
    },
    "Scaly-breasted Munia": {
        "code": "scalybreastedmunia",
        "sci_name": "Lonchura punctulata",
        "info": "Small finch with distinctive scaly breast pattern."
    },
    "White-rumped Munia": {
        "code": "whiterumpedmunia",
        "sci_name": "Lonchura striata",
        "info": "Finch with brown body and contrasting white rump."
    },
    "House Sparrow": {
        "code": "housesparrow",
        "sci_name": "Passer domesticus",
        "info": "Common urban sparrow with stout body and noisy calls."
    },
    "Forest Wagtail": {
        "code": "forestwagtail",
        "sci_name": "Dendronanthus indicus",
        "info": "Unique wagtail that wags its tail side to side."
    },
    "Gray Wagtail": {
        "code": "graywagtail",
        "sci_name": "Motacilla cinerea",
        "info": "Slender wagtail with long tail and yellow underparts."
    },
    "Western Yellow Wagtail": {
        "code": "westernyellowwagtail",
        "sci_name": "Motacilla flava",
        "info": "Bright yellow wagtail common in wetlands during migration."
    },
    "White-browed Wagtail": {
        "code": "whitebrowedwagtail",
        "sci_name": "Motacilla maderaspatensis",
        "info": "Black and white wagtail often found near water bodies."
    },
    "Paddyfield Pipit": {
        "code": "paddyfieldpipit",
        "sci_name": "Anthus rufulus",
        "info": "Small, streaked bird common in grasslands and fields."
    },
    "Common Rosefinch": {
        "code": "commonrosefinch",
        "sci_name": "Carpodacus erythrinus",
        "info": "Bright red male finch known for its melodious song."
    },
    "Lesser Whistling-Duck": {
        "code": "leswhd",
        "sci_name": "Dendrocygna javanica",
        "info": "Nocturnal duck with distinctive whistling calls."
    },
    "Garganey": {
        "code": "gargan",
        "sci_name": "Spatula querquedula",
        "info": "Small dabbling duck that migrates long distances."
    },
    "Indian Spot-billed Duck": {
        "code": "indspotduck",
        "sci_name": "Anas poecilorhyncha",
        "info": "Medium-sized duck with distinctive yellow spot on bill."
    },
    "Indian Peafowl": {
        "code": "indpeafowl",
        "sci_name": "Pavo cristatus",
        "info": "National bird of India, famous for its spectacular courtship display."
    },
    "Red Spurfowl": {
        "code": "redspurfowl",
        "sci_name": "Galloperdix spadicea",
        "info": "Ground-dwelling bird with reddish plumage and spurs on legs."
    },
    "Gray Junglefowl": {
        "code": "grayjunglefowl",
        "sci_name": "Gallus sonneratii",
        "info": "Wild relative of domestic chicken with gray body and colorful neck."
    },
    "Gray Francolin": {
        "code": "grayfrancolin",
        "sci_name": "Ortygornis pondicerianus",
        "info": "Common partridge-like bird with grayish plumage."
    },
    "Little Grebe": {
        "code": "littlegrebe",
        "sci_name": "Tachybaptus ruficollis",
        "info": "Small diving waterbird with a distinctive sharp call."
    },
    "Rock Pigeon": {
        "code": "rockpigeon",
        "sci_name": "Columba livia",
        "info": "Common urban pigeon, highly adaptable to city life."
    },
    "Nilgiri Wood-Pigeon": {
        "code": "nilgiriwoodpigeon",
        "sci_name": "Columba elphinstonii",
        "info": "Large pigeon with pale neck patch, endemic to Western Ghats."
    },
    "Eurasian Collared-Dove": {
        "code": "eurasiancollareddove",
        "sci_name": "Streptopelia decaocto",
        "info": "Pale dove with a black collar around its neck."
    },
    "Spotted Dove": {
        "code": "spotteddove",
        "sci_name": "Spilopelia chinensis",
        "info": "Medium-sized dove with distinctive spotted neck patch."
    },
    "Laughing Dove": {
        "code": "laughingdove",
        "sci_name": "Spilopelia senegalensis",
        "info": "Small dove known for its soft, laughing calls."
    },
    "Asian Emerald Dove": {
        "code": "asianemeralddove",
        "sci_name": "Chalcophaps indica",
        "info": "Small green dove with a metallic emerald sheen."
    },
    "Gray-fronted Green-Pigeon": {
        "code": "grayfrontedgreenpigeon",
        "sci_name": "Treron affinis",
        "info": "Forest pigeon with gray forehead and green body."
    },
    "Mountain Imperial-Pigeon": {
        "code": "mountainimperialpigeon",
        "sci_name": "Ducula badia",
        "info": "Large forest pigeon with chestnut and gray coloration."
    },
    "Greater Coucal": {
        "code": "greatercoucal",
        "sci_name": "Centropus sinensis",
        "info": "Large crow-like bird with deep booming calls."
    },
    "Asian Koel": {
        "code": "asiankoel",
        "sci_name": "Eudynamys scolopaceus",
        "info": "Cuckoo known for its loud, melodious calls during breeding season."
    },
    "Gray-bellied Cuckoo": {
        "code": "graybelliedcuckoo",
        "sci_name": "Cacomantis passerinus",
        "info": "Small cuckoo with gray belly and repetitive calls."
    },
    "Common Hawk-Cuckoo": {
        "code": "commonhawkcuckoo",
        "sci_name": "Hierococcyx varius",
        "info": "Also called brainfever bird for its loud and repetitive call."
    },
    "Great Eared-Nightjar": {
        "code": "greatearednightjar",
        "sci_name": "Lyncornis macrotis",
        "info": "Nocturnal bird with prominent ear tufts and camouflaged plumage."
    },
    "Jungle Nightjar": {
        "code": "junglenightjar",
        "sci_name": "Caprimulgus indicus",
        "info": "Nightjar with distinctive wing markings and silent flight."
    },
    "Little Swift": {
        "code": "littleswift",
        "sci_name": "Apus affinis",
        "info": "Small fast-flying bird commonly seen near buildings."
    },
    "Asian Palm-Swift": {
        "code": "asianpalmswift",
        "sci_name": "Cypsiurus balasiensis",
        "info": "Swift often seen gliding around palm trees."
    },
    "Eurasian Moorhen": {
        "code": "eurasianmoorhen",
        "sci_name": "Gallinula chloropus",
        "info": "Common waterbird with red frontal shield and white undertail."
    },
    "Eurasian Coot": {
        "code": "eurasiancoot",
        "sci_name": "Fulica atra",
        "info": "Black waterbird with a distinctive white frontal shield."
    },
    "Gray-headed Swamphen": {
        "code": "grayheadedswamphen",
        "sci_name": "Porphyrio poliocephalus",
        "info": "Large colorful waterbird with gray head and purple body."
    },
    "White-breasted Waterhen": {
        "code": "whitebreastedwaterhen",
        "sci_name": "Amaurornis phoenicurus",
        "info": "Waterbird with white face and breast, often seen near wetlands."
    },
    "Black-winged Stilt": {
        "code": "blackwingedstilt",
        "sci_name": "Himantopus himantopus",
        "info": "Long-legged wader with black wings and white body."
    },
    "Red-wattled Lapwing": {
        "code": "redwattledlapwing",
        "sci_name": "Vanellus indicus",
        "info": "Wader known for its loud alarm calls and red facial wattles."
    },
    "Kentish Plover": {
        "code": "kentishplover",
        "sci_name": "Charadrius alexandrinus",
        "info": "Small shorebird with pale plumage and dark eye patches."
    },
    "Little Ringed Plover": {
        "code": "littleringedplover",
        "sci_name": "Charadrius dubius",
        "info": "Small plover with a distinctive black neck ring."
    },
    "Bronze-winged Jacana": {
        "code": "bronzewingedjacana",
        "sci_name": "Metopidius indicus",
        "info": "Waterbird with long toes that walk on floating vegetation."
    },
    "Common Sandpiper": {
        "code": "commonsandpiper",
        "sci_name": "Actitis hypoleucos",
        "info": "Small wader with distinctive bobbing motion."
    },
    "Green Sandpiper": {
        "code": "greensandpiper",
        "sci_name": "Tringa ochropus",
        "info": "Wader with dark upperparts and pale underparts."
    },
    "Common Greenshank": {
        "code": "commongreenshank",
        "sci_name": "Tringa nebularia",
        "info": "Long-legged wader with slightly upturned bill."
    },
    "Marsh Sandpiper": {
        "code": "marshsandpiper",
        "sci_name": "Tringa stagnatilis",
        "info": "Slim wader with long legs and delicate build."
    },
    "Wood Sandpiper": {
        "code": "woodsandpiper",
        "sci_name": "Tringa glareola",
        "info": "Medium-sized wader with speckled upperparts and yellowish legs."
    },
    "Whiskered Tern": {
        "code": "whiskeredtern",
        "sci_name": "Chlidonias hybrida",
        "info": "Graceful marsh tern with black cap and grey body during breeding."
    },
    "Asian Openbill": {
        "code": "asianopenbill",
        "sci_name": "Anastomus oscitans",
        "info": "Stork with distinctive gap between upper and lower bill."
    },
    "Painted Stork": {
        "code": "paintedstork",
        "sci_name": "Mycteria leucocephala",
        "info": "Large stork with pink tertials and yellow beak."
    },
    "Oriental Darter": {
        "code": "orientaldarter",
        "sci_name": "Anhinga melanogaster",
        "info": "Slim waterbird also called snakebird for its long neck."
    },
    "Gray Heron": {
        "code": "grayheron",
        "sci_name": "Ardea cinerea",
        "info": "Large wader with gray plumage and a long neck."
    },
    "Purple Heron": {
        "code": "purpleheron",
        "sci_name": "Ardea purpurea",
        "info": "Slender heron with rich chestnut neck and streaked underparts."
    },
    "Great Egret": {
        "code": "greategret",
        "sci_name": "Ardea alba",
        "info": "Elegant large egret with pure white plumage and yellow bill."
    },
    "Intermediate Egret": {
        "code": "intermediateegret",
        "sci_name": "Ardea intermedia",
        "info": "Medium-sized egret with shorter neck than great egret."
    },
    "Little Egret": {
        "code": "littleegret",
        "sci_name": "Egretta garzetta",
        "info": "Small white egret with black legs and yellow feet."
    },
    "Cattle Egret": {
        "code": "cattleegret",
        "sci_name": "Bubulcus ibis",
        "info": "Often found near grazing animals; white plumage with buff patches in breeding."
    },
    "Indian Pond-Heron": {
        "code": "indianpondheron",
        "sci_name": "Ardeola grayii",
        "info": "Stocky heron often seen still at pond edges; brown in flight turns white."
    },
    "Black-crowned Night-Heron": {
        "code": "blackcrownednightheron",
        "sci_name": "Nycticorax nycticorax",
        "info": "Nocturnal heron with black crown and back, and red eyes."
    },
    "Glossy Ibis": {
        "code": "glossyibis",
        "sci_name": "Plegadis falcinellus",
        "info": "Slender ibis with iridescent dark plumage."
    },
    "Black-winged Kite": {
        "code": "blackwingedkite",
        "sci_name": "Elanus caeruleus",
        "info": "Small raptor with red eyes and hovering flight."
    },
    "Oriental Honey-buzzard": {
        "code": "orientalhoneybuzzard",
        "sci_name": "Pernis ptilorhynchus",
        "info": "Forest raptor feeding mainly on bee and wasp nests."
    },
    "Crested Serpent-Eagle": {
        "code": "crestedserpenteagle",
        "sci_name": "Spilornis cheela",
        "info": "Medium raptor with broad wings and a loud whistling call."
    },
    "Black Eagle": {
        "code": "blackeagle",
        "sci_name": "Ictinaetus malaiensis",
        "info": "Large black raptor with distinct fingered wings and slow flight."
    },
    "Eurasian Marsh-Harrier": {
        "code": "eurasianmarshharrier",
        "sci_name": "Circus aeruginosus",
        "info": "Harrier that glides low over wetlands hunting prey."
    },
    "Crested Goshawk": {
        "code": "crestedgoshawk",
        "sci_name": "Accipiter trivirgatus",
        "info": "Forest hawk with bold crest and barred underparts."
    },
    "Shikra": {
        "code": "shikra",
        "sci_name": "Accipiter badius",
        "info": "Small hawk with red eyes and a sharp hunting style."
    },
    "Black Kite": {
        "code": "blackkite",
        "sci_name": "Milvus migrans",
        "info": "Common scavenger with forked tail and graceful flight."
    },
    "Brahminy Kite": {
        "code": "brahminykite",
        "sci_name": "Haliastur indus",
        "info": "Striking raptor with white head and chestnut body."
    },
    "Indian Scops-Owl": {
        "code": "indianscopsowl",
        "sci_name": "Otus bakkamoena",
        "info": "Tiny owl with ear tufts and camouflaged plumage."
    },
    "Spot-bellied Eagle-Owl": {
        "code": "spotbelleagleowl",
        "sci_name": "Ketupa nipalensis",
        "info": "Large and powerful owl with bold spots and fierce call."
    },
    "Brown Fish-Owl": {
        "code": "brownfishowl",
        "sci_name": "Ketupa zeylonensis",
        "info": "Shaggy-looking owl near water bodies, hunts fish and frogs."
    },
    "Jungle Owlet": {
        "code": "jungleowlet",
        "sci_name": "Glaucidium radiatum",
        "info": "Small diurnal owl often seen perched in daylight."
    },
    "Spotted Owlet": {
        "code": "spottedowlet",
        "sci_name": "Athene brama",
        "info": "Familiar small owl with white spots and curious gaze."
    },
    "Brown Wood-Owl": {
        "code": "brownwoodowl",
        "sci_name": "Strix leptogrammica",
        "info": "Shy forest owl with dark eyes and rich brown plumage."
    },
    "Brown Boobook": {
        "code": "brownboobook",
        "sci_name": "Ninox scutulata",
        "info": "Medium-sized hawk-owl active at dusk and dawn."
    },
    "Malabar Trogon": {
        "code": "malabartrogon",
        "sci_name": "Harpactes fasciatus",
        "info": "Beautiful Western Ghats endemic with vivid colors."
    },
    "Eurasian Hoopoe": {
        "code": "eurasianhoopoe",
        "sci_name": "Upupa epops",
        "info": "Distinctive crest and undulating flight with zebra wings."
    },
    "Great Hornbill": {
        "code": "greathornbill",
        "sci_name": "Buceros bicornis",
        "info": "Massive forest hornbill with huge yellow casque."
    },
    "Malabar Gray Hornbill": {
        "code": "malabargrayhornbill",
        "sci_name": "Ocyceros griseus",
        "info": "Western Ghats endemic hornbill with a curved bill and nasal calls."
    },
    "Common Kingfisher": {
        "code": "commonkingfisher",
        "sci_name": "Alcedo atthis",
        "info": "Tiny jewel-like kingfisher often perched near water."
    },
    "Stork-billed Kingfisher": {
        "code": "storkbilledkingfisher",
        "sci_name": "Pelargopsis capensis",
        "info": "Large kingfisher with heavy red bill and loud call."
    },
    "White-throated Kingfisher": {
        "code": "whitethroatedkingfisher",
        "sci_name": "Halcyon smyrnensis",
        "info": "Common bright blue kingfisher with white throat and chestnut head."
    },
    "Pied Kingfisher": {
        "code": "piedkingfisher",
        "sci_name": "Ceryle rudis",
        "info": "Black-and-white kingfisher that hovers over water to dive."
    },
    "Green Bee-eater": {
        "code": "greenbeeater",
        "sci_name": "Merops orientalis",
        "info": "Slender bird with green plumage and acrobatic flight."
    },
    "Blue-tailed Bee-eater": {
        "code": "bluetailedbeeater",
        "sci_name": "Merops philippinus",
        "info": "Migratory bee-eater with elegant blue tail and chestnut throat."
    },
    "Chestnut-headed Bee-eater": {
        "code": "chestnutheadedbeeater",
        "sci_name": "Merops leschenaulti",
        "info": "Colorful bee-eater with chestnut crown and nape, found in open forests."
    },
    "Indian Roller": {
        "code": "indianroller",
        "sci_name": "Coracias benghalensis",
        "info": "Bright blue and brown bird, often seen perched in open areas."
    },
    "Malabar Barbet": {
        "code": "malabarbarbet",
        "sci_name": "Psilopogon malabaricus",
        "info": "Endemic barbet with a red crown, mostly green plumage, and repetitive call."
    },
    "Coppersmith Barbet": {
        "code": "coppersmithbarbet",
        "sci_name": "Psilopogon haemacephalus",
        "info": "Common barbet known for its metallic 'tuk-tuk-tuk' call, green body, red forehead."
    },
    "White-cheeked Barbet": {
        "code": "whitecheekedbarbet",
        "sci_name": "Psilopogon viridis",
        "info": "Green barbet with white cheeks and distinctive repetitive call."
    },
    "Speckled Piculet": {
        "code": "speckledpiculet",
        "sci_name": "Picumnus innominatus",
        "info": "Tiny woodpecker-like bird with spotted plumage and short tail."
    },
    "Heart-spotted Woodpecker": {
        "code": "heartspottedwoodpecker",
        "sci_name": "Hemicircus canente",
        "info": "Compact woodpecker with heart-shaped spots on white underparts."
    },
    "Brown-capped Pygmy Woodpecker": {
        "code": "browncappedpygmywoodpecker",
        "sci_name": "Yungipicus nanus",
        "info": "Smallest woodpecker in India with a brown crown and barred back."
    },
    "Greater Flameback": {
        "code": "greaterflameback",
        "sci_name": "Chrysocolaptes guttacristatus",
        "info": "Large woodpecker with golden wings and loud rattling call."
    },
    "Rufous Woodpecker": {
        "code": "rufouswoodpecker",
        "sci_name": "Micropternus brachyurus",
        "info": "Unique rufous-brown woodpecker that nests in ant nests."
    },
    "Common Flameback": {
        "code": "commonflameback",
        "sci_name": "Dinopium javanense",
        "info": "Golden-backed woodpecker with red crest and distinctive call."
    },
    "Black-rumped Flameback": {
        "code": "blackrumpedflameback",
        "sci_name": "Dinopium benghalense",
        "info": "Widespread woodpecker with black rump and golden wings."
    },
    "Lesser Yellownape": {
        "code": "lesseryellownape",
        "sci_name": "Picus chlorolophus",
        "info": "Green woodpecker with yellow nape and streaked underparts."
    },
    "Streak-throated Woodpecker": {
        "code": "streakthroatedwoodpecker",
        "sci_name": "Picus xanthopygaeus",
        "info": "Woodpecker with streaked throat and olive green back."
    },
    "White-bellied Woodpecker": {
        "code": "whitebelliedwoodpecker",
        "sci_name": "Dryocopus javensis",
        "info": "Large black woodpecker with striking white belly and loud calls."
    },
    "Rose-ringed Parakeet": {
        "code": "roseringedparakeet",
        "sci_name": "Psittacula krameri",
        "info": "Green parakeet with red bill and neck ring in males."
    },
    "Plum-headed Parakeet": {
        "code": "plumheadedparakeet",
        "sci_name": "Psittacula cyanocephala",
        "info": "Colorful parakeet with plum-colored head and green body."
    },
    "Malabar Parakeet": {
        "code": "malabarparakeet",
        "sci_name": "Psittacula columboides",
        "info": "Western Ghats endemic with blue plumage and long tail."
    },
    "Vernal Hanging-Parrot": {
        "code": "vernalhangingparrot",
        "sci_name": "Loriculus vernalis",
        "info": "Tiny green parrot that hangs upside down while feeding."
    },
    "Indian Pitta": {
        "code": "indianpitta",
        "sci_name": "Pitta brachyura",
        "info": "Vibrantly colored bird with short tail and whistling call, seen in undergrowth."
    },
     "Small Minivet": {
        "code": "smallminivet",
        "sci_name": "Pericrocotus cinnamomeus",
        "info": "Small, active bird with males showing orange underparts and females yellow."
    },
    "Orange Minivet": {
        "code": "orangeminivet",
        "sci_name": "Pericrocotus flammeus",
        "info": "Striking forest bird with fiery orange and black plumage in males."
    },
    "Indian Golden Oriole": {
        "code": "indiangoldenoriole",
        "sci_name": "Oriolus kundoo",
        "info": "Brilliant yellow oriole with black wings and a fluty, melodious song."
    },
    "Black-hooded Oriole": {
        "code": "blackhoodedoriole",
        "sci_name": "Oriolus xanthornus",
        "info": "Tropical oriole with a jet black head and vibrant yellow body."
    },
    "Ashy Woodswallow": {
        "code": "ashywoodswallow",
        "sci_name": "Artamus fuscus",
        "info": "Soft gray bird with a stubby bill and a fondness for wires and bare branches."
    },
    "Malabar Woodshrike": {
        "code": "malabarwoodshrike",
        "sci_name": "Tephrodornis sylvicola",
        "info": "Western Ghats endemic woodshrike with ashy plumage and a shrike-like call."
    },
    "Bar-winged Flycatcher-shrike": {
        "code": "barwingedflycatchershrike",
        "sci_name": "Hemipus picatus",
        "info": "Small bird with black upperparts and white wingbars, often seen in mixed flocks."
    },
    "Common Iora": {
        "code": "commoniora",
        "sci_name": "Aegithina tiphia",
        "info": "Bright yellow and green bird with loud whistling calls, males have black upperparts."
    },
    "Black Drongo": {
        "code": "blackdrongo",
        "sci_name": "Dicrurus macrocercus",
        "info": "Glossy black bird with a deeply forked tail and aggressive behavior."
    },
    "Ashy Drongo": {
        "code": "ashydrongo",
        "sci_name": "Dicrurus leucophaeus",
        "info": "Slate-gray drongo with a less forked tail, common in wooded areas."
    },
    "Bronzed Drongo": {
        "code": "bronzeddrongo",
        "sci_name": "Dicrurus aeneus",
        "info": "Glossy black drongo with a metallic sheen and slightly forked tail, found in forests."
    },
    "Greater Racket-tailed Drongo": {
        "code": "greaterrackettaileddrongo",
        "sci_name": "Dicrurus paradiseus",
        "info": "Large drongo with long tail rackets and mimicry skills, often noisy and conspicuous."
    },
    "Black-naped Monarch": {
        "code": "blacknapedmonarch",
        "sci_name": "Hypothymis azurea",
        "info": "Delicate blue flycatcher with a black nape band and sweet whistling call."
    },
    "Indian Paradise-Flycatcher": {
        "code": "indianparadiseflycatcher",
        "sci_name": "Terpsiphone paradisi",
        "info": "Elegant flycatcher with long tail streamers; males may be rufous or white."
    },
    "Brown Shrike": {
        "code": "brownshrike",
        "sci_name": "Lanius cristatus",
        "info": "Migratory shrike with a brown back, pale underparts, and strong hooked beak."
    },
    "Rufous Treepie": {
        "code": "rufoustreepie",
        "sci_name": "Dendrocitta vagabunda",
        "info": "Long-tailed rufous and grey bird of open forests, often noisy and social."
    },
    "White-bellied Treepie": {
        "code": "whitebelliedtreepie",
        "sci_name": "Dendrocitta leucogastra",
        "info": "Endemic treepie of the Western Ghats with striking white and black coloration."
    },
    "House Crow": {
        "code": "housecrow",
        "sci_name": "Corvus splendens",
        "info": "Common city-dwelling crow with a gray nape and intelligent behavior."
    },
    "Large-billed Crow": {
        "code": "largebilledcrow",
        "sci_name": "Corvus macrorhynchos",
        "info": "All-black forest crow with a bulky bill and varied vocalizations."
    },
    "Gray-headed Canary-Flycatcher": {
        "code": "grayheadedcanaryflycatcher",
        "sci_name": "Culicicapa ceylonensis",
        "info": "Small yellow and gray flycatcher with a fluttery flight and sweet voice."
    },
    "Indian Yellow Tit": {
        "code": "indianyellowtit",
        "sci_name": "Machlolophus aplonotus",
        "info": "Bright yellow tit with a black crest and bib, active and acrobatic."
    },
    "Jerdon's Bushlark": {
        "code": "jerdonsbushlark",
        "sci_name": "Mirafra affinis",
        "info": "Streaky brown lark found in open country, known for display flights and bubbling song."
    },
    "Common Tailorbird": {
        "code": "commontailorbird",
        "sci_name": "Orthotomus sutorius",
        "info": "Small, active bird with a loud call and remarkable nest-stitching behavior."
    },
    "Gray-breasted Prinia": {
        "code": "graybreastedprinia",
        "sci_name": "Prinia hodgsonii",
        "info": "Small warbler with gray breast and tail-wagging habit, often in scrub."
    },
    "Ashy Prinia": {
        "code": "ashyprinia",
        "sci_name": "Prinia socialis",
        "info": "Tiny, dusky warbler with a metallic call and jerky movements in undergrowth."
    },
    "Plain Prinia": {
        "code": "plainprinia",
        "sci_name": "Prinia inornata",
        "info": "Slender warbler with plain plumage and sharp calls, common in grasslands."
    },
    "Zitting Cisticola": {
        "code": "zittingcisticola",
        "sci_name": "Cisticola juncidis",
        "info": "Tiny bird with a buzzing song and bouncing flight display over grassy areas."
    },
    "Thick-billed Warbler": {
        "code": "thickbilledwarbler",
        "sci_name": "Arundinax aedon",
        "info": "Migratory skulker with a strong bill and loud chattering call, prefers dense scrub."
    },
    "Blyth's Reed Warbler": {
        "code": "blythsreedwarbler",
        "sci_name": "Acrocephalus dumetorum",
        "info": "Secretive warbler with olive-brown plumage, a summer breeder in Eurasia, winters in India."
    },
    "Barn Swallow": {
        "code": "barnswallow",
        "sci_name": "Hirundo rustica",
        "info": "Graceful swallow with a forked tail and agile flight, often nesting near human habitation."
    },
    "Red-rumped Swallow": {
        "code": "redrumpedswallow",
        "sci_name": "Cecropis daurica",
        "info": "Swallow with reddish rump and slower flight, builds mud nests under ledges."
    },
    "Flame-throated Bulbul": {
        "code": "flamethroatedbulbul",
        "sci_name": "Rubigula gularis",
        "info": "Western Ghats endemic bulbul with a bright red throat and yellow body."
    },
    "Red-vented Bulbul": {
        "code": "redventedbulbul",
        "sci_name": "Pycnonotus cafer",
        "info": "Widespread bulbul with a dark head and red vent, adaptable and vocal."
    },
    "Red-whiskered Bulbul": {
        "code": "redwhiskeredbulbul",
        "sci_name": "Pycnonotus jocosus",
        "info": "Stylish bulbul with red cheek patch and crest, often seen in gardens and forests."
    },
    "White-browed Bulbul": {
        "code": "whitebrowedbulbul",
        "sci_name": "Pycnonotus luteolus",
        "info": "Bulbul with distinctive white eyebrow and brownish back, prefers scrub habitats."
    },
    "Yellow-browed Bulbul": {
        "code": "yellowbrowedbulbul",
        "sci_name": "Acritillas indica",
        "info": "Western Ghats endemic with yellow-green plumage and prominent yellow brow."
    },
    "Square-tailed Bulbul": {
        "code": "squaretailedbulbul",
        "sci_name": "Hypsipetes ganeesa",
        "info": "Dark bulbul with a square tail, endemic to the hills of southern India."
    },
    "Tickell's Leaf Warbler": {
        "code": "tickellsleafwarbler",
        "sci_name": "Phylloscopus affinis",
        "info": "Small greenish warbler with two wingbars, often seen in the canopy during migration."
    },
    "Green Warbler": {
        "code": "greenwarbler",
        "sci_name": "Phylloscopus nitidus",
        "info": "Bright green warbler with striking yellow supercilium, migrates through India."
    },
    "Greenish Warbler": {
        "code": "greenishwarbler",
        "sci_name": "Phylloscopus trochiloides",
        "info": "Inconspicuous migratory warbler with greenish tones and a variable high-pitched call."
    },
    "Large-billed Leaf Warbler": {
        "code": "largebilledleafwarbler",
        "sci_name": "Phylloscopus magnirostris",
        "info": "Robust warbler with a large bill and bold eyebrow, active in mid to high canopy."
    },
    "Dark-fronted Babbler": {
        "code": "darkfrontedbabbler",
        "sci_name": "Dumetia atriceps",
        "info": "Small babbler with dark face and chestnut cap, seen in small flocks in undergrowth."
    },
    "Indian Scimitar-Babbler": {
        "code": "indianscimitarbabbler",
        "sci_name": "Pomatorhinus horsfieldii",
        "info": "Babbler with long downcurved bill and white eyebrow, skulks in dense vegetation."
    },
    "Puff-throated Babbler": {
        "code": "puffthroatedbabbler",
        "sci_name": "Pellorneum ruficeps",
        "info": "Shy ground babbler with streaked throat and rich call, often heard before seen."
    },
    "Brown-cheeked Fulvetta": {
        "code": "browncheekedfulvetta",
        "sci_name": "Alcippe poioicephala",
        "info": "Soft-plumaged bird with warm brown cheeks and back, forages in mixed flocks."
    },
    "Palani Laughingthrush": {
        "code": "palanilaughingthrush",
        "sci_name": "Montecincla fairbanki",
        "info": "Endemic to the Palani Hills, this laughingthrush has a harsh call and secretive habits."
    },
    "Rufous Babbler": {
        "code": "rufousbabbler",
        "sci_name": "Argya subrufa",
        "info": "Rusty-colored babbler found in Western Ghats, moves in noisy flocks."
    },
    "Jungle Babbler": {
        "code": "junglebabbler",
        "sci_name": "Argya striata",
        "info": "Common 'seven sisters' babbler seen in noisy groups, grey with a yellow eye."
    },
    "Yellow-billed Babbler": {
        "code": "yellowbilledbabbler",
        "sci_name": "Argya affinis",
        "info": "Pale, scruffy babbler with bright yellow bill, gregarious and often near habitation."
    },
    "Wayanad Laughingthrush": {
        "code": "wayanadlaughingthrush",
        "sci_name": "Montecincla jerdoni",
        "info": "Endemic to Wayanad, this elusive laughingthrush prefers montane forest edges."
    },
    "Velvet-fronted Nuthatch": {
        "code": "velvetfrontednuthatch",
        "sci_name": "Sitta frontalis",
        "info": "Colorful nuthatch with violet-blue plumage and bright red bill, creeps along tree trunks."
    },
    "Southern Hill Myna": {
        "code": "southernhillmyna",
        "sci_name": "Gracula indica",
        "info": "Glossy black starling with orange-yellow wattles, excellent mimic, found in forests."
    },
    "Rosy Starling": {
        "code": "rosystarling",
        "sci_name": "Pastor roseus",
        "info": "Migratory starling with pink body and glossy black head, seen in flocks."
    },
    "Brahminy Starling": {
        "code": "brahminystarling",
        "sci_name": "Sturnia pagodarum",
        "info": "Elegant starling with buffy plumage and black crest, frequents open wooded areas."
    },
    "Common Myna": {
        "code": "commonmyna",
        "sci_name": "Acridotheres tristis",
        "info": "Familiar urban bird with yellow eye patch and confident strut, very vocal."
    },
    "Jungle Myna": {
        "code": "junglemyna",
        "sci_name": "Acridotheres fuscus",
        "info": "Similar to Common Myna but with tufted forehead and bluish eye skin."
    },
    "Indian Blackbird": {
        "code": "indianblackbird",
        "sci_name": "Turdus simillimus",
        "info": "A subspecies of the Eurasian Blackbird, found in forested hills of peninsular India."
    },
    "Asian Brown Flycatcher": {
        "code": "asianbrownflycatcher",
        "sci_name": "Muscicapa dauurica",
        "info": "Small, plain flycatcher with pale underparts and a faint eye ring, common in forests."
    },
    "Indian Robin": {
        "code": "indianrobin",
        "sci_name": "Saxicoloides fulicatus",
        "info": "Male with black body and white shoulder patch, females brown, found in open scrub."
    },
    "White-bellied Sholakili": {
        "code": "whitebelliedsholakili",
        "sci_name": "Sholicola albiventris",
        "info": "Endemic to the shola forests of Western Ghats, secretive small bird with a white belly."
    },
    "White-bellied Blue Flycatcher": {
        "code": "whitebelliedblueflycatcher",
        "sci_name": "Cyornis pallipes",
        "info": "Shiny blue upperparts with white belly, inhabits dense forest undergrowth."
    },
    "Tickell's Blue Flycatcher": {
        "code": "tickellsblueflycatcher",
        "sci_name": "Cyornis tickelliae",
        "info": "Bright blue male with orange throat, a common insectivore of forest edges."
    },
    "Nilgiri Flycatcher": {
        "code": "nilgiriflycatcher",
        "sci_name": "Eumyias albicaudatus",
        "info": "Dark indigo blue flycatcher endemic to Nilgiris, prefers dense shola forests."
    },
    "Indian Blue Robin": {
        "code": "indianbluerobin",
        "sci_name": "Larvivora brunnea",
        "info": "Male bright blue upperparts with rufous underparts, shy and found in thick undergrowth."
    },
    "Malabar Whistling-Thrush": {
        "code": "malabarwhistlingthrush",
        "sci_name": "Myophonus horsfieldii",
        "info": "Large thrush with glossy blue-black plumage, famous for loud whistling calls."
    },
    "Black-and-orange Flycatcher": {
        "code": "blackandorangeflycatcher",
        "sci_name": "Ficedula nigrorufa",
        "info": "Striking orange and black flycatcher endemic to shola forests of Western Ghats."
    },
    "Rusty-tailed Flycatcher": {
        "code": "rustytailedflycatcher",
        "sci_name": "Ficedula ruficauda",
        "info": "Small flycatcher with rust-colored tail and olive-brown upperparts, prefers open forests."
    },
    "Pied Bushchat": {
        "code": "piedbushchat",
        "sci_name": "Saxicola caprata",
        "info": "Male mostly black with white under wings, female brownish, common in open habitats."
    },
    "Pale-billed Flowerpecker": {
        "code": "palebilledflowerpecker",
        "sci_name": "Dicaeum erythrorhynchos",
        "info": "Tiny flowerpecker with pale bill, feeds mainly on berries and nectar."
    },
    "Nilgiri Flowerpecker": {
        "code": "nilgiriflowerpecker",
        "sci_name": "Dicaeum concolor",
        "info": "Small endemic flowerpecker of Nilgiris, olive-green with whitish underparts."
    },
    "Purple-rumped Sunbird": {
        "code": "purplerumpedsunbird",
        "sci_name": "Leptocoma zeylonica",
        "info": "Male with metallic purple rump and greenish head, nectar feeder with rapid wingbeats."
    },
    "Crimson-backed Sunbird": {
        "code": "crimsonbackedsunbird",
        "sci_name": "Leptocoma minima",
        "info": "Small sunbird with crimson back and bright colors, found in scrub and gardens."
    },
    "Purple Sunbird": {
        "code": "purplesunbird",
        "sci_name": "Cinnyris asiaticus",
        "info": "Common sunbird with metallic purple male breeding plumage, hovers to feed on nectar."
    },
    "Loten's Sunbird": {
        "code": "lotenssunbird",
        "sci_name": "Cinnyris lotenius",
        "info": "Bright iridescent sunbird named after Dr. Loten, inhabits dry scrub and gardens."
    },
    "Little Spiderhunter": {
        "code": "littlespiderhunter",
        "sci_name": "Arachnothera longirostra",
        "info": "Small bird with long curved bill specialized for spider and nectar feeding."
    },
    "Golden-fronted Leafbird": {
        "code": "goldenfrontedleafbird",
        "sci_name": "Chloropsis aurifrons",
        "info": "Bright green leafbird with golden forehead patch, mimics calls of other birds."
    },
    "Scaly-breasted Munia": {
        "code": "scalybreastedmunia",
        "sci_name": "Lonchura punctulata",
        "info": "Small finch with distinctive scaly breast pattern, common in grasslands and fields."
    },
    "White-rumped Munia": {
        "code": "whiterumpedmunia",
        "sci_name": "Lonchura striata",
        "info": "Finch with white rump and streaked underparts, often seen in flocks on open land."
    },
    "House Sparrow": {
        "code": "housesparrow",
        "sci_name": "Passer domesticus",
        "info": "Ubiquitous small bird in human habitations worldwide, males with chestnut crown."
    },
    "Forest Wagtail": {
        "code": "forestwagtail",
        "sci_name": "Dendronanthus indicus",
        "info": "Unique wagtail with habit of wagging tail side to side, inhabits wooded streams."
    },
    "Gray Wagtail": {
        "code": "graywagtail",
        "sci_name": "Motacilla cinerea",
        "info": "Slender wagtail with long tail and yellow underparts, often near running water."
    },
    "Western Yellow Wagtail": {
        "code": "westernyellowwagtail",
        "sci_name": "Motacilla flava",
        "info": "Migratory wagtail with bright yellow belly, breeds in Europe and winters in Asia."
    },
    "White-browed Wagtail": {
        "code": "whitebrowedwagtail",
        "sci_name": "Motacilla maderaspatensis",
        "info": "Large wagtail with striking white eyebrow and black back, common near water."
    },
    "Paddyfield Pipit": {
        "code": "paddyfieldpipit",
        "sci_name": "Anthus rufulus",
        "info": "Small, brown pipit found in grasslands and fields, known for its insect-like call."
    }}
    
    return model, bird_classes, bird_info, bird_d

def main():
    st.title("🎵 Advanced Bird Sound Classifier")
    st.markdown("Upload bird audio recordings to identify species using our deep learning model (pls give audio recordings <=5sec, else the first 5 seconds of the recording will be taken)")
    with st.sidebar:
        st.header("🌍 Why This Matters")
        st.markdown("""
**Birds are excellent indicators of biodiversity change** due to their mobility and diverse habitat needs. Shifts in bird populations can signal the success or failure of ecological restoration.

🚫 **Traditional surveys** are expensive and logistically tough to scale.

🎙️ **Passive Acoustic Monitoring (PAM)**, when paired with **machine learning**, enables efficient biodiversity monitoring across large regions and over time.

🌿 **The Western Ghats**, a UNESCO-listed Global Biodiversity Hotspot, hosts unique birdlife found nowhere else on Earth. But it's under threat from landscape and climate changes.

🔧 Our tool empowers conservationists to **monitor bird diversity rapidly**, helping drive effective restoration and conservation strategies.
        """)
    # Load resources
    model, bird_classes, bird_info,bird_d = load_resources()
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an audio file (WAV, MP3)", 
        type=["wav", "mp3", "ogg"],
        accept_multiple_files=False
    )
    
    if uploaded_file is not None and st.button("Analyze Audio", type="primary"):
        with st.spinner("Processing audio..."):
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            try:
                # Preprocess audio
                mel_spec = audio_to_melspectrogram(tmp_path)
                
                if mel_spec is not None:
                    # Prepare input for model (add batch and channel dimensions)
                    input_tensor = np.expand_dims(mel_spec, axis=0)  # Add batch dim
                    input_tensor = np.expand_dims(input_tensor, axis=-1)  # Add channel dim
                    print(input_tensor)
                    # Model prediction
                    if model:
                        predictions = model.predict(input_tensor)[0]
                    else:
                        # Demo fallback if model not loaded
                        predictions = np.random.random(len(bird_classes))
                        predictions = predictions / predictions.sum()
                    
                    # Get top prediction
                    top_idx = np.argmax(predictions)
                    print(top_idx)
                    top_bird = bird_d[bird_classes[top_idx]]
                    top_confidence = predictions[top_idx]
                    
                    # Display results
                    import streamlit.components.v1 as components
                    with st.container():
                        # Main result card
                        components.html(f"""
<div class="result-card">
    <div class="bird-name">{top_bird}</div>
    <div class="bird-sci-name">{bird_info[top_bird]['sci_name']} • {bird_info[top_bird]['code']}</div>
    <p>{bird_info[top_bird]['info']}</p>
    
    <div style="margin-top: 1.5rem;">
        <strong>Confidence:</strong> {top_confidence*100:.1f}%
        <div class="confidence-meter">
            <div class="confidence-fill" style="width: {top_confidence*100}%"></div>
        </div>
    </div>
</div>

<style>
.result-card {{
    border-radius: 12px;
    padding: 24px;
    margin: 16px 0;
    background: white;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    border-left: 6px solid #4CAF50;
    font-family: sans-serif;
}}
.bird-name {{
    color: #2C3E50;
    font-size: 1.8rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}}
.bird-sci-name {{
    color: #7F8C8D;
    font-style: italic;
    margin-bottom: 1rem;
}}
.confidence-meter {{
    height: 10px;
    background: #E0E0E0;
    border-radius: 5px;
    margin: 1rem 0;
    overflow: hidden;
}}
.confidence-fill {{
    height: 100%;
    background: linear-gradient(90deg, #4CAF50, #8BC34A);
    transition: width 0.5s ease;
}}
</style>
""", height=300) 
                        
                        # Spectrogram visualization
                        with st.expander("🔍 View Audio Spectrogram", expanded=False):
                            # Convert mel spec to image
                            img = (mel_spec).astype(np.uint8)
                            img = Image.fromarray(img).convert("RGB")
                            st.image(img, caption="Mel Spectrogram", use_container_width=True)
                        
                        # Top 5 predictions
                        st.subheader("Top Predictions")
                        top_indices = np.argsort(predictions)[::-1][:5]
                        
                        for idx in top_indices:
                            conf = predictions[idx]
                            st.markdown(f"""
                            <div class="top-prediction">
                                <div style="display: flex; justify-content: space-between;">
                                    <span><strong>{bird_classes[idx]}</strong> ({bird_info[bird_d[bird_classes[idx]]]['code']})</span>
                                    <span>{conf*100:.1f}%</span>
                                </div>
                                <div class="confidence-meter">
                                    <div class="confidence-fill" style="width: {conf*100}%"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Additional bird information
                        with st.expander("ℹ️ Species Information"):
                            st.markdown(f"""
                            **{top_bird}: ** {bird_info[top_bird]['info']}  
                            """)
                            st.audio(uploaded_file)
                
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
            finally:
                os.unlink(tmp_path)

if __name__ == "__main__":
    main()
