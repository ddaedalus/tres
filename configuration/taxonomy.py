## Change this taxonomy modifying the example below ##
## Use your domain name (e.g. Hardware) as the key of the dict
## Use zeros for dict values as in the example below.

taxonomy = {

	# Example
    "Hardware": [
        # Here: Keywords
        {
            "hardware": 0,
            "buses": 0,
            "cables": 0,
            "calculators": 0,
            "components": 0,
            "embedded": 0,
            "SCSI": 0,
            "storage": 0,
            "systems": 0,
            "peripheral": 0,
            "programmable": 0
        },
        # Here: Keyphrases
        {
            "device drivers": 0,
            "programmable logic": 0,
            "open source": 0,
            "technical support": 0
        }
    ],
    "Sports": [ 
        {
            "airsoft": 0,
            "archery": 0,
            "badminton": 0,
            "baseball": 0,
            "basketball": 0,
            "bocce": 0,
            "boomerang": 0,
            "bowling": 0,
            "boxing": 0,
            "canoeing": 0,  
            "kayaking": 0,
            "caving": 0,
            "cheerleading": 0,
            "cricket": 0,
            "croquet": 0,
            "cycling": 0,
            "darts": 0,
            "equestrian": 0,
            "fencing": 0,
            "fishing": 0,
            "footbag": 0,
            "football": 0,
            "gaelic": 0,
            "goalball": 0,
            "golf": 0,
            "gymnastic": 0,
            "handball": 0,
            "hockey": 0,
            "hunting": 0,
            "kabbadi": 0,
            "korfball": 0,
            "lacrosse": 0,
            "lumberjack": 0,
            "motorsport": 0,
            "netball": 0,
            "olympic": 0,
            "orienteering": 0,
            "paddleball": 0,
            "paintball": 0,
            "pesäpallo": 0,
            "petanque": 0,
            "racing": 0,
            "racquetbal­l": 0,
            "rodeo": 0,
            "rounders": 0,
            "rowing": 0,
            "rugby": 0,
            "running": 0,
            "skateboarding": 0,
            "skating": 0,
            "skiing": 0,
            "soccer": 0,
            "softball": 0,
            "squash": 0,
            "surfing": 0,
            "swimming": 0,
            "tchoukball": 0,
            "tennis": 0,
            "volleyball": 0,
            "walking" :0,
            "wrestling": 0,
            "sport": 0
        },
        {
            "winter sports": 0,
            "water sports": 0,
            "track and field": 0,
            "team handball": 0,
            "table tennis": 0,
            "swimming and diving": 0,
            "strength sports": 0,
            "sepak takraw": 0,
            "rugby union": 0,
            "rugly league": 0,
            "rope skipping": 0,
            "multi-sports": 0,
            "martial arts": 0,
            "laser games": 0,
            "jai alai": 0,
            "ice hockey": 0,
            "ice skating": 0,
            "informal sports": 0,
            "horse racing": 0,
            "greyhound racing": 0,
            "cue sports": 0,
            "extreme sports": 0,
            "american football": 0,
            "animal sports": 0,
            "adventure racing": 0
        }
    ],

    "Food": [ 
        {
            "drink": 0,
            "wine": 0,
            "cheese": 0,
            "spicy": 0,
            "confectionary": 0,
            "durian": 0,
            "chefs": 0,
            "meat": 0,
            "snacks": 0,
            "food": 0
        },

        {
            "organic food": 0,
            "fast food": 0,
            "wild foods": 0,
            "slow food": 0,
            "jell-o":0,
            "dining guides": 0,
            "culinary tours": 0,
            "philosophy of food": 0
        }
    ]

}


## Ignore this part ##

from .config import *
from utils.hyperparameters import *
import pickle

path = "./files/"

print(path + 'new_keywords_' + domain + '.pickle')

taxonomy_keywords = taxonomy[domain][0]
taxonomy_phrases = taxonomy[domain][1]

try:
    with open(path + 'new_keywords_' + domain + '.pickle', 'rb') as handle:
        new_keywords = pickle.load(handle)
    handle.close()
except: new_keywords = {}

print(new_keywords)
