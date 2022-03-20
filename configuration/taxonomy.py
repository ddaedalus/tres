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
except: new_keywords = {}

print(new_keywords)
