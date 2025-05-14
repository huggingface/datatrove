from datatrove.pipeline.filters.preprocess_beta1_filter import PreprocessBeta1Filter
import numpy as np

from datatrove.data import Document
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.text import PUNCTUATION_SET, split_into_words
from datatrove.utils.typeshelper import Languages

TEXT = """
Balance Bar Nutrition Energy Bars Smores

Roll over image to magnify & Click to Zoom
Images may vary from actual product

Why Set & Save?

  • Lock in the guaranteed lowest prices
  • Schedule delivery according to YOUR needs
  • Surprise gifts & more
  • Never run out of your favorite items
  • Change or cancel at any time

Just click "Add to Set & Save" when placing items in your cart.

Enjoy our low price guarantee and never run out with Set & Save.

Balance Bar Nutrition Energy Bars Smores -- 15 Bars

★★★★★★★★★★
You Save: 30%

Retail price: $23.84

Vitacost price: $16.49

In stock
Deliver 1 Time Only
Add to Set & Save

Added to My List as a guest*.

*Your guest list will be temporarily saved until you end this browser session.

1 item added to your list

Balance Bar Nutrition Energy Bars Smores -- 15 Bars

Oops! Something went wrong and we were unable to process your request. Please try again.

Quantity:

Nutrition Facts
Serving Size: 1 Bar (50 g)
Servings per Container: 15
Amount Per Serving% Daily Value
Calories200
   Calories from Fat60
Total Fat7 g11%
Saturated Fat4 g20%
Trans Fat0 g
Cholesterol5 mg2%
Sodium220 mg9%
Potassium120 mg3%
Total Carbohydrate23 g8%
   Dietary Fiber3 g12%
   Sugars14 g
Protein14 g27%
Vitamin A50%
Vitamin C100%
Calcium15%
Iron15%
Vitamin D25%
Vitamin E100%
Vitamin K15%
Thiamin15%
Riboflavin15%
Niacin15%
Vitamin B615%
Folate15%
Vitamin B1215%
Biotin15%
Pantothenic Acid15%
Phosphorus15%
Iodine15%
Zinc15%
Selenium15%
Copper15%
Manganese15%
Chromium15%
Molybdenum15%
Other Ingredients: Soy protein nuggets (soy protein isolate, tapioca starch, salt), protein blend (soy protein isolate, whey protein isolate, calcium caseinate, egg white, enzyme modified soy protein, casein, partially hydrolyzed milk protein isolate), glucose syrup, sugar, fractionated palm kernel and palm oil, whey protein concentrate, fructose, invert sugar, oligofructose, water, cocoa (processed with alkali), high oleic sunflower oil, glycerine. Contains less than 2% of natural flavor, ground corn, whole wheat flour, soy lecithin, invert evaporated cane juice, nonfat milk, maltitol syrup, brown sugar, maltodextrin, citrus fiber, salt, pectin, butterfat, potassium lactate, wheat starch, honey, sodium bicarbonate, soybean oil, carrageenan, tocopherols added to protect flavor.

Vitamins and Minerals: calcium phosphate, ascorbic acid, alpha-tocopherol acetate, ferric orthophosphate, niacinamide, zinc oxide, copper gluconate, calcium pantothenate, manganese sulfate, vitamin A acetate, pyridoxine hydrochloride, riboflavin, thiamine mononitrate, chromium chloride, folic acid, biotin, potassium iodide, sodium molybdate, sodium selenite, phytonadione, vitamin D3, vitamin B12.
Contains soybean, milk, wheat, egg. Produced on equipment that also processes peanuts, tree nuts and sesame.

Balance Bar Nutrition Energy Bars Smores Description

  • 14 Grams of Protein
  • 23 Vitamins & Minerals
  • Glycemic Index (38)
  • Excellent Source of Antioxidants (Vit A, C & E)
  • With Three Indulgent Layers
  • Kosher

We all strive for Balance in our hectic lives - juggling between work, family, working out and eating right. Since 1992, to make life easier, Balance Bar® challenged themselves to create balanced nutrition that tastes great. With a proven formula that has the right amount of protein, carbohydrates and dietary fat, they make sure to provide you with energy that lasts so you can get through your busy day and feel good about what you eat. Enjoy what keeps you going.

- from your friends at Balance Bar

Disclaimer
These statements have not been evaluated by the FDA. These products are not intended to diagnose, treat, cure, or prevent any disease.
Know someone who would love this product? Don’t keep it a secret – share it now!
Note: Your friend's email address is used for this one-time notification only. We will not collect or store or share it with any other parties.
Friend's Name:
Your Name:
Friend's Email:
Your Email:
Message to your friend: (optional, max 500 characters)
CAPTCHA
Change the CAPTCHA codeSpeak the CAPTCHA code
 
Enter the code exactly as shown: 
Frequently Bought Together:
Balance Bar Nutrition Energy Bars Smores
+
Balance Bar Gold Nutrition Energy Bar Caramel Nut Blast

Retail price together: $47.68

Vitacost price: $31.88

You Save: 33%

People Who Viewed This Item Also Viewed
Enter Shipping Zip Code:
Please enter a valid zip code
FLDC1
"""

doc = Document
doc.text = TEXT


def in_whitelist(w):
    return w.isdigit() or w == '%' or w == '(' or w == ')'


def is_pure_alpha_word(w):
    for c in w:
        if not c.isalpha():
            return False
    return True

def check_pure_alpha_word_ratio(words, min_pure_alpha_word_ratio: float = 0.5):
    num = 0
    for w in words:
        if is_pure_alpha_word(w):
           num += 1
    return num / len(words) > min_pure_alpha_word_ratio 
    
    
def check_char_dup_ratio(words, max_non_alpha_words_ratio: float | None = 0.8):
    n_words = len(words)

    # that 80 % of words in a document contain at least one alphabetic character
    if (
        max_non_alpha_words_ratio
        # nb of words with at least 1 alpha char < 0.8
        and sum([any((c.isalpha() for c in w)) or in_whitelist(w) for w in words]) / n_words < max_non_alpha_words_ratio
    ):
        return False
    return True    
    

def get_non_alpha_words(text: str | None):
    if not text:
        text = doc.text
    words = split_into_words(text, Languages.english)
    invalids = []
    for w in words:
        if not any((c.isalpha() for c in w)):
            invalids.append(w)
    m = {}
    for item in invalids:
        if not m.get(item):
            m[item] = 1
        else:
            m[item] +=1

    sorted_m = dict(sorted(m.items(), key=lambda item: item[1], reverse=True))            
    return sorted_m


def get_non_alpha_words_ratio(text: str | None, use_whitelist = True):
    if not text:
        text = doc.text
    words = split_into_words(text, Languages.english)
    return sum([any((c.isalpha() for c in w)) or (use_whitelist and in_whitelist(w)) for w in words]) / len(words)


def check_line_word_num(words, min_word_num: int = 3):
    return len(words) >= min_word_num


def is_line_valid(line: str):
    if line == '':
        return True
    words = split_into_words(line, Languages.english)
    if len(words) == 0:
        return False
    return check_line_word_num(words) \
        and check_char_dup_ratio(words)
        # and check_pure_alpha_word_ratio(words)


def modify_doc_by_paragraph(doc, valid_line_in_paragraph_ratio: float = 0.5):
    text = doc.text
    paras = text.split('\n\n')
    new_paras = []
    for para in paras:
        lines = para.split('\n')
        total_num = len(lines)
        invalid_line_num = 0
        for line in lines:
            if not is_line_valid(line):
                invalid_line_num += 1

        if (len(lines)-invalid_line_num) / total_num >= valid_line_in_paragraph_ratio:
            new_paras.append(para)
        

    new_text = '\n\n'.join(new_paras)
    doc.text = new_text
        


def print_ratios():
    print(get_non_alpha_words_ratio(None, False))
    print(get_non_alpha_words_ratio(None, True))
    print(get_non_alpha_words_ratio(new_text, False))
    print(get_non_alpha_words_ratio(new_text, True))



def test():
    doc = Document
    doc.text = """
User: Guest  Login
Document type:
Konferenzbeitrag 
Contribution type:
Vortrag / Präsentation 
Author(s):
Bao, Fengqing; Arend, L.; Bertl, S.; Detlefsen, J. 
Title:
Application of FMCW radar principle for fast inhomogeneity identification on transmission lines 
Abstract:
A measurement system based on FMCW radar principle for fast inhomogeneity identification of a transmission line in short range application is presented. In comparison with traditional TDR method where the analysis of reflected waveform is directly processed in time domain, FMCW radar principle converts the analysis work into frequency domain and greatly reduces the hardware requirements consequently. Experiments show up both accurate detection results and fast detection ability. 
Book / Congress title:
Proc. German Microwave Conf. (GeMIC) 
Year:
2011 
Pages:
1--4, __ 
Reviewed:
ja 
Language:
en 
Publication format:
Print     
    """

    PreprocessBeta1Filter().filter(doc)
