import numpy as np


LETTERS_NODIA = "acdeeinorstuuyz"
LETTERS_NODIA_SINLE = "acdeinorstuyz"
LETTERS_DIA = "áčďéěíňóřšťúůýž"

def preprocess_data(data_str : str, surrounding : int):
    by_letter_dict = {}
    for letter in LETTERS_NODIA:
        by_letter_dict[letter] = []

    text_parts = windowify_text(data_str, surrounding)
    for part in text_parts:
        key_letter = chr(part[len(part)//2])
        by_letter_dict[key_letter].append(part)

    for letter in LETTERS_NODIA:
        by_letter_dict[letter] = np.array(by_letter_dict[letter])

    return by_letter_dict

def windowify_text(data_str : str, surrounding : int):
    data_str = data_str.lower()
    parts : list[list[int]] = []

    part : list[int] = []
    for i in range(surrounding):
        part.append(ord('.'))
    for j in data_str[:surrounding + 1]:
        part.append(ord(j))

    for i in range(surrounding + 1, len(data_str) + surrounding + 1):
        if chr(part[surrounding]) in LETTERS_NODIA:
            parts.append(part)
        next = [ord(data_str[i]) if i < len(data_str) else ord('.')]
        part = part[1:] + next

    return np.array(parts)

LETTERS_2OPT_0 = "acdinorstyz"
LETTERS_2OPT_1 = "áčďíňóřšťýž"
LETTERS_3OPT_0 = "eu"
LETTERS_3OPT_1 = "éú"
LETTERS_3OPT_2 = "ěů"
diacritics_dict = { 'á' : 'a', 'č' : 'c', 'ď' : 'd', 'í' : 'i', 'ň' : 'n', 
        'ó' : 'o', 'ř' : 'r', 'š' : 's', 'ť' : 't', 'ý' : 'y', 'ž' : 'z', 
        'é' : 'e', 'ě' : 'e', 'ú' : 'u', 'ů' : 'u'}
def preprocess_target(target_str : str):
    by_letter_dict = {}
    for letter in LETTERS_NODIA:
        by_letter_dict[letter] = []

    target_str = target_str.lower()
    for letter in target_str:
        if letter in LETTERS_2OPT_0:
            by_letter_dict[letter].append([1,0])
        elif letter in LETTERS_2OPT_1:
            by_letter_dict[diacritics_dict[letter]].append([0,1])
        elif letter in LETTERS_3OPT_0:
            by_letter_dict[letter].append([1,0,0])
        elif letter in LETTERS_3OPT_1:
            by_letter_dict[diacritics_dict[letter]].append([0,1,0])
        elif letter in LETTERS_3OPT_2:
            by_letter_dict[diacritics_dict[letter]].append([0,0,1])
    
    for letter in LETTERS_NODIA:
        by_letter_dict[letter] = np.array(by_letter_dict[letter])

    return by_letter_dict