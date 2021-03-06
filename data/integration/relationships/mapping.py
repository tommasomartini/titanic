coarse_description_to_fine_description = {
    'parent': [
        'mother',
        'stepfather',
        'father',
    ],
    'child': [
        'daughter',
        'son',
        'stepdaughter',
    ],
    'sibling': [
        'half-brother',
        'brother?',
        'sister',
        'brother',
    ],
    'spouse': [
        'mistress',
        'fiancé',
        'wife',
        'lover',
        'fiancée',
        'husband',
    ],
    'friend': [
        'shared cabin',
        'travelling companion',
        'friend',
        'cabin companion',
        'cabin companion?',
        'colleague',
        'same cabin',
        'employers maid',
        'employers valet',
        'companion',
        'escort',
    ],
    'employee': [
        'maid',
        'chauffeur',
        'employee (chauffeur)',
        'wifes maid',
        'landlady',
        'nurse',
        'valet',
        'landlord',
        'fathers valet',
        'manservant',
        'governess',
        'nursmaid',
        'husbands valet',
        'nursemaid',
        'employee',
        'employee (cook)',
        'mothers maid',
        'childrens nurse',
    ],
    'employer': [
        'lodger',
        'employers',
        'employer',
        'employers son',
    ],
    'relative': [
        'grandmother',
        'related',
        'godchild',
        'godfather',
        'father-in-law',
        'sister-in-law',
        'nephew',
        '1043', # cousin
        'brother-in-law',
        'in-law',
        'aunt',
        'son-in-law',
        'brother-in-law?',
        'relative',
        'relative?',
        'great-aunt',
        'uncle',
        'distant cousin',
        'cousin',
        'niece',
    ],
    'knows': [],    # The most generic category.
}


_reciprocals = {
    'parent': 'child',
    'child': 'parent',
    'employee': 'employer',
    'employer': 'employee',
}


# Only for coarse relationships.
reciprocals = {
    rel: _reciprocals.get(rel, rel)
    for rel
    in coarse_description_to_fine_description.keys()
}


fine_description_to_coarse_description = {}
for coarse, fine_list in coarse_description_to_fine_description.items():
    for fine in fine_list:
        fine_description_to_coarse_description[fine] = coarse


relationship_type_to_coarse_description = {
    'childen': 'child',
    'knows': 'knows',
    'children': 'child',
    'relatedto': 'relative',
    'known': 'knows',
    'sibling': 'sibling',
    'parent': 'parent',
    'spouse': 'spouse',
    'colleague': 'friend',
}
