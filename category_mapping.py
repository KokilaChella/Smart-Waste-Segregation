"""
category_mapping.py

Maps the 30 product-level classes from the Kaggle "Recyclable and Household
Waste Classification Dataset" to a smaller, practical disposal category:
recyclable / organic / non-recyclable / hazardous.

NOTE: Recycling rules vary a lot by municipality (e.g. some places accept
plastic straws curbside, most don't). These mappings follow common,
widely-cited general recycling guidance in the US/UK context. If you
present this project, mention that the mapping is a general reference,
not a substitute for local municipal recycling guidelines -- that's an
honest, sensible caveat that also shows you thought about the problem
rather than just slapping a label on everything.
"""

CLASS_TO_CATEGORY = {
    # --- Plastic ---
    "plastic_water_bottles":        "recyclable",
    "plastic_soda_bottles":         "recyclable",
    "plastic_detergent_bottles":    "recyclable",
    "plastic_shopping_bags":        "non-recyclable",   # most curbside programs reject thin film plastic
    "plastic_trash_bags":           "non-recyclable",
    "plastic_food_containers":      "recyclable",        # if rinsed; often accepted
    "disposable_plastic_cutlery":   "non-recyclable",
    "plastic_straws":               "non-recyclable",
    "plastic_cup_lids":             "non-recyclable",

    # --- Paper & cardboard ---
    "newspaper":                    "recyclable",
    "office_paper":                 "recyclable",
    "magazines":                    "recyclable",
    "cardboard_boxes":              "recyclable",
    "cardboard_packaging":          "recyclable",

    # --- Glass ---
    "glass_beverage_bottles":       "recyclable",
    "glass_food_jars":              "recyclable",
    "glass_cosmetic_containers":    "recyclable",

    # --- Metal ---
    "aluminum_soda_cans":           "recyclable",
    "aluminum_food_cans":           "recyclable",
    "steel_food_cans":              "recyclable",
    "aerosol_cans":                 "hazardous",         # pressurized / may contain residue

    # --- Organic ---
    "food_waste":                   "organic",
    "eggshells":                    "organic",
    "coffee_grounds":               "organic",
    "tea_bags":                     "organic",

    # --- Textile ---
    "clothing":                     "non-recyclable",    # needs textile-specific recycling/donation, not curbside
    "shoes":                        "non-recyclable",

    # --- Styrofoam / paper cups ---
    "styrofoam_cups":               "non-recyclable",
    "styrofoam_food_containers":    "non-recyclable",
    "paper_cups":                   "non-recyclable",    # usually plastic-lined, not curbside recyclable
}

# Sanity check: should be exactly 30 classes
assert len(CLASS_TO_CATEGORY) == 30, f"Expected 30 classes, got {len(CLASS_TO_CATEGORY)}"

CATEGORY_COLORS = {
    "recyclable":     "#2e7d32",   # green
    "organic":        "#8d6e63",   # brown
    "non-recyclable": "#616161",   # grey
    "hazardous":      "#c62828",   # red
}

CATEGORY_TIPS = {
    "recyclable": "Rinse if needed and place in your recycling bin.",
    "organic": "Compost if possible, or dispose of with food/green waste.",
    "non-recyclable": "Place in general waste — not accepted in most curbside recycling.",
    "hazardous": "Do not place in regular trash — take to a hazardous waste / household chemical drop-off point.",
}


def get_category(class_name: str) -> str:
    """Returns the disposal category for a given dataset class name."""
    return CLASS_TO_CATEGORY.get(class_name, "unknown")


def get_tip(class_name: str) -> str:
    category = get_category(class_name)
    return CATEGORY_TIPS.get(category, "Check your local recycling guidelines.")


if __name__ == "__main__":
    # Quick printout to sanity-check the mapping
    from collections import Counter
    counts = Counter(CLASS_TO_CATEGORY.values())
    print("Category distribution across 30 classes:")
    for cat, n in counts.items():
        print(f"  {cat:16s}: {n}")
